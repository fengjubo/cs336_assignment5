# homework/train_sft_vllm_eval_math12k.py

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from homework.helpers import get_response_log_probs, tokenize_prompt_and_output
from homework.sft_train_step import sft_microbatch_train_step


@dataclass
class SFTVLLMEvalConfig:
    model_path: str = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B"
    train_path: str = str(REPO_ROOT / "data/math12k/train-00000-of-00001.parquet")
    eval_path: str | None = str(REPO_ROOT / "data/math12k/test-00000-of-00001.parquet")
    prompt_path: str = str(REPO_ROOT / "cs336_alignment/prompts/r1_zero.prompt")
    question_column: str = "problem"
    answer_column: str = "answer"

    max_examples: int | None = 128
    eval_max_examples: int | None = 500

    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    num_epochs: int = 1
    max_grad_norm: float = 1.0

    seed: int = 0
    log_every: int = 1
    eval_every: int = 50

    sample_generations: int = 3
    sample_max_new_tokens: int = 256
    eval_max_new_tokens: int = 512

    policy_device: str = "cuda:0"
    vllm_device: str = "cuda:1"
    vllm_gpu_memory_utilization: float = 0.85
    slow_grader: bool = False

    use_wandb: bool = False
    wandb_project: str = "cs336-sft"
    wandb_run_name: str | None = None


class SFTDataset(Dataset):
    def __init__(
        self,
        path: str,
        prompt_path: str,
        question_column: str = "problem",
        answer_column: str = "answer",
        max_examples: int | None = None,
        seed: int = 0,
    ):
        self.question_column = question_column
        self.answer_column = answer_column
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8").strip()
        self.examples = load_examples(
            path=path,
            max_examples=None,
            seed=seed,
        )
        validate_columns(
            examples=self.examples,
            required_columns=[question_column, answer_column],
            path=path,
        )

        random.Random(seed).shuffle(self.examples)

        if max_examples is not None:
            self.examples = self.examples[:max_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        item = self.examples[idx]
        question = item[self.question_column]
        final_answer = extract_math12k_final_answer(item[self.answer_column])

        return {
            "prompt": self.prompt_template.format(question=question),
            "response": f"</think> <answer>{final_answer}</answer>",
        }


def extract_math12k_final_answer(answer: Any) -> str:
    return str(answer).strip()


def load_examples(path: str, max_examples: int | None = None, seed: int = 0):
    data_path = Path(path)

    if data_path.suffix == ".parquet":
        import pandas as pd

        examples = pd.read_parquet(data_path).to_dict(orient="records")
    elif data_path.suffix == ".jsonl":
        examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    else:
        raise ValueError(
            f"Unsupported data format for {path!r}. Expected .parquet or .jsonl."
        )

    random.Random(seed).shuffle(examples)

    if max_examples is not None:
        examples = examples[:max_examples]

    return examples


def validate_columns(examples: list[dict], required_columns: list[str], path: str):
    if not examples:
        return

    missing_columns = [
        column for column in required_columns if column not in examples[0]
    ]

    if missing_columns:
        available_columns = sorted(examples[0].keys())
        raise ValueError(
            f"{path!r} is missing columns {missing_columns}. "
            f"Available columns: {available_columns}"
        )


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        prompt_strs = [x["prompt"] for x in batch]
        output_strs = [x["response"] for x in batch]

        return tokenize_prompt_and_output(
            prompt_strs=prompt_strs,
            output_strs=output_strs,
            tokenizer=tokenizer,
        )

    return collate_fn


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
):
    """
    Start a vLLM inference process on a GPU separate from the policy model.
    """
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm):
    """
    Copy policy weights into the existing vLLM instance before evaluation.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def load_eval_prompts_and_ground_truths(
    eval_path: str,
    prompt_path: str,
    question_column: str,
    answer_column: str,
    max_examples: int | None,
    seed: int,
) -> tuple[list[dict], list[str], list[str]]:
    prompt_template = Path(prompt_path).read_text(encoding="utf-8").strip()
    examples = load_examples(
        path=eval_path,
        max_examples=max_examples,
        seed=seed,
    )
    validate_columns(
        examples=examples,
        required_columns=[question_column, answer_column],
        path=eval_path,
    )

    prompts = [
        prompt_template.format(question=example[question_column])
        for example in examples
    ]
    ground_truths = [
        extract_math12k_final_answer(example[answer_column])
        for example in examples
    ]

    return examples, prompts, ground_truths


@torch.no_grad()
def evaluate_with_vllm(
    llm,
    sampling_params: Any,
    prompts: list[str],
    ground_truths: list[str],
    debug_examples: int = 3,
    fast: bool = True,
) -> dict[str, Any]:
    raw_outputs = llm.generate(prompts, sampling_params)

    correct = 0
    total = 0
    records = []

    for prompt, ground_truth, output in tqdm(
        zip(prompts, ground_truths, raw_outputs),
        total=len(ground_truths),
        desc="Scoring vLLM generations",
    ):
        response = output.outputs[0].text.strip()
        response_for_reward = "<think>" + response

        reward_info = r1_zero_reward_fn(
            response=response_for_reward,
            ground_truth=ground_truth,
            fast=fast,
        )
        is_correct = bool(reward_info["answer_reward"] > 0)

        if total < debug_examples:
            print("=" * 80)
            print("EVAL PROMPT:")
            print(prompt)
            print("\nGROUND TRUTH:")
            print(ground_truth)
            print("\nMODEL RESPONSE:")
            print(response)
            print("\nREWARD INFO:")
            print(reward_info)
            print("\nIS CORRECT:")
            print(is_correct)

        records.append(
            {
                "prompt": prompt,
                "ground_truth": ground_truth,
                "response": response,
                "response_for_reward": response_for_reward,
                "reward_info": reward_info,
                "is_correct": is_correct,
            }
        )

        correct += int(is_correct)
        total += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "records": records,
    }


@torch.no_grad()
def print_sample_generations(
    model,
    tokenizer,
    dataset: SFTDataset,
    device: str,
    num_examples: int,
    max_new_tokens: int,
):
    if num_examples <= 0 or len(dataset) == 0:
        return

    model.eval()

    print("=" * 80)
    print("Sample generations after SFT:")

    for idx in range(min(num_examples, len(dataset))):
        example = dataset[idx]
        prompt = example["prompt"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]
        response_ids = output_ids[0, prompt_len:]
        response = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        )

        print("=" * 80)
        print(f"SAMPLE {idx + 1}")
        print("PROMPT:")
        print(prompt)
        print("\nMODEL OUTPUT:")
        print(response)


def maybe_init_wandb(cfg: SFTVLLMEvalConfig):
    if not cfg.use_wandb:
        return None

    import wandb

    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        config=cfg.__dict__,
    )

    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    return run


def maybe_log_wandb(wandb_run, metrics: dict[str, Any]):
    if wandb_run is None:
        return

    import wandb

    wandb.log(metrics)


def run_periodic_eval(
    model,
    llm,
    sampling_params,
    eval_prompts: list[str],
    eval_ground_truths: list[str],
    global_step: int,
    eval_step: int,
    cfg: SFTVLLMEvalConfig,
    wandb_run,
) -> int:
    print("=" * 80)
    print(f"Loading policy weights into vLLM at train_step={global_step}")
    load_policy_into_vllm_instance(policy=model, llm=llm)

    result = evaluate_with_vllm(
        llm=llm,
        sampling_params=sampling_params,
        prompts=eval_prompts,
        ground_truths=eval_ground_truths,
        debug_examples=0,
        fast=not cfg.slow_grader,
    )

    eval_step += 1
    print(
        f"eval_step={eval_step} "
        f"train_step={global_step} "
        f"accuracy={result['accuracy']:.6f} "
        f"correct={result['correct']} "
        f"total={result['total']}"
    )

    maybe_log_wandb(
        wandb_run,
        {
            "eval_step": eval_step,
            "train_step": global_step,
            "eval/accuracy": result["accuracy"],
            "eval/correct": result["correct"],
            "eval/total": result["total"],
        },
    )

    return eval_step


def train_sft_with_vllm_eval(cfg: SFTVLLMEvalConfig):
    set_seed(cfg.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("This script expects CUDA for two-GPU SFT + vLLM eval.")

    if cfg.eval_every > 0 and cfg.eval_path is None:
        raise ValueError("--eval-path is required when --eval-every > 0.")

    print(f"Using policy device: {cfg.policy_device}")
    print(f"Using vLLM device: {cfg.vllm_device}")
    print(f"Loading tokenizer from {cfg.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading policy model from {cfg.model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(cfg.policy_device)

    model.train()

    train_dataset = SFTDataset(
        path=cfg.train_path,
        prompt_path=cfg.prompt_path,
        question_column=cfg.question_column,
        answer_column=cfg.answer_column,
        max_examples=cfg.max_examples,
        seed=cfg.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
    )

    eval_prompts = []
    eval_ground_truths = []
    llm = None
    sampling_params = None

    if cfg.eval_every > 0:
        from vllm import SamplingParams

        _, eval_prompts, eval_ground_truths = load_eval_prompts_and_ground_truths(
            eval_path=cfg.eval_path,
            prompt_path=cfg.prompt_path,
            question_column=cfg.question_column,
            answer_column=cfg.answer_column,
            max_examples=cfg.eval_max_examples,
            seed=cfg.seed,
        )

        print(f"Number of eval examples: {len(eval_prompts)}")
        print(f"Initializing vLLM from {cfg.model_path}")
        llm = init_vllm(
            model_id=cfg.model_path,
            device=cfg.vllm_device,
            seed=cfg.seed,
            gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=cfg.eval_max_new_tokens,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )

    wandb_run = maybe_init_wandb(cfg)

    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Gradient accumulation steps: {cfg.gradient_accumulation_steps}")
    print(f"Eval every optimizer steps: {cfg.eval_every}")

    global_step = 0
    eval_step = 0
    micro_step = 0
    examples_seen = 0

    optimizer.zero_grad()

    for epoch in range(cfg.num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            micro_step += 1
            examples_seen += batch["input_ids"].shape[0]

            input_ids = batch["input_ids"].to(cfg.policy_device)
            labels = batch["labels"].to(cfg.policy_device)
            response_mask = batch["response_mask"].to(cfg.policy_device)

            outputs = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
            )

            policy_log_probs = outputs["log_probs"]
            normalize_constant = response_mask.sum().item()

            if normalize_constant == 0:
                raise ValueError("response_mask has no True tokens.")

            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                normalize_constant=normalize_constant,
            )

            should_update = micro_step % cfg.gradient_accumulation_steps == 0

            if should_update:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.max_grad_norm,
                )

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % cfg.log_every == 0:
                    print(
                        f"epoch={epoch} "
                        f"global_step={global_step} "
                        f"micro_step={micro_step} "
                        f"examples_seen={examples_seen} "
                        f"loss={loss.item():.6f} "
                        f"loss_before_accum="
                        f"{metadata['loss_before_grad_accum'].item():.6f} "
                        f"num_response_tokens="
                        f"{metadata['num_response_tokens'].item()}"
                    )

                    maybe_log_wandb(
                        wandb_run,
                        {
                            "train_step": global_step,
                            "train/loss": loss.item(),
                            "train/loss_before_grad_accum": metadata[
                                "loss_before_grad_accum"
                            ].item(),
                            "train/num_response_tokens": metadata[
                                "num_response_tokens"
                            ].item(),
                            "train/examples_seen": examples_seen,
                        },
                    )

                if (
                    cfg.eval_every > 0
                    and global_step % cfg.eval_every == 0
                    and llm is not None
                    and sampling_params is not None
                ):
                    eval_step = run_periodic_eval(
                        model=model,
                        llm=llm,
                        sampling_params=sampling_params,
                        eval_prompts=eval_prompts,
                        eval_ground_truths=eval_ground_truths,
                        global_step=global_step,
                        eval_step=eval_step,
                        cfg=cfg,
                        wandb_run=wandb_run,
                    )

    if micro_step > 0 and micro_step % cfg.gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        print(
            f"global_step={global_step} "
            f"micro_step={micro_step} "
            f"examples_seen={examples_seen} "
            "applied final partial gradient accumulation step"
        )

        if (
            cfg.eval_every > 0
            and global_step % cfg.eval_every == 0
            and llm is not None
            and sampling_params is not None
        ):
            eval_step = run_periodic_eval(
                model=model,
                llm=llm,
                sampling_params=sampling_params,
                eval_prompts=eval_prompts,
                eval_ground_truths=eval_ground_truths,
                global_step=global_step,
                eval_step=eval_step,
                cfg=cfg,
                wandb_run=wandb_run,
            )

    print_sample_generations(
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        device=cfg.policy_device,
        num_examples=cfg.sample_generations,
        max_new_tokens=cfg.sample_max_new_tokens,
    )

    if wandb_run is not None:
        wandb_run.finish()

    print("Done.")

    return model, tokenizer


def parse_optional_int(value: str) -> int | None:
    if value == "full":
        return None
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default=SFTVLLMEvalConfig.model_path)
    parser.add_argument("--train-path", type=str, default=SFTVLLMEvalConfig.train_path)
    parser.add_argument("--eval-path", type=str, default=SFTVLLMEvalConfig.eval_path)
    parser.add_argument("--prompt-path", type=str, default=SFTVLLMEvalConfig.prompt_path)
    parser.add_argument(
        "--question-column",
        type=str,
        default=SFTVLLMEvalConfig.question_column,
        help="Column containing MATH12k problem text.",
    )
    parser.add_argument(
        "--answer-column",
        type=str,
        default=SFTVLLMEvalConfig.answer_column,
        help="Column containing MATH12k final answers.",
    )
    parser.add_argument(
        "--max-examples",
        type=str,
        default=str(SFTVLLMEvalConfig.max_examples),
    )
    parser.add_argument(
        "--eval-max-examples",
        type=str,
        default=str(SFTVLLMEvalConfig.eval_max_examples),
    )

    parser.add_argument("--batch-size", type=int, default=SFTVLLMEvalConfig.batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=SFTVLLMEvalConfig.gradient_accumulation_steps,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=SFTVLLMEvalConfig.learning_rate,
    )
    parser.add_argument("--num-epochs", type=int, default=SFTVLLMEvalConfig.num_epochs)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=SFTVLLMEvalConfig.max_grad_norm,
    )

    parser.add_argument("--seed", type=int, default=SFTVLLMEvalConfig.seed)
    parser.add_argument("--log-every", type=int, default=SFTVLLMEvalConfig.log_every)
    parser.add_argument("--eval-every", type=int, default=SFTVLLMEvalConfig.eval_every)

    parser.add_argument(
        "--sample-generations",
        type=int,
        default=SFTVLLMEvalConfig.sample_generations,
    )
    parser.add_argument(
        "--sample-max-new-tokens",
        type=int,
        default=SFTVLLMEvalConfig.sample_max_new_tokens,
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=SFTVLLMEvalConfig.eval_max_new_tokens,
    )

    parser.add_argument(
        "--policy-device",
        type=str,
        default=SFTVLLMEvalConfig.policy_device,
    )
    parser.add_argument(
        "--vllm-device",
        type=str,
        default=SFTVLLMEvalConfig.vllm_device,
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=SFTVLLMEvalConfig.vllm_gpu_memory_utilization,
    )
    parser.add_argument(
        "--slow-grader",
        action="store_true",
        help="Use slower math_verify fallback by setting fast=False.",
    )

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=SFTVLLMEvalConfig.wandb_project,
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=SFTVLLMEvalConfig.wandb_run_name,
    )

    return parser.parse_args()


def config_from_args(args) -> SFTVLLMEvalConfig:
    if args.log_every <= 0:
        raise ValueError("--log-every must be a positive integer.")

    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")

    return SFTVLLMEvalConfig(
        model_path=args.model_path,
        train_path=args.train_path,
        eval_path=args.eval_path,
        prompt_path=args.prompt_path,
        question_column=args.question_column,
        answer_column=args.answer_column,
        max_examples=parse_optional_int(args.max_examples),
        eval_max_examples=parse_optional_int(args.eval_max_examples),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        sample_generations=args.sample_generations,
        sample_max_new_tokens=args.sample_max_new_tokens,
        eval_max_new_tokens=args.eval_max_new_tokens,
        policy_device=args.policy_device,
        vllm_device=args.vllm_device,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        slow_grader=args.slow_grader,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    cfg = config_from_args(parse_args())
    train_sft_with_vllm_eval(cfg)
