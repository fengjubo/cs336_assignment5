# train_sft.py

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW

# homework/train_sft.py
from homework.helpers import tokenize_prompt_and_output, get_response_log_probs
from homework.sft_train_step import sft_microbatch_train_step



@dataclass
class SFTConfig:
    model_path: str = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B"
    train_path: str = "/root/autodl-tmp/assignment5-alignment-main/data/gsm8k/train.jsonl"
    prompt_path: str = (
        "/root/autodl-tmp/assignment5-alignment-main/"
        "cs336_alignment/prompts/r1_zero.prompt"
    )
    output_dir: str = "./checkpoints/sft_debug_128"

    max_examples: int | None = 128

    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    num_epochs: int = 1
    max_grad_norm: float = 1.0

    seed: int = 0
    log_every: int = 1

    sample_generations: int = 3
    sample_max_new_tokens: int = 256


class SFTDataset(Dataset):
    def __init__(
        self,
        path: str,
        prompt_path: str,
        max_examples: int | None = None,
        seed: int = 0,
    ):
        self.examples = []
        self.prompt_template = Path(prompt_path).read_text(
            encoding="utf-8",
        ).strip()

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

        random.Random(seed).shuffle(self.examples)

        if max_examples is not None:
            self.examples = self.examples[:max_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        item = self.examples[idx]
        reasoning, final_answer = format_gsm8k_r1_response_parts(item["answer"])

        return {
            "prompt": self.prompt_template.format(question=item["question"]),
            "response": f"{reasoning}\n</think> <answer>{final_answer}</answer>",
        }


def format_gsm8k_r1_response_parts(answer: str) -> tuple[str, str]:
    marker = "####"
    if marker not in answer:
        raise ValueError(f"GSM8K answer is missing final-answer marker {marker!r}")

    reasoning, final_answer = answer.rsplit(marker, maxsplit=1)
    return reasoning.strip(), final_answer.strip()


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


def train_sft(cfg: SFTConfig):
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading tokenizer from {cfg.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {cfg.model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    model.train()

    train_dataset = SFTDataset(
        path=cfg.train_path,
        prompt_path=cfg.prompt_path,
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

    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Gradient accumulation steps: {cfg.gradient_accumulation_steps}")

    global_step = 0
    micro_step = 0

    optimizer.zero_grad()

    for epoch in range(cfg.num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            micro_step += 1

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

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
                        f"loss={loss.item():.6f} "
                        f"loss_before_accum="
                        f"{metadata['loss_before_grad_accum'].item():.6f} "
                        f"num_response_tokens="
                        f"{metadata['num_response_tokens'].item()}"
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
            "applied final partial gradient accumulation step"
        )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Saving model to {cfg.output_dir}")

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print_sample_generations(
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        device=device,
        num_examples=cfg.sample_generations,
        max_new_tokens=cfg.sample_max_new_tokens,
    )

    print("Done.")

    return model, tokenizer


if __name__ == "__main__":
    cfg = SFTConfig()
    train_sft(cfg)
