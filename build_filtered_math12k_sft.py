# homework/build_filtered_math12k_sft.py

# conda run -n cs336 python homework/train_sft_vllm_eval_math12k.py \
#   --model-path /root/autodl-tmp/models/Qwen2.5-Math-1.5B \
#   --train-path data/math12k/filtered_reasoning_sft.jsonl \
#   --eval-path data/math12k/test-00000-of-00001.parquet \
#   --max-examples full \
#   --eval-max-examples 500 \
#   --eval-every 50

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B"
DEFAULT_DATA_PATH = REPO_ROOT / "data/math12k/train-00000-of-00001.parquet"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data/math12k/filtered_reasoning_sft.jsonl"
DEFAULT_PROMPT_PATH = REPO_ROOT / "cs336_alignment/prompts/r1_zero.prompt"


def parse_optional_int(value: str) -> int | None:
    if value == "full":
        return None
    return int(value)


def load_examples(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")

    if path.suffix == ".jsonl":
        examples = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        return examples

    raise ValueError(f"Unsupported data format for {path}. Expected .parquet or .jsonl.")


def validate_columns(
    examples: list[dict[str, Any]],
    required_columns: list[str],
    path: Path,
) -> None:
    if not examples:
        return

    missing_columns = [
        column for column in required_columns if column not in examples[0]
    ]
    if missing_columns:
        available_columns = sorted(examples[0].keys())
        raise ValueError(
            f"{path} is missing columns {missing_columns}. "
            f"Available columns: {available_columns}"
        )


def to_jsonable(value: Any) -> Any:
    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()

    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [to_jsonable(item) for item in value]

    return value


def build_prompts(
    examples: list[dict[str, Any]],
    prompt_template: str,
    question_column: str,
) -> list[str]:
    return [
        prompt_template.format(question=example[question_column])
        for example in examples
    ]


def filter_correct_generations(
    examples: list[dict[str, Any]],
    prompts: list[str],
    raw_outputs: list[Any],
    question_column: str,
    answer_column: str,
    model_path: str,
    reward_fn: Any,
    fast_grader: bool,
) -> list[dict[str, Any]]:
    filtered_records = []

    for source_index, (example, prompt, output) in tqdm(
        enumerate(zip(examples, prompts, raw_outputs)),
        total=len(examples),
        desc="Filtering correct generations",
    ):
        generation = output.outputs[0].text.strip()
        response_for_reward = "<think>" + generation
        ground_truth = str(example[answer_column]).strip()

        reward_info = reward_fn(
            response=response_for_reward,
            ground_truth=ground_truth,
            fast=fast_grader,
        )
        reward_info = to_jsonable(reward_info)
        is_correct = bool(reward_info["answer_reward"] > 0)

        if not is_correct:
            continue

        filtered_records.append(
            {
                "problem": str(example[question_column]),
                "answer": ground_truth,
                "prompt": prompt,
                "response": generation,
                "model_generation": generation,
                "reward_info": reward_info,
                "is_correct": is_correct,
                "source_index": source_index,
                "model_name_or_path": model_path,
            }
        )

    return filtered_records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a filtered MATH12k reasoning SFT dataset."
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--question-column", type=str, default="problem")
    parser.add_argument("--answer-column", type=str, default="answer")
    parser.add_argument("--max-examples", type=str, default="full")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--slow-grader",
        action="store_true",
        help="Use slower math_verify fallback by setting fast=False.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_examples = parse_optional_int(args.max_examples)

    random.seed(args.seed)

    examples = load_examples(args.data_path)
    validate_columns(
        examples=examples,
        required_columns=[args.question_column, args.answer_column],
        path=args.data_path,
    )

    if max_examples is not None:
        examples = examples[:max_examples]

    prompt_template = args.prompt_path.read_text(encoding="utf-8").strip()
    prompts = build_prompts(
        examples=examples,
        prompt_template=prompt_template,
        question_column=args.question_column,
    )

    from vllm import LLM, SamplingParams
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    vllm_set_random_seed(args.seed)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.num_gpus,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    raw_outputs = llm.generate(prompts, sampling_params)
    filtered_records = filter_correct_generations(
        examples=examples,
        prompts=prompts,
        raw_outputs=raw_outputs,
        question_column=args.question_column,
        answer_column=args.answer_column,
        model_path=args.model_path,
        reward_fn=r1_zero_reward_fn,
        fast_grader=not args.slow_grader,
    )
    write_jsonl(args.output_path, filtered_records)

    total_examples = len(examples)
    correct_examples = len(filtered_records)
    filtered_accuracy = correct_examples / total_examples if total_examples else 0.0

    print("=" * 80)
    print(f"total_examples={total_examples}")
    print(f"correct_examples={correct_examples}")
    print(f"filtered_accuracy={filtered_accuracy:.6f}")
    print(f"output_path={args.output_path}")


if __name__ == "__main__":
    main()
