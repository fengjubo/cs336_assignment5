# homework/evaluate_sft.py

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def load_jsonl(path: str, max_examples: int | None = None, seed: int = 0):
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    random.Random(seed).shuffle(examples)

    if max_examples is not None:
        examples = examples[:max_examples]

    return examples


def load_prompt_template(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def get_prompt_and_ground_truth(item: dict, prompt_template: str):
    prompt = prompt_template.format(question=item["question"])
    ground_truth = extract_gsm8k_final_answer(item["answer"])
    return prompt, ground_truth


def extract_gsm8k_final_answer(answer: str) -> str:
    """Extract the final GSM8K answer after the #### marker."""
    marker = "####"
    if marker in answer:
        return answer.split(marker)[-1].strip()
    return answer.strip()


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    examples,
    prompt_template: str,
    device: str,
    max_new_tokens: int = 512,
    debug_examples: int = 3,
    fast: bool = True,
):
    model.eval()

    correct = 0
    total = 0
    records = []

    for item in tqdm(examples):
        prompt, ground_truth = get_prompt_and_ground_truth(
            item=item,
            prompt_template=prompt_template,
        )

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

        response_for_reward = "<think>" + response
        reward_info = r1_zero_reward_fn(
            response=response_for_reward,
            ground_truth=ground_truth,
            fast=fast,
        )

        is_correct = bool(reward_info["answer_reward"] > 0)

        if total < debug_examples:
            print("=" * 80)
            print("PROMPT:")
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--eval-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=(
            "/root/autodl-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
        ),
    )

    parser.add_argument("--max-examples", type=str, default="100")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug-examples", type=int, default=3)

    parser.add_argument(
        "--slow-grader",
        action="store_true",
        help="Use slower math_verify fallback by setting fast=False.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_examples == "full":
        max_examples = None
    else:
        max_examples = int(args.max_examples)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading model from {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    prompt_template = load_prompt_template(args.prompt_path)

    examples = load_jsonl(
        path=args.eval_path,
        max_examples=max_examples,
        seed=args.seed,
    )

    print(f"Number of eval examples: {len(examples)}")

    result = evaluate(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        prompt_template=prompt_template,
        device=device,
        max_new_tokens=args.max_new_tokens,
        debug_examples=args.debug_examples,
        fast=not args.slow_grader,
    )

    summary = {
        "model_path": args.model_path,
        "eval_path": args.eval_path,
        "prompt_path": args.prompt_path,
        "accuracy": result["accuracy"],
        "correct": result["correct"],
        "total": result["total"],
        "max_examples": args.max_examples,
        "max_new_tokens": args.max_new_tokens,
        "fast_grader": not args.slow_grader,
    }

    print("=" * 80)
    print("SUMMARY:")
    print(summary)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **summary,
                "records": result["records"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
