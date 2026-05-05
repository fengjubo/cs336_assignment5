# homework/draw.py

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))


CHECKPOINT_RE = re.compile(r"^checkpoint-step-(\d+)$")
LEGACY_CHECKPOINT_RE = re.compile(r"^checkpoint-example-(\d+)$")
DEFAULT_PROMPT_PATH = (
    "/root/autodl-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--eval-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=DEFAULT_PROMPT_PATH,
    )

    parser.add_argument("--max-examples", type=str, default="100")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug-examples", type=int, default=0)

    parser.add_argument(
        "--slow-grader",
        action="store_true",
        help="Use slower math_verify fallback by setting fast=False.",
    )

    return parser.parse_args()


def parse_max_examples(value: str) -> int | None:
    if value == "full":
        return None
    return int(value)


def find_checkpoints(checkpoint_dir: str | Path) -> list[tuple[str, int, Path]]:
    checkpoint_dir = Path(checkpoint_dir)

    checkpoints = []
    for path in checkpoint_dir.iterdir():
        if not path.is_dir():
            continue

        match = CHECKPOINT_RE.match(path.name)
        if match is not None:
            checkpoints.append(("step", int(match.group(1)), path))
            continue

        legacy_match = LEGACY_CHECKPOINT_RE.match(path.name)
        if legacy_match is not None:
            checkpoints.append(("example", int(legacy_match.group(1)), path))

    checkpoints.sort(key=lambda item: (item[0], item[1]))

    if not checkpoints:
        raise ValueError(
            f"No checkpoint-step-* or checkpoint-example-* directories found in "
            f"{checkpoint_dir}"
        )

    return checkpoints


def load_model_and_tokenizer(checkpoint_path: Path, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading checkpoint from {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    return model, tokenizer


def save_json(path: str | Path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def plot_accuracy_curve(summary_records: list[dict], output_path: str | Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = [record["checkpoint_index"] for record in summary_records]
    accuracies = [record["accuracy"] for record in summary_records]
    x_label = summary_records[0]["checkpoint_unit"]
    if x_label == "step":
        x_label = "Optimizer step"
    else:
        x_label = "Training examples processed"

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, accuracies, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("Validation accuracy")
    plt.title("SFT Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()

    import torch

    from homework.evaluate_sft import evaluate, load_jsonl, load_prompt_template

    max_examples = parse_max_examples(args.max_examples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading eval data from {args.eval_path}")

    prompt_template = load_prompt_template(args.prompt_path)
    examples = load_jsonl(
        path=args.eval_path,
        max_examples=max_examples,
        seed=args.seed,
    )

    print(f"Number of eval examples: {len(examples)}")

    checkpoints = find_checkpoints(args.checkpoint_dir)
    print(f"Found {len(checkpoints)} checkpoints")

    summary_records = []

    for checkpoint_unit, checkpoint_index, checkpoint_path in checkpoints:
        model, tokenizer = load_model_and_tokenizer(
            checkpoint_path=checkpoint_path,
            device=device,
        )

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

        record = {
            "checkpoint_unit": checkpoint_unit,
            "checkpoint_index": checkpoint_index,
            "checkpoint_path": str(checkpoint_path),
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
        }

        print("=" * 80)
        print("CHECKPOINT SUMMARY:")
        print(record)

        detail_path = output_dir / f"eval_{checkpoint_path.name}.json"
        save_json(
            detail_path,
            {
                **record,
                "eval_path": args.eval_path,
                "prompt_path": args.prompt_path,
                "max_examples": args.max_examples,
                "max_new_tokens": args.max_new_tokens,
                "fast_grader": not args.slow_grader,
                "records": result["records"],
            },
        )

        summary_records.append(record)

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "checkpoint_dir": args.checkpoint_dir,
        "eval_path": args.eval_path,
        "prompt_path": args.prompt_path,
        "max_examples": args.max_examples,
        "max_new_tokens": args.max_new_tokens,
        "fast_grader": not args.slow_grader,
        "results": summary_records,
    }

    summary_path = output_dir / "summary.json"
    save_json(summary_path, summary)

    plot_path = output_dir / "validation_accuracy_curve.png"
    plot_accuracy_curve(summary_records, plot_path)

    print(f"Saved summary to {summary_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
