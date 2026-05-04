"""
Evaluate Qwen2.5-Math-1.5B zero-shot performance on GSM8K with vLLM.

Example:
    conda run -n cs336 python homework/evaluate_gsm8k_zero_shot.py \
        --data-path data/gsm8k/test.jsonl \
        --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
        --output-path results/gsm8k_qwen2_5_math_1_5b_zero_shot.jsonl
"""

import argparse
import json
import logging
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))


logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dictionaries."""
    with path.open() as f:
        return [json.loads(line) for line in f]


def load_prompt_template(path: Path) -> str:
    """Read the prompt template used to turn each question into a model prompt."""
    return path.read_text().strip()


def extract_gsm8k_answer(answer: str) -> str:
    """
    Extract GSM8K's final answer.

    GSM8K answers end with a marker like:
        #### 18

    The grader only needs the final target answer, so we keep the text after
    the final marker instead of the full chain-of-thought solution.
    """
    marker = "####"
    if marker not in answer:
        return answer.strip()
    return answer.split(marker)[-1].strip()


def extract_r1_answer(response: str) -> str | None:
    """Extract the text inside <answer>...</answer> from an r1-style response."""
    match = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def normalize_numeric_answer(answer: str) -> str:
    """Normalize common GSM8K final-answer formatting before exact comparison."""
    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = answer.rstrip(".")

    number_match = re.search(r"-?\d+(?:\.\d+)?", answer)
    if number_match is not None:
        answer = number_match.group(0)

    try:
        decimal = Decimal(answer)
    except InvalidOperation:
        return answer
    return str(decimal.normalize())


def gsm8k_r1_reward_fn(response: str, ground_truth: str) -> dict[str, float]:
    """
    Lightweight GSM8K reward function.

    It enforces the same r1 output tags as the assignment prompt, then compares
    the normalized final number inside <answer>...</answer> to the GSM8K target.
    """
    model_answer = extract_r1_answer(response)
    is_formatted = "</think>" in response and model_answer is not None
    is_correct = (
        is_formatted
        and normalize_numeric_answer(model_answer) == normalize_numeric_answer(ground_truth)
    )
    return {
        "format_reward": float(is_formatted),
        "answer_reward": float(is_correct),
        "reward": float(is_correct),
    }


def get_reward_fn() -> Callable[[str, str], dict[str, float]]:
    """
    Prefer the assignment's robust math grader, with a local GSM8K fallback.

    The fallback keeps this script usable in lightweight environments where the
    optional symbolic-math grader dependencies are not installed.
    """
    try:
        from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
    except ModuleNotFoundError:
        logger.warning(
            "Could not import cs336_alignment.drgrpo_grader; falling back to "
            "the local GSM8K numeric grader."
        )
        return gsm8k_r1_reward_fn
    return r1_zero_reward_fn


def format_prompts(examples: list[dict], prompt_template: str) -> list[str]:
    """Fill the r1_zero prompt template with each GSM8K question."""
    return [
        prompt_template.format(question=example["question"])
        for example in examples
    ]


def evaluate_vllm(
    vllm_model: Any,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: Any,
) -> tuple[list[str], list[dict[str, float]]]:
    """
    Evaluate a language model on a list of prompts.

    This function is intentionally reusable:
    1. vLLM generates one response per prompt.
    2. reward_fn compares each generated response with its ground truth.
    3. The function returns both raw generations and per-example metrics.
    """
    raw_outputs = vllm_model.generate(prompts, eval_sampling_params)

    generations = []
    metrics = []
    for output, ground_truth in tqdm(
        zip(raw_outputs, ground_truths),
        total=len(ground_truths),
        desc="Scoring generations",
    ):
        generation = output.outputs[0].text.strip()

        # r1_zero.prompt already ends with "Assistant: <think>". The generated
        # text starts after that token, but the reward function expects the full
        # assistant-side text when checking for </think> <answer>...</answer>.
        generation_for_reward = "<think>" + generation
        example_metrics = reward_fn(generation_for_reward, ground_truth)

        generations.append(generation)
        metrics.append(example_metrics)

    return generations, metrics


def summarize_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average each metric key over all examples."""
    if not metrics:
        return {}
    return {
        key: mean(example_metrics[key] for example_metrics in metrics)
        for key in sorted(metrics[0])
    }


def write_results(
    output_path: Path,
    examples: list[dict],
    prompts: list[str],
    ground_truths: list[str],
    generations: list[str],
    metrics: list[dict[str, float]],
    model_name_or_path: str,
) -> None:
    """Serialize per-example inputs, generations, and scores as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for example, prompt, ground_truth, generation, example_metrics in zip(
            examples,
            prompts,
            ground_truths,
            generations,
            metrics,
        ):
            record = {
                **example,
                "ground_truth": ground_truth,
                "model_name_or_path": model_name_or_path,
                "model_prompt": prompt,
                "model_generation": generation,
                "metrics": example_metrics,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot Qwen2.5-Math-1.5B on GSM8K."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data" / "gsm8k" / "test.jsonl",
        help="Path to the GSM8K JSONL split.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=REPO_ROOT / "cs336_alignment" / "prompts" / "r1_zero.prompt",
        help="Path to the r1_zero prompt template.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Hugging Face model name or a local model path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "results" / "gsm8k_qwen2_5_math_1_5b_zero_shot.jsonl",
        help="Where to write per-example evaluation results.",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional small-run limit for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    examples = load_jsonl(args.data_path)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    logger.info("Loaded %d examples from %s", len(examples), args.data_path)

    prompt_template = load_prompt_template(args.prompt_path)
    prompts = format_prompts(examples, prompt_template)
    ground_truths = [extract_gsm8k_answer(example["answer"]) for example in examples]

    try:
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "vLLM is required to run model evaluation. Install it in the cs336 "
            "environment, or run this script on a machine/environment where "
            "`python -c 'import vllm'` works."
        ) from exc

    model = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.num_gpus,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    generations, metrics = evaluate_vllm(
        vllm_model=model,
        reward_fn=get_reward_fn(),
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
    )
    write_results(
        output_path=args.output_path,
        examples=examples,
        prompts=prompts,
        ground_truths=ground_truths,
        generations=generations,
        metrics=metrics,
        model_name_or_path=args.model_name_or_path,
    )

    summary = summarize_metrics(metrics)
    for key, value in summary.items():
        logger.info("%s: %.4f", key, value)
    logger.info("Wrote per-example results to %s", args.output_path)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger.info("running %s", " ".join(sys.argv))
    main()
    logger.info("finished running %s", sys.argv[0])
