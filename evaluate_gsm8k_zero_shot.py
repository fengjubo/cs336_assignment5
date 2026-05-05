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
import sys
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


def get_reward_fn() -> Callable[..., dict[str, float]]:
    """Load the assignment's official r1_zero reward function."""
    try:
        from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import cs336_alignment.drgrpo_grader and its "
            "dependencies. Install the assignment grader dependencies in this "
            "environment, or run on the server environment where they are "
            "available."
        ) from exc
    return r1_zero_reward_fn


def format_prompts(examples: list[dict], prompt_template: str) -> list[str]:
    """Fill the r1_zero prompt template with each GSM8K question."""
    return [
        prompt_template.format(question=example["question"])
        for example in examples
    ]


def evaluate_vllm(
    vllm_model: Any,
    reward_fn: Callable[..., dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: Any,
    reward_fast: bool = True,
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
        example_metrics = reward_fn(
            response=generation_for_reward,
            ground_truth=ground_truth,
            fast=reward_fast,
        )

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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--slow-grader",
        action="store_true",
        help="Use the slower math_verify fallback by setting fast=False.",
    )
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
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    generations, metrics = evaluate_vllm(
        vllm_model=model,
        reward_fn=get_reward_fn(),
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        reward_fast=not args.slow_grader,
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



# conda run -n cs336 python homework/evaluate_gsm8k_zero_shot.py \
#   --data-path data/gsm8k/test.jsonl \
#   --model-name-or-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
#   --output-path results/gsm8k_qwen2_5_math_1_5b_zero_shot.jsonl \
#   --max-examples 5