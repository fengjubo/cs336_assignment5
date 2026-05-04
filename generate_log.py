# generation_logging.py

from typing import Any, Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from helpers import compute_entropy


def _to_python(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            return x.item()
        return x
    return x


def _normalize_reward_info(reward_info: Any) -> dict[str, Any]:
    if reward_info is None:
        return {}

    if isinstance(reward_info, dict):
        return {k: _to_python(v) for k, v in reward_info.items()}

    return {"total_reward": _to_python(reward_info)}


def _safe_mean(xs: list[float | int]) -> float | None:
    if len(xs) == 0:
        return None
    return sum(xs) / len(xs)


def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    ground_truth_answers: list[str] | None = None,
    reward_fn: Callable[[str, str, str | None], dict[str, Any] | float] | None = None,
    max_new_tokens: int = 256,
    do_sample: bool = False,
) -> dict[str, Any]:
    """
    Log model generations during SFT/RL training.

    For each example, log:
    - prompt
    - generated response
    - ground-truth answer
    - reward information
    - average response token entropy
    - response length
    """
    if ground_truth_answers is not None:
        assert len(prompts) == len(ground_truth_answers)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    batch = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    prompt_len = batch["input_ids"].shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # generated_ids[:, :-1] predicts generated_ids[:, 1:]
        logits = model(generated_ids[:, :-1]).logits
        token_entropy = compute_entropy(logits)

    examples = []
    response_lengths = []
    correct_lengths = []
    incorrect_lengths = []
    entropies = []

    for i, prompt in enumerate(prompts):
        response_ids = generated_ids[i, prompt_len:]

        response = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        )

        response_length = len(
            tokenizer.encode(response, add_special_tokens=False)
        )

        gt_answer = None
        if ground_truth_answers is not None:
            gt_answer = ground_truth_answers[i]

        if reward_fn is not None:
            reward_info = _normalize_reward_info(
                reward_fn(prompt, response, gt_answer)
            )
        else:
            reward_info = {}

        # logits[:, j] predicts generated_ids[:, j + 1]
        # first response token is generated_ids[:, prompt_len]
        # so its entropy comes from token_entropy[:, prompt_len - 1]
        entropy_start = max(prompt_len - 1, 0)
        response_entropy = token_entropy[i, entropy_start:]

        if tokenizer.pad_token_id is not None:
            target_ids = generated_ids[i, 1:]
            valid_mask = target_ids[entropy_start:] != tokenizer.pad_token_id
            response_entropy = response_entropy[valid_mask]

        if response_entropy.numel() > 0:
            avg_entropy = response_entropy.mean().detach().cpu().item()
        else:
            avg_entropy = float("nan")

        is_correct = None
        if "is_correct" in reward_info:
            is_correct = bool(reward_info["is_correct"])
        elif "correct" in reward_info:
            is_correct = bool(reward_info["correct"])
        elif "answer_reward" in reward_info:
            is_correct = reward_info["answer_reward"] > 0

        response_lengths.append(response_length)
        entropies.append(avg_entropy)

        if is_correct is True:
            correct_lengths.append(response_length)
        elif is_correct is False:
            incorrect_lengths.append(response_length)

        item = {
            "prompt": prompt,
            "response": response,
            "ground_truth_answer": gt_answer,
            "reward_info": reward_info,
            "avg_token_entropy": avg_entropy,
            "response_length": response_length,
            "is_correct": is_correct,
        }

        examples.append(item)

        print("=" * 80)
        print(f"Example {i}")
        print("-" * 80)
        print("Prompt:")
        print(prompt)
        print("\nResponse:")
        print(response)
        print("\nGround truth:")
        print(gt_answer)
        print("\nReward info:")
        print(reward_info)
        print("\nAvg token entropy:")
        print(avg_entropy)
        print("\nResponse length:")
        print(response_length)
        print("\nIs correct:")
        print(is_correct)

    summary = {
        "avg_token_entropy": _safe_mean(entropies),
        "avg_response_length": _safe_mean(response_lengths),
        "avg_correct_response_length": _safe_mean(correct_lengths),
        "avg_incorrect_response_length": _safe_mean(incorrect_lengths),
    }

    print("=" * 80)
    print("Summary")
    print(summary)

    if was_training:
        model.train()

    return {
        "examples": examples,
        "summary": summary,
    }