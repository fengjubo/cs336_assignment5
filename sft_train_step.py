import torch

from homework.helpers import masked_normalize

def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    if policy_log_probs.shape != response_mask.shape:
        raise ValueError(
            f"policy_log_probs shape {policy_log_probs.shape} "
            f"does not match response_mask shape {response_mask.shape}"
        )

    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    if normalize_constant <= 0:
        raise ValueError("normalize_constant must be positive")
    
    per_token_loss = -policy_log_probs

    loss_bf_grad_accum = masked_normalize(
        per_token_loss,
        response_mask,
        normalize_constant,
    )

    loss_bf_grad_accum = loss_bf_grad_accum / policy_log_probs.shape[0]

    loss = loss_bf_grad_accum / gradient_accumulation_steps

    loss.backward()

    metadata = {
        "loss_before_grad_accum": loss_bf_grad_accum.detach(),
        "num_response_tokens": response_mask.sum().detach(),
    }

    return loss.detach(), metadata