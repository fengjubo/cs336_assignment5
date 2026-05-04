from transformers import AutoModelForCausalLM, AutoTokenizer  , PreTrainedModel
import torch
import torch.nn.functional as F


# model = AutoModelForCausalLM.from_pretrained( "/data/a5-alignment/models/Qwen2.5-Math-1.5B", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", )  

# tokenizer = AutoTokenizer.from_pretrained("/data/a5-alignment/models/Qwen2.5-Math-1.5B")

model_path = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B"

# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     # device_map="auto",
# )
# tokenizer = AutoTokenizer.from_pretrained(model_path)


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    assert len(prompt_strs) == len(output_strs)

    batch_input_ids = []
    batch_labels = []
    batch_response_masks = []

    for prompt_str, output_str in zip(prompt_strs, output_strs):

        prompt_ids = tokenizer.encode(
            prompt_str,
            add_special_tokens = False,
        )
        output_ids = tokenizer.encode(
            output_str,
            add_special_tokens = False,
        )

        full_ids = prompt_ids + output_ids

        if len(full_ids) < 2:
            raise ValueError("prompt + output must contain at least two tokens")
        
        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        response_mask = [0] * len(labels)

        prompt_len = len(prompt_ids)

        response_start = max(prompt_len - 1, 0)

        for i in range(response_start, len(labels)):
            response_mask[i] = 1

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
        batch_response_masks.append(response_mask)

    max_len = max(len(x) for x in batch_input_ids)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:

        pad_token_id = tokenizer.eos_token_id

    padded_input_ids = []
    padded_labels = []
    padded_response_masks = []

    for input_ids, labels, response_mask in zip(
        batch_input_ids,
        batch_labels,
        batch_response_masks,
    ):

        pad_len = max_len - len(input_ids)

        padded_input_ids.append(input_ids + [pad_token_id] * pad_len)
        padded_labels.append(labels + [pad_token_id] * pad_len)
        padded_response_masks.append(response_mask + [0] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "response_mask": torch.tensor(padded_response_masks, dtype=torch.bool),
    }



def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    
    log_Z = torch.logsumexp(logits, dim = -1, keepdim = True)
    # torch.unsqueeze(log_Z, dim = -1) 错这里是展开
    log_probs = logits - log_Z 
    probs = torch.exp(log_probs)

    entropy = -(probs * log_probs).sum(dim = -1)

    return entropy


def get_response_log_probs(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    
   
    outputs = model(input_ids)
    logits = outputs.logits

    log_probs = F.log_softmax(logits, dim = -1)

    selected_log_probs = torch.gather(
        log_probs,
        dim = -1,
        index = labels.unsqueeze(-1),
    ).squeeze(-1)

    result = {
        "log_probs": selected_log_probs
    }

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result
    

def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float,
        dim: int| None = None,
) -> torch.Tensor:
    mask = mask.to(tensor.dtype)
    masked_tensor = tensor * mask

    if dim == None:
        summed = masked_tensor.sum()

    else:
        summed = masked_tensor.sum(dim = dim)

    return summed / normalize_constant


    
