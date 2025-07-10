
# GRIFFIN-Plus: Prompt-Driven Structured Pruning with
# Periodic Refresh for Long-Form Generation
# Improved version of https://arxiv.org/abs/2402.19427
# --------------------------------------------------

# A research-grade prototype implementing the core ideas of the GRIFFIN
# algorithm and adds one improvement: periodic expert refresh during very
#long generations to curb quality drift.
#based on https://arxiv.org/abs/2402.19427
#Requirements torch>2.0 transformers datasets tqdm

from __future__ import annotations
import argparse, copy, math
from typing import Dict, List

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm


class ActivationCatcher:
    """Collect pre-activation tensors from each FF block during a forward pass."""

    def __init__(self, model: nn.Module):
        self._handles = []
        self._cache: Dict[str, List[torch.Tensor]] = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name.endswith(".c_fc"):
                self._cache[name] = []
                handle = module.register_forward_hook(self._hook(name))
                self._handles.append(handle)

    def _hook(self, layer_name: str):
        def fn(module, inp, out):
            self._cache[layer_name].append(out.detach())
        return fn

    def activations(self) -> Dict[str, torch.Tensor]:
        return {k: torch.cat(v, dim=1) for k, v in self._cache.items()}

    def remove(self):
        for h in self._handles:
            h.remove()


def topk_indices(tensor: torch.Tensor, k: int) -> torch.Tensor:
    scores = torch.linalg.norm(tensor, ord=2, dim=1)
    return torch.topk(scores, k=k, dim=0).indices


def build_pruned_linear(orig: nn.Linear, keep_idx: torch.Tensor, dim: str) -> nn.Linear:
    if dim == "out":
        new_linear = nn.Linear(orig.in_features, len(keep_idx), bias=orig.bias is not None,
                               device=orig.weight.device, dtype=orig.weight.dtype)
        new_linear.weight.data = orig.weight.data[keep_idx].clone()
        if orig.bias is not None:
            new_linear.bias.data = orig.bias.data[keep_idx].clone()
    elif dim == "in":
        new_linear = nn.Linear(len(keep_idx), orig.out_features, bias=orig.bias is not None,
                               device=orig.weight.device, dtype=orig.weight.dtype)
        new_linear.weight.data = orig.weight.data[:, keep_idx].clone()
        if orig.bias is not None:
            new_linear.bias.data = orig.bias.data.clone()
    else:
        raise ValueError("dim must be 'in' or 'out'")
    return new_linear


def graft_ff_block(block: nn.Module, keep_idx: torch.Tensor):
    fc_in: nn.Linear = block.c_fc
    fc_out: nn.Linear = block.c_proj
    block.c_fc = build_pruned_linear(fc_in, keep_idx, dim="out")
    block.c_proj = build_pruned_linear(fc_out, keep_idx, dim="in")


def make_griffin_plus(model: AutoModelForCausalLM, prompt_ids: torch.Tensor, topk_ratio: float = 0.5):
    model.eval()
    catcher = ActivationCatcher(model)
    with torch.no_grad():
        model(prompt_ids)
    activations = catcher.activations()
    catcher.remove()

    pruned = copy.deepcopy(model).eval()
    for name, block in pruned.named_modules():
        if isinstance(block, type(pruned.transformer.h[0])):
            layer_id = name.split(".")[2]
            act_key = f"transformer.h.{layer_id}.mlp.c_fc"
            z = activations[act_key].squeeze(0)
            hidden = z.size(-1)
            k = max(1, int(topk_ratio * hidden))
            idx = topk_indices(z, k)
            graft_ff_block(block.mlp, idx)

    return pruned


def generate_with_refresh(base_model, tokenizer, prompt: str, max_new_tokens: int = 128,
                          topk_ratio: float = 0.5, refresh_interval: int = 64,
                          device: str = "cpu") -> str:
    model = base_model.to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = prompt_ids.clone()

    while generated.size(1) - prompt_ids.size(1) < max_new_tokens:
        tail = generated[:, -refresh_interval:]
        expert_model = make_griffin_plus(model, tail, topk_ratio).to(device)

        gen_cfg = GenerationConfig(max_new_tokens=refresh_interval,
                                   do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id,
                                   eos_token_id=tokenizer.eos_token_id)
        with torch.no_grad():
            out = expert_model.generate(generated, generation_config=gen_cfg)
        generated = out

    return tokenizer.decode(generated.squeeze(), skip_special_tokens=True)


def perplexity(model, tokenizer, text_list, stride=512, device="cpu"):
    model.to(device).eval()
    nlls, n_tokens = 0.0, 0
    for txt in tqdm(text_list, desc="PPL eval"):
        ids = tokenizer(txt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            for i in range(0, ids.size(1), stride):
                chunk = ids[:, i:i+stride]
                out = model(chunk, labels=chunk)
                nlls += out.loss.item() * chunk.numel()
                n_tokens += chunk.numel()
    return math.exp(nlls / n_tokens)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model).eval()

    prompt = "In a shocking finding, scientists discovered that"
    print("=== GRIFFIN-Plus generation ===")
    out = generate_with_refresh(base, tok, prompt, max_new_tokens=80,
                                topk_ratio=args.ratio, device=args.device)
    print(out)

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")[:100]
    full_ppl = perplexity(base, tok, data["text"], device=args.device)

    pruned_model = make_griffin_plus(
        base,
        tok(prompt, return_tensors="pt").input_ids.to(args.device),
        args.ratio,
    )
    pruned_ppl = perplexity(pruned_model, tok, data["text"], device=args.device)

    print(f"\nFull model PPL:   {full_ppl:.2f}")
    print(f"GRIFFIN-Plus PPL: {pruned_ppl:.2f} (keep={args.ratio})")


if __name__ == "__main__":
    main()
