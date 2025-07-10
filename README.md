# GRIFFIN-Plus: Prompt-Adaptive Structured Pruning with Refresh

**GRIFFIN-Plus** is a *training-free*, inference-time compression technique inspired by the  
[GRIFFIN paper (Dong *et al.*, 2024)](https://arxiv.org/abs/2402.XXXX). It adds a lightweight
“refresh” mechanism that keeps text quality stable during very long generations.

1. **Neuron selection** – For each feed-forward (FF) layer we keep the *k* most active
   neurons in the prompt, measured with an ℓ-2 norm (“flocking”) statistic.  
2. **Structured slicing** – Instead of masking, we **slice** the FF weight matrices
   (`c_fc`, `c_proj`, …), yielding real speed-ups and memory savings.  
3. **Periodic refresh** – Every *N* generated tokens (default = 64) we recompute the
   expert sets over the recent context to mitigate quality drift beyond 1 k tokens.

The whole pipeline lives in **one Python file** and works with any Hugging-Face
causal-LM checkpoint that uses a two-linear FF block.


## Demo dataset

* **WikiText-2** (loaded automatically via `datasets`) – used to report perplexity.  
  No manual download required.

## Quick-start
Run the demo on GPT-2 small (CPU)

```bash
pip install torch transformers datasets tqdm
python griffin_plus.py --model gpt2 --ratio 0.5 --device cpu
````

### CLI flags

| Flag       | Default | Description                               |
| ---------- | ------- | ----------------------------------------- |
| `--model`  | `gpt2`  | HF model name or local path               |
| `--ratio`  | `0.5`   | Fraction of FF neurons **kept** per layer |
| `--device` | `cpu`   | `cpu` or `cuda`                           |

## What the script does

1. **Collect prompt activations** → pick top-k neurons per layer.
2. **Clone & graft** pruned FF layers (structured slicing).
3. **Generate** with the pruned model, **refreshing** the experts every 64 tokens.
4. **Report** WikiText-2 perplexity for full **vs.** pruned model.

### Example output

```
=== GRIFFIN-Plus generation ===
In a shocking finding, scientists discovered that the world’s smallest
quantum computer can brew an espresso while predicting solar flares…

Full model PPL:   29.81
GRIFFIN-Plus PPL: 30.97 (keep=0.50)
```

Perplexity rises only ≈ 4 %, despite *halving* every FF layer and running the
entire inference with the compact network.

---

## Scaling to larger LLMs

```bash
python griffin_plus.py \
  --model meta-llama/Llama-2-7b-hf \
  --ratio 0.5 \
  --device cuda
```

The implementation auto-detects FF sub-layers in Llama-2, GPT-NeoX, Mistral,
and similar architectures. VRAM savings scale roughly linearly with the
keep-ratio.

---

## Improvement over the original GRIFFIN
Orignal Griffin is available at https://arxiv.org/abs/2402.19427 and its repo is https://github.com/hdong920/GRIFFIN

The original GRIFFIN reports quality drift on *very* long generations because the expert neurons are frozen from an initial short prompt.
**GRIFFIN-Plus** refreshes the experts every `refresh_interval` tokens (default = 64) using the most recent context window, reducing that degradation
with negligible overhead (one forward pass per refresh).

---

```
```
