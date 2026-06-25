# Casting on Neuron

*Fine-tune ESM-2 in mixed precision. What stays in BF16? What can go to FP8?*

```{admonition} Run it yourself
:class: tip
Script for this chapter (run on trn2.3xlarge with Neuron venv activated):

- `scripts/ch12_mixed_precision.py` — BF16 vs FP16 range, AMP autocast allowlist, the outlier problem, microscaling, FP8 throughput, and the shifted bottleneck

Shows the full precision story: from "why BF16 just works" through "why 2× FP8 matmul doesn't give 2× end-to-end" (the insight from Ron Diamant's re:Invent 2025 talk on Trn3).
```

---

## Mixed precision: the idea

Not every tensor in a model needs the same precision. Mixed precision training uses high precision where it matters (accumulation, optimizer) and low precision where it doesn't (forward pass matmuls, backward pass matmuls). The result: faster training with negligible accuracy loss.

---

## Automatic casting with the Neuron compiler

The simplest approach — the compiler handles it:

```python
# The compiler auto-casts FP32 matmuls to BF16 by default
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")  # FP32 weights
model.to("neuron")  # Compiler casts matmul weights to BF16 at compile time
```

The default compiler flag is `--auto-cast matmult --auto-cast-type bf16`. This means:
- FP32 **matmul** weights and operands are cast to BF16 (tensor engine runs in BF16)
- Matmul accumulation stays in FP32 (PSUM is always FP32)
- **Non-matmul ops** (softmax, layer norm, GELU) stay in FP32 on the vector/scalar engines
- Copy from PSUM → SBUF can cast back to BF16 at zero cost

This is mixed precision by default: matmuls are fast (BF16), reductions are precise (FP32). You can override with `--auto-cast all` (cast everything to BF16) or `--auto-cast none` (keep all FP32).

For most models, this "just works" with no accuracy impact. But you should always validate:

```python
# Compare FP32 vs auto-cast BF16 outputs
with torch.no_grad():
    out_fp32 = model_fp32(**inputs)
    out_bf16 = model_bf16(**inputs)
    max_diff = (out_fp32.last_hidden_state - out_bf16.last_hidden_state).abs().max()
    print(f"Max difference: {max_diff:.6f}")  # Should be < 0.01 for most models
```

---

## PyTorch AMP (Automatic Mixed Precision)

For training, PyTorch's AMP context manager gives finer control:

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()  # Only needed for FP16, not BF16

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast(device_type="neuron", dtype=torch.bfloat16):
        output = model(batch)
        loss = criterion(output, labels)
    
    # Backward in BF16 (or FP32 where needed)
    loss.backward()
    optimizer.step()
```

Why does `GradScaler` exist for FP16 but not BF16? FP16's dynamic range maxes out at 65,504 — gradients in deep networks can easily exceed this and overflow to infinity. The scaler multiplies the loss by a large factor before backward (keeping gradients in representable range), then divides gradients back down before the optimizer step. BF16 shares FP32's full dynamic range (±3.4×10³⁸), so overflow never happens in practice and no scaler is needed. On Neuron, BF16 is the default — you won't need `GradScaler`.

What `autocast` does:
- Matmuls and convolutions → BF16 (tensor engine inputs)
- Reductions (softmax, LayerNorm) → stays FP32 (numerical stability)
- Loss computation → FP32
- Optimizer states → FP32

You don't specify per-layer — PyTorch has a built-in allowlist/denylist of which ops are safe in lower precision.

---

## Where precision matters in ESM-2

| Component | Safe in BF16? | Safe in FP8? | Why |
|-----------|:---:|:---:|-----|
| Embedding lookup | ✓ | ✓ | Discrete tokens, precision barely matters |
| QKV projection (matmul) | ✓ | ✓ | Large matmul, well-conditioned |
| Attention scores (QK^T) | ✓ | Careful | Small values, softmax amplifies errors |
| Softmax | ✗ (keep FP32) | ✗ | Exponentiation: tiny input differences → huge output differences |
| Attention × V (matmul) | ✓ | ✓ | Matmul with normalized inputs |
| FFN up projection | ✓ | ✓ | Large matmul, well-conditioned |
| GELU activation | FP32 internal | ✗ | Transcendental function, needs precision |
| FFN down projection | ✓ | ✓ | Large matmul |
| LayerNorm | ✗ (keep FP32) | ✗ | Variance computation: small differences matter |
| Loss (cross-entropy) | ✗ (keep FP32) | ✗ | Log of small probabilities |

**Pattern:** matmuls are safe to quantize (the tensor engine accumulates in FP32 anyway). Reductions and nonlinearities need full precision.

---

## FP8 on Trn2: the roofline benefit

Recall from Chapter 9: a BF16 matmul becomes compute-bound at B > 230. With FP8 weights:
- Load half the bytes → critical batch size drops to B > 115
- Token generation (B=1): still memory-bound, but 2× faster because DMA moves half the data
- FP8 tensor engine throughput: 1299 TFLOPS vs 667 TFLOPS (BF16) — nearly 2× compute as well

The combination makes FP8 particularly valuable for inference:
- Weights stored in FP8 on HBM (half the memory footprint → fit larger models)
- Loaded to SBUF faster (half the bytes)
- Computed faster on tensor engine (2× TFLOPS)
- Accumulated in FP32 (no precision loss during matmul)

The only precision loss is at the *casting boundary* — when you quantize your BF16 weights to FP8 before deployment.

---

## Practical FP8 quantization workflow

```python
# Quantize weights post-training (weight-only quantization)
import torch

def quantize_linear_to_fp8(module):
    """Replace Linear weight with FP8 E4M3 quantized version"""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            # Per-tensor absmax scaling
            scale = child.weight.abs().max() / torch.finfo(torch.float8_e4m3fn).max
            child.weight.data = (child.weight / scale).to(torch.float8_e4m3fn)
            child.weight_scale = scale  # Store for dequant at compute time
        else:
            quantize_linear_to_fp8(child)
```

For production, use the Neuron compiler's built-in quantization:
```bash
# Compiler flag for FP8 weight quantization
NEURON_CC_FLAGS="--auto-cast=matmult --auto-cast-type=fp8_e4m3"
```

For fine-tuning on limited hardware, **QLoRA** stores the base model weights in 4-bit (NF4) while keeping LoRA adapter weights and gradients in BF16. This lets you fine-tune a 7B model on a single chip — the base weights are frozen and compressed, while only the small adapters train in full precision. On Neuron, this workflow is available through `optimum-neuron` with HuggingFace's PEFT library.

---

## The asymmetric precision strategy

Not all model components benefit equally from quantization. A common pattern for multimodal models (like OpenCLIP from Chapter 8):

- **Vision encoder (ViT):** FP8 weights, BF16 activations. ViT is robust to quantization — large matmuls, well-conditioned weights, minimal impact on embedding quality
- **Text encoder:** BF16 weights, BF16 activations. Text embeddings are more sensitive — vocabulary softmax requires precision
- **Contrastive head:** FP32. The similarity computation and contrastive loss need full precision

This asymmetric approach gives you most of the speed benefit with minimal accuracy cost.

---

## Validating accuracy after quantization

Always measure task-specific accuracy, not just tensor-level differences:

```python
# Bad: checking max absolute difference (doesn't reflect task performance)
max_diff = (fp32_output - fp8_output).abs().max()  # Might be large but irrelevant

# Good: checking downstream task metric
fp32_accuracy = evaluate(model_fp32, test_set)  # e.g., perplexity, contact prediction F1
fp8_accuracy = evaluate(model_fp8, test_set)
print(f"FP32: {fp32_accuracy:.4f}, FP8: {fp8_accuracy:.4f}, Delta: {fp32_accuracy - fp8_accuracy:.4f}")
```

For ESM-2 on protein contact prediction:
- FP32 → BF16: typically < 0.1% accuracy drop (safe)
- BF16 → FP8 (weights only): typically < 0.5% accuracy drop (validate per task)
- BF16 → FP8 (weights + activations): can degrade 1-3% (requires calibration)

---

## The shifted bottleneck

FP8 makes matmuls 2× faster (half the bytes, double the TFLOPS). But a transformer layer isn't just matmuls. Between each matmul sits a softmax, a layer norm, a GELU — all running on the Vector and Scalar engines in BF16. When you halve the matmul time, those non-matmul ops don't get any faster: They become a larger fraction of total runtime.

Before FP8: matmuls take 80% of the time, non-matmul takes 20%. 
After FP8: matmuls take 67% of the time, non-matmul takes 33%.

You got a ~1.2× end-to-end speedup, not the 2× you expected. The bottleneck shifted from the Tensor Engine to the Vector/Scalar engines — from compute to the ops *between* compute.

This is Amdahl's Law applied to hardware engines. And it's exactly why Part V exists. The non-matmul gaps — softmax, normalization, activation functions, reductions — are where NKI kernels deliver the next jump. A fused attention kernel that keeps scores in SBUF and does softmax inline eliminates the gap entirely. A fused RMSNorm+Quantize kernel (like `nisa.quantize_mx` on Trn3) turns two engine handoffs into one instruction.

The precision ladder gets you from 30% MFU to 50%. Custom kernels for the shifted bottleneck get you from 50% to 80%.

---

*The profiler shows gaps between matmuls. The matmuls themselves are fast. The bottleneck has shifted to ops the compiler can't fuse well enough. Time to write your own.*

*Next: Part V — Writing your own kernels (NKI).*
