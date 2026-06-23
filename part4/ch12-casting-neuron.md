# Casting on Neuron

*Fine-tune ESM-2 in mixed precision. What stays in BF16? What can go to FP8?*

---

## Mixed precision: the idea

Not every tensor in a model needs the same precision. Mixed precision training uses high precision where it matters (accumulation, optimizer) and low precision where it doesn't (forward pass matmuls, backward pass matmuls). The result: faster training with negligible accuracy loss.

---

## Automatic casting with the Neuron compiler

The simplest approach — the compiler handles everything:

```python
# The compiler auto-casts FP32 models to BF16 by default
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")  # FP32 weights
model.to("neuron")  # Compiler auto-casts to BF16 at compile time
```

Under the hood:
- FP32 weight tensors are cast to BF16 before loading to SBUF
- Matmul accumulation stays in FP32 (PSUM is always FP32)
- Copy from PSUM → SBUF can cast back to BF16 at zero cost
- The model runs in BF16 but with FP32-equivalent accumulation precision

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

Why does `GradScaler` exist for FP16 but not BF16? FP16's dynamic range maxes out at 65,504 — gradients in deep networks can easily exceed this and overflow to infinity. The scaler multiplies the loss by a large factor before backward (keeping gradients in representable range), then divides gradients back down before the optimizer step. BF16 shares FP32's full dynamic range (±3.4×10³⁸), so overflow essentially never happens and no scaler is needed. On Neuron, BF16 is the default — you won't need `GradScaler`.

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

## Summary: the precision ladder on Neuron

| Stage | Weights | Activations | Optimizer | Use case |
|-------|---------|-------------|-----------|----------|
| **FP32 baseline** | FP32 | FP32 | FP32 | Reference accuracy |
| **BF16 training** | BF16 | BF16 | FP32 | Default on Neuron (auto-cast) |
| **FP8 training** | FP8 E4M3 | BF16 | FP32 | Faster iteration on Trn2 |
| **FP8 inference** | FP8 E4M3 | FP8 E4M3 | — | Maximum throughput |
| **INT8 inference** | INT8 | BF16 | — | Alternative to FP8 |

Each step down the ladder trades precision for speed. The compiler's FP32 accumulation in PSUM means the precision loss happens only at the storage/casting level — the math itself is always full precision.

*Question raised → "I've squeezed what I can from precision and compilation. The profiler still shows gaps between matmuls. What now?"*

*Next: Part V — Writing your own kernels (NKI).*
