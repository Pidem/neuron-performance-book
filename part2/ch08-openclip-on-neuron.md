# Case Study: OpenCLIP on Neuron

*A pure transformer model that works on Neuron out of the box, except for one hidden indexing op.*

```{admonition} Run it yourself
:class: tip
Scripts for this chapter (run on trn2.3xlarge with Neuron venv activated):

- `scripts/ch8_openclip_example.py` — profiles OpenCLIP, benchmarks vision/text encoders, scales batch and model size

The original OpenCLIP library is open source: [github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip.git).
```

---

## The Model: OpenCLIP

CLIP (Contrastive Language-Image Pre-training) learns a shared embedding space for images and text. It's the backbone of modern multimodal AI — image search, zero-shot classification, text-to-image generation all rely on CLIP embeddings.

The architecture is two transformers:
- **Vision encoder** (ViT): patch embedding → transformer layers → pooling → projection
- **Text encoder**: token embedding → transformer layers → pooling → projection
- **Contrastive loss**: cosine similarity between normalized image and text embeddings

Unlike SchNet (Ch7), CLIP has no scatter/gather in its core architecture. It's pure linear projections, attention, layer norms, and GELU activations — all ops that should map cleanly to the tensor engine. Let's see if that hypothesis holds.

---

## Move to Neuron — vision works, text doesn't

```python
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32")
model = model.eval().to("neuron")

images = torch.randn(8, 3, 224, 224, device="neuron")
text = tokenizer(["a photo of a cat"] * 8).to("neuron")

# Vision encoder: works perfectly
img_features = model.encode_image(images, normalize=True)  # ✓

# Text encoder: crashes
txt_features = model.encode_text(text, normalize=True)     # ✗
```

The vision encoder runs with **zero code changes** and **zero numerical error** (max diff vs CPU: 0.000000). But the text encoder fails:

```
RuntimeError: Compilation error occurred on Neuron for operation=aten::add.Tensor;
error message="COMPILATION FAILED: Memory Location: {add.3.156_set}@SB0,0,0,0,0,0"
```

**Root cause:** the text encoder's pooling layer uses fancy indexing:

```python
# In text_global_pool() — selects the EOS token per sample
pooled = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)]
```

This is `x[batch_indices, variable_position]` — the same class of indirect memory access that broke SchNet. Even a "standard" transformer has hidden indexing ops in its pooling layer.

---

## Zero Graph Breaks for the Vision Encoder

```python
compiled = torch.compile(model.encode_image, backend="neuron")
```

Dynamo analysis:
```
Graph count: 1
Graph break count: 0
```

The entire ViT — patch embedding, 12 transformer layers, pooling, projection — compiles into a single graph with zero breaks. This is the happy path: when all ops have native Neuron support, the compiler can see and fuse everything.

---

## Benchmark Results

### Vision encoder performance (ViT-B/32, batch=8)

| Mode | Time | vs Eager | vs CPU |
|------|------|----------|--------|
| CPU | 92.78ms | — | — |
| Eager (Neuron) | 19.41ms | — | 4.8× |
| Compiled (Neuron) | 8.24ms | 2.4× | **11.3×** |

Even eager mode is 4.8× faster than CPU — every op runs on the tensor engine natively. Compilation adds another 2.4× by fusing layer norms, GELUs, and residual adds into the surrounding matmuls.

### Scaling with batch size

| Batch | Compiled (Neuron) | CPU | Neuron vs CPU |
|-------|-------------------|-----|---------------|
| 1 | 2.35ms | 21.83ms | **9.3×** |
| 4 | 5.48ms | 56.72ms | **10.3×** |
| 8 | 8.26ms | 92.96ms | **11.3×** |
| 16 | 8.92ms | 169.89ms | **19.0×** |
| 32 | 16.82ms | 347.61ms | **20.7×** |
| 64 | 34.64ms | 727.09ms | **21.0×** |

The speedup grows with batch size. At batch=1, the tensor engine is underutilized (9.3×). At batch=16+, we hit the sweet spot: the matmuls are large enough to saturate the systolic array, and the speedup plateaus at ~21×.

```{admonition} Why larger batches help more
:class: note
The NeuronCore's tensor engine is a systolic array — it needs large matrix dimensions to keep all compute units busy. At batch=1 with ViT-B/32, the attention matmul is [1, 50, 768] × [1, 768, 50] — too small for full utilization. At batch=16, it's [16, 50, 768] × [16, 768, 50] — the 16 batch elements can be pipelined through the array efficiently.
```

### Scaling with model size

| Model | Vision Params | CPU | Neuron (compiled) | Speedup |
|-------|--------------|-----|-------------------|---------|
| ViT-B/32 | 88M | 91.7ms | 8.3ms | **11.1×** |
| ViT-B/16 | 86M | 349.4ms | 17.6ms | **19.9×** |
| ViT-L/14 | 304M | 1460.5ms | 89.2ms | **16.4×** |

ViT-B/16 gets 20× because the sequence length is 4× longer than B/32 (196 patches vs 49), making each attention matmul 4× larger — right in the tensor engine's sweet spot. ViT-L/14 at 304M params gets 16× — still excellent, with the slightly lower ratio likely due to memory bandwidth becoming the bottleneck at this scale.

---

## Fixing the Text Encoder

The text encoder's pooling uses `x[arange, text.argmax(dim=-1)]` to select the EOS token. We can apply the same trick from Ch7: express the selection as a matmul with a one-hot matrix.

```python
def encode_text_fixed_pool(self, text, normalize=False):
    cast_dtype = self.transformer.get_cast_dtype()
    x = self.token_embedding(text).to(cast_dtype)
    x = x + self.positional_embedding.to(cast_dtype)
    x = self.transformer(x, attn_mask=self.attn_mask)
    x = self.ln_final(x)

    # ORIGINAL: x[torch.arange(B), text.argmax(dim=-1)]  ← fancy indexing, crashes
    # FIX: one-hot matmul to select the EOS token
    eos_pos = text.argmax(dim=-1)
    one_hot = torch.zeros(x.shape[0], 1, x.shape[1], dtype=x.dtype, device=x.device)
    one_hot.scatter_(2, eos_pos.view(-1, 1, 1).to(x.device), 1.0)
    x = (one_hot @ x).squeeze(1)  # [B, 1, seq] @ [B, seq, dim] → [B, dim]

    if self.text_projection is not None:
        if isinstance(self.text_projection, nn.Linear):
            x = self.text_projection(x)
        else:
            x = x @ self.text_projection
    return F.normalize(x, dim=-1) if normalize else x
```

The one-hot matrix has a single 1 at the EOS position for each sample. Multiplying `[B, 1, seq] @ [B, seq, dim]` picks exactly one row — identical to `x[batch_idx, eos_idx]` — but expressed as a matmul that the tensor engine can execute natively.

```{admonition} The pattern repeats
:class: tip
Chapter 7: `x[idx_j]` (gather) → `G @ x` (matmul with selection matrix)
Chapter 8: `x[arange, argmax]` (2D gather) → `one_hot @ x` (matmul with one-hot)

Same principle, different shape. Any time you see `tensor[index]` on Neuron, think: "can I express this as a matmul with a binary matrix?"
```

---

## Contrast with SchNet

| | SchNet (Ch7) | OpenCLIP (Ch8) |
|---|---|---|
| Architecture | Message-passing GNN | Pure transformer |
| Core ops | scatter/gather + linear | attention + linear + norms |
| Works on Neuron? | Yes, but 22.7% fallback overhead | Vision: yes, zero fallbacks. Text: fails on pooling |
| Graph breaks | 0 (but CPU fallback within graph) | 0 (vision encoder fully native) |
| Compiled speedup vs CPU | Slower than CPU at small scale | **11–21× faster** even at batch=1 |
| Bottleneck | Irregular memory access (scatter/gather) | Text pooling uses fancy indexing |
| Fix | Dense adjacency matmul (14.7× vs original) | One-hot matmul for token selection |

The key difference: SchNet's bottleneck is in the **hot loop** (every layer, every forward pass), so it dominates performance. OpenCLIP's bottleneck is in the **pooling layer** (once at the end) — the 12 transformer layers are all native. This is why the vision encoder is already 11× faster without any code changes.

---

## When does it work out of the box?

The vision encoder is a best-case scenario for Neuron. What made it work:

1. **All ops are standard** — linear, attention (SDPA), LayerNorm, GELU, residual add
2. **Fixed tensor shapes** — no dynamic indexing, no variable-length sequences
3. **Large matmuls** — ViT attention at 768 width × 50-196 seq length utilizes the systolic array well
4. **No custom ops** — no scatter, no gather, no sort, no topk

If your model fits this profile, expect 10-20× speedups from `torch.compile(model, backend="neuron")` with no code changes needed.

---

## Summary

| What we tried | Result |
|---|---|
| Vision encoder → Neuron | ✓ Zero code changes, zero numerical error, zero graph breaks |
| `torch.compile` | 2.4× over eager, **11× over CPU** at batch=8 |
| Batch scaling | Up to **21× over CPU** at batch=64 |
| Model scaling | ViT-B/16 gets **20×**, ViT-L/14 gets **16×** |
| Text encoder | ✗ Fails — `argmax` pooling uses fancy indexing |
| Text fix | One-hot matmul for token selection (same trick as Ch7) |

The lesson: **transformers are the happy path on Neuron** — as long as you avoid indexing ops. When you hit one (even in a seemingly innocent pooling layer), the fix is the same: express the indexing as a matmul with a binary selection matrix.

---

*The compiler did all the heavy lifting for vision — zero manual intervention needed. But what if we want to go even faster? Next we'll look at profiling tools that reveal whether we're compute-bound or memory-bound, and what that means for optimization.*
