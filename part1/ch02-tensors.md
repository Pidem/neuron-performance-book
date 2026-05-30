# Tensors aren't just arrays

In Chapter 1, we watched the dispatcher route `aten::matmul` to a backend kernel. But what exactly is the dispatcher *passing* to that kernel? Not "a matrix" — that's a mathematical concept. The kernel needs to know: where does the data live in memory? How big is each element? How do I find element `[i, j]`?

The answer is the **tensor** — not the mathematical object, but PyTorch's data structure. A tensor is a *view* into a block of memory, equipped with just enough metadata to navigate it. Understanding this data structure is the key to understanding why some operations are free and others are expensive — and why the same mathematical operation can have wildly different performance depending on how the data is laid out.

---

## Act 1: What a tensor actually is

Let's look at ESM-2's embedding table — the lookup table that converts amino acid tokens into 1280-dimensional vectors:

```python
from transformers import EsmModel
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

embed = model.embeddings.word_embeddings.weight
print(f"Shape:    {embed.shape}")
print(f"Dtype:    {embed.dtype}")
print(f"Strides:  {embed.stride()}")
print(f"Device:   {embed.device}")
print(f"Pointer:  {embed.data_ptr()}")
```

```
Shape:    torch.Size([33, 1280])
Dtype:    torch.float32
Strides:  (1280, 1)
Device:   cpu
Pointer:  140234567890944
```

That's the whole tensor. Five fields:

```
┌─────────────────────────────────────────────────────────┐
│  Tensor                                                 │
│                                                         │
│  data_ptr ──────► [raw bytes in memory]                 │
│  shape    = (33, 1280)                                  │
│  strides  = (1280, 1)                                   │
│  dtype    = float32  (4 bytes per element)              │
│  device   = cpu                                         │
└─────────────────────────────────────────────────────────┘
```

The data pointer says *where*. The shape says *how big* (logically). The dtype says *what kind*. The device says *which hardware*. But strides — strides are the interesting one.

---

## Act 2: Strides — the navigation system

Memory is flat. A 2D tensor is a human concept. The hardware sees a linear sequence of bytes. Strides tell you how to convert a logical index `[i, j]` into a physical memory offset:

```
offset = i * stride[0] + j * stride[1]
```

Here's how it works on a small example — a 2×3 tensor with strides `(3, 1)`:

```
  Logical view:         Physical memory (flat):
  ┌───┬───┬───┐        ┌───┬───┬───┬───┬───┬───┐
  │ 1 │ 2 │ 3 │        │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │
  ├───┼───┼───┤        └───┴───┴───┴───┴───┴───┘
  │ 4 │ 5 │ 6 │          ↑               ↑
  └───┴───┴───┘        [0,0]           [1,0]

  tensor[1, 0]:
    offset = 1 × stride[0] + 0 × stride[1]
           = 1 × 3          + 0 × 1
           = 3  → element "4"
```

For our embedding table with strides `(1280, 1)`:

```python
# Where does embed[5, 100] live?
offset = 5 * 1280 + 100 * 1 = 6500  # elements from the start
bytes_offset = 6500 * 4  # 4 bytes per float32 = 26000 bytes
```

Visually, the 33×1280 table is stored as a flat array of 42,240 floats:

```
Memory: [row0_col0, row0_col1, ..., row0_col1279, row1_col0, row1_col1, ...]
         ├──────── 1280 elements ────────────────┤
         stride[0] = 1280 (skip this many to reach next row)
         stride[1] = 1 (skip this many to reach next column)
```

This is **row-major** (C-contiguous) layout: elements within a row are adjacent in memory. PyTorch uses row-major by default.

Why does this matter? Because hardware reads memory in **chunks**. When you access `embed[5]` (an entire row), the hardware loads 1280 consecutive floats — one efficient burst. When you access `embed[:, 100]` (a column), the hardware must skip 1280 elements between each value — 33 separate small reads.

---

## Act 3: Views — zero-cost reshaping

Here's where tensors diverge from arrays. Many operations that *look* like they create new data actually just create a new **view** — a different set of (shape, strides, offset) pointing at the same memory:

```python
row = embed[5]  # slice one row
print(f"Shape: {row.shape}, Strides: {row.stride()}")
print(f"Same memory: {row.data_ptr() == embed.data_ptr() + 5 * 1280 * 4}")
```

```
Shape: torch.Size([1280]), Strides: (1,)
Same memory: True
```

No copy. The new tensor just points 5×1280×4 bytes into the same buffer, with shape `(1280,)` and stride `(1,)`.

Transpose is the most powerful example:

```python
embed_t = embed.T
print(f"Shape:   {embed_t.shape}")
print(f"Strides: {embed_t.stride()}")
print(f"Same memory: {embed_t.data_ptr() == embed.data_ptr()}")
```

```
Shape:   torch.Size([1280, 33])
Strides: (1, 1280)
Same memory: True
```

`.T` didn't move a single byte. It just swapped the strides: what was stride `(1280, 1)` became `(1, 1280)`. Both tensors are just different *views* of the same underlying storage:

```
  embed                         embed.T
  ┌─────────────────┐          ┌─────────────────┐
  │ shape=(33, 1280)│          │ shape=(1280, 33)│
  │ strides=(1280,1)│          │ strides=(1,1280)│
  │ offset=0        │          │ offset=0        │
  └────────┬────────┘          └────────┬────────┘
           │                            │
           └────────────┬───────────────┘
                        ▼
              ┌───────────────────┐
              │  Storage          │
              │  [42,240 floats]  │
              │  device=cpu       │
              └───────────────────┘
```

The "rows" of the transposed tensor are the columns of the original — and they're spaced 1 element apart in memory, while the "columns" of the transposed tensor are spaced 1280 apart.

This is free. But there's a catch.

---

## Act 4: Contiguity — when free becomes expensive

A tensor is **contiguous** when its strides match what you'd expect from its shape — i.e., elements are laid out in memory in the "natural" row-major order. Our original embedding is contiguous. The transpose is not:

```python
print(f"embed contiguous:   {embed.is_contiguous()}")
print(f"embed.T contiguous: {embed_t.is_contiguous()}")
```

```
embed contiguous:   True
embed.T contiguous: False
```

Why does this matter? Because many operations **require** contiguous input. When you call `.reshape()` on a non-contiguous tensor, PyTorch must allocate new memory and copy:

```python
# Contiguous tensor: reshape is free (just a view)
flat = embed.reshape(-1)
print(f"Reshape contiguous: view = {flat.data_ptr() == embed.data_ptr()}")

# Non-contiguous tensor: reshape forces a copy
flat_t = embed_t.reshape(-1)
print(f"Reshape transposed: view = {flat_t.data_ptr() == embed_t.data_ptr()}")
```

```
Reshape contiguous: view = True
Reshape transposed: view = False
```

That second reshape allocated 33×1280×4 = 169 KB and copied every element into the new layout. On a single embedding table, this is negligible. On attention matrices in a 33-layer model, these hidden copies add up.

The rule:

| Operation | Contiguous input | Non-contiguous input |
|-----------|-----------------|---------------------|
| `.view()` | Free (view) | **Error** (refuses) |
| `.reshape()` | Free (view) | Silent copy |
| `.contiguous()` | No-op | Explicit copy |
| `.T` / `.transpose()` | Always free | Always free |

---

## Act 5: ESM-2's attention matrices — a real example

Let's look at where layout matters in practice. Run a forward pass and grab the attention weights:

```python
from transformers import EsmTokenizer
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

inputs = tokenizer("FVNQHLCGSHLVEALYLVCGERGFFYTPKT", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

attn = outputs.attentions[0]  # first layer
print(f"Shape:   {attn.shape}")
print(f"Strides: {attn.stride()}")
```

```
Shape:   torch.Size([1, 20, 32, 32])
Strides: (20480, 1024, 32, 1)
```

The shape is `[batch, heads, seq_len, seq_len]`. The strides tell us the memory layout:

```
batch  stride = 20480  (= 20 × 1024 = heads × seq² — skip one full batch)
head   stride = 1024   (= 32 × 32 = seq² — skip one full head)
row    stride = 32     (= seq_len — skip one row)
col    stride = 1      (adjacent in memory)
```

This means: **within one attention head, the seq×seq matrix is contiguous**. Accessing `attn[0, 0]` (one head's full attention pattern) gives you a contiguous 32×32 block — perfect for a kernel that processes one head at a time.

But what if you wanted all heads' attention for position 5 attending to position 10?

```python
across_heads = attn[0, :, 5, 10]  # shape: [20]
print(f"Strides: {across_heads.stride()}")
print(f"Contiguous: {across_heads.is_contiguous()}")
```

```
Strides: (1024,)
Contiguous: True
```

Wait — this *is* contiguous? Yes, because a 1D tensor with any stride is always "contiguous" (there's only one dimension, so elements are trivially in order). But the stride is 1024 — meaning each element is 1024 floats apart in memory. A DMA engine loading this would fetch 20 values scattered across 80 KB of memory, rather than 20 adjacent values in 80 bytes.

This is why attention implementations carefully choose which dimension to parallelize over. Parallelizing over heads (dim 1) gives each thread a contiguous seq×seq block. Parallelizing over sequence positions would give each thread scattered data.

---

## Act 6: Why this matters for accelerators

On any accelerator — GPU or Neuron — there's a memory hierarchy:

```
┌─────────────────────────────────────────────┐
│  On-chip SRAM (fast, small)                 │
│  - GPU: shared memory, L1/L2 cache          │
│  - Neuron: SBUF (State Buffer), PSUM        │
├─────────────────────────────────────────────┤
│  Device memory (slow, large)                │
│  - GPU: HBM (High Bandwidth Memory)        │
│  - Neuron: HBM                              │
└─────────────────────────────────────────────┘
```

Kernels work by loading **tiles** from device memory into on-chip SRAM, computing on them, and writing results back. The DMA engine that moves these tiles works best when the data is contiguous — it can issue one large transfer instead of many small ones.

When a kernel receives a non-contiguous tensor:
1. It might call `.contiguous()` internally (hidden copy — you pay for it but don't see it)
2. It might use strided access (slower DMA, underutilized bandwidth)
3. On Neuron specifically: the compiler may insert explicit data rearrangement instructions to pack tiles into SBUF-friendly layouts

This is the connection to Chapter 1's fusion story. Remember:

```
Without fusion: matmul → write to HBM → read from HBM → softmax → ...
With fusion:    Load once → compute everything in SRAM → write once
```

Fusion eliminates HBM round-trips. But even *within* a single kernel, the layout of your tensor determines how efficiently data moves from HBM to SRAM. A contiguous tile loads in one burst. A strided tile requires gather operations.

---

## Act 7: The dispatcher cares about layout

Back in Chapter 1, we said the dispatcher routes ops based on device and dtype. There's a third axis: **layout**.

```python
# Dense strided tensor — the default
dense = torch.randn(1000, 1000)
print(f"Layout: {dense.layout}")  # torch.strided

# Sparse tensor — different layout, different kernels
sparse = dense.to_sparse()
print(f"Layout: {sparse.layout}")  # torch.sparse_coo
```

```
Layout: torch.strided
Layout: torch.sparse_coo
```

The dispatcher uses (device, dtype, layout) as a triple to select the right kernel. Same mathematical operation, completely different implementation:

```
torch.mm(A, B)
  ├── device=cuda, layout=strided  → cuBLAS GEMM
  ├── device=cuda, layout=sparse   → cuSPARSE SpMM
  ├── device=cpu,  layout=strided  → MKL/OpenBLAS GEMM
  └── device=neuron, layout=strided → NEFF matmul kernel
```

We'll see this concretely in Chapter 10 when we bring PyTorch Geometric's sparse operations to Neuron — the dispatcher must find (or fall back from) kernels for sparse layouts on the Neuron device.

---

## The big picture

```
Chapter 1: model(x) → dispatcher → kernel
                         │
Chapter 2: What does the kernel receive?
                         │
                         ▼
           ┌─────────────────────────────┐
           │  Tensor                     │
           │  - data_ptr (where)         │
           │  - shape    (logical size)  │
           │  - strides  (navigation)    │
           │  - dtype    (element type)  │
           │  - device   (hardware)      │
           └─────────────────────────────┘
                         │
           Layout determines:
           - Whether views are free or force copies
           - How efficiently DMA loads tiles
           - Which kernel the dispatcher selects
```

The kernel doesn't see "a matrix." It sees a pointer, a shape, and strides. Its job is to load tiles from device memory into fast on-chip SRAM, compute, and write back. How well it can do this depends entirely on how the data is laid out.

But here's the thing: in eager mode (Chapter 1), each kernel makes its own local decision about tiling and layout. No kernel knows what the *next* kernel needs. If kernel A writes its output in layout X, and kernel B needs layout Y, someone has to pay for the rearrangement.

*What if something could see the whole graph of operations and choose layouts globally?*

That's the compiler. Chapter 3.
