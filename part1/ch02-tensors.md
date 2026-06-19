# Tensors aren't just arrays

In Chapter 1, we watched the dispatcher route `aten::matmul` to a backend kernel. But what exactly does the kernel receive? Not "a matrix" — that's a mathematical concept. The kernel needs a physical address, a way to navigate the memory layout, and enough metadata to interpret the bytes.

That's what a **tensor** is in PyTorch: a view into a block of memory, plus the metadata to navigate it. This distinction — data vs. view of data — determines which operations are free and which silently copy gigabytes behind your back.

---

## What is a tensor?

Let's look at ESM-2's embedding table — the lookup table that converts amino acid tokens into 1280-dimensional vectors:

```python
from transformers import EsmModel
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

embed = model.embeddings.word_embeddings.weight
print(f"Pointer:  {embed.data_ptr()}")
print(f"Shape:    {embed.shape}")
print(f"Dtype:    {embed.dtype}")
print(f"Device:   {embed.device}")
print(f"Strides:  {embed.stride()}")
```

```none
Pointer:  494627968
Shape:    torch.Size([33, 1280])
Dtype:    torch.float32
Device:   cpu
Strides:  (1280, 1)
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

The data pointer gives the starting address. The shape describes logical dimensions. The dtype says how many bytes per element and how to interpret them. The device tells the dispatcher which hardware owns this memory. And strides — the interesting one — encode how to convert a multi-dimensional index into a byte offset, which is what makes transpose and reshape possible without copying.

---

## Strides 

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

## Views — zero-cost reshaping

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

`.T` didn't move a single byte. It just swapped the strides: what was stride `(1280, 1)` became `(1, 1280)`. Both tensors are just different *views* of the same underlying storage. The "rows" of the transposed tensor are the columns of the original — and they're spaced 1 element apart in memory, while the "columns" of the transposed tensor are spaced 1280 apart.

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
              └───────────────────┘
```



---

## The importancew of contiguity

A tensor is **contiguous** when its strides match what you'd expect from its shape — i.e., elements are laid out in memory in the "natural" row-major order. Our original embedding is contiguous. The transpose is not:

```python
print(f"embed contiguous:   {embed.is_contiguous()}")
print(f"embed.T contiguous: {embed_t.is_contiguous()}")
```

```
embed contiguous:   True
embed.T contiguous: False
```

Why does this matter? Because kernels load data from device memory (HBM) into fast on-chip SRAM via DMA transfers that operate on **contiguous address ranges**. One DMA instruction says "starting at address X, copy N consecutive bytes." If your tensor is contiguous, that's one burst. If it's non-contiguous — say, a transpose where consecutive logical elements are 1280 floats apart — the hardware must issue separate transfers for each row, wasting bandwidth.

This applies on any accelerator including Neuron (where DMA loads tiles from HBM into SBUF). We'll see the hardware details in Chapter 4.

When you call `.reshape()` on a non-contiguous tensor, PyTorch must allocate new memory and copy:

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

### The impact of layouts on performance experiment

Let's look at where layout matters in practice. Run a forward pass and grab the attention weights:

```python
import torch
from transformers import EsmTokenizer, EsmModel

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D", attn_implementation="eager")

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

This means: **within one attention head, the seq×seq matrix is contiguous**. Accessing `attn[0, 0]` (one head's full attention pattern) gives you a contiguous 32×32 block — exactly what a kernel wants when it processes one head at a time.

But what if you wanted all heads' attention for position 5 attending to position 10?

```python
across_heads = attn[0, :, 5, 10]  # shape: [20]
print(f"Strides: {across_heads.stride()}")
```
```
Strides: (1024,)
```

PyTorch reports this 1D tensor as "contiguous" (a 1D tensor is trivially contiguous — there's no second dimension to be out of order with). But look at the stride: 1024. Each element is 1024 floats apart in physical memory. Loading these 20 values means touching 20 scattered cache lines across 80 KB, rather than reading 80 consecutive bytes.

This is why attention implementations carefully choose which dimension to parallelize over. Parallelizing over heads (dim 1) gives each thread a contiguous seq×seq block. Parallelizing over sequence positions would give each thread scattered data.

Here we create two tensors with the exact same values and shape, but different memory layouts — then benchmark the same matmul:

```python
import torch, time

device = "neuron"

# Create a non-contiguous tensor via slicing
big = torch.randn(1024, 2048, device=device, dtype=torch.bfloat16)
A_noncontig = big[:, ::2]                  # shape [1024, 1024], stride=(2048, 2)
A_contig = A_noncontig.contiguous()        # shape [1024, 1024], stride=(1024, 1)

B = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)

print(f"Same values: {torch.equal(A_contig, A_noncontig)}")
print(f"A_contig    stride: {A_contig.stride()}")
print(f"A_noncontig stride: {A_noncontig.stride()}")

# Warmup
for _ in range(3):
    _ = torch.mm(A_contig, B)
    _ = torch.mm(A_noncontig, B)
torch.neuron.synchronize()

# Benchmark
torch.neuron.synchronize()
start = time.time()
for _ in range(100):
    _ = torch.mm(A_contig, B)
torch.neuron.synchronize()
t_contig = (time.time() - start) / 100

torch.neuron.synchronize()
start = time.time()
for _ in range(100):
    _ = torch.mm(A_noncontig, B)
torch.neuron.synchronize()
t_noncontig = (time.time() - start) / 100

print(f"\nContiguous:     {t_contig*1000:.3f}ms")
print(f"Non-contiguous: {t_noncontig*1000:.3f}ms")
print(f"Ratio: {t_noncontig/t_contig:.2f}x")
```

```none
Same values: True
A_contig    stride: (1024, 1)
A_noncontig stride: (2048, 2)

Contiguous:     0.058ms
Non-contiguous: 0.077ms
Ratio: 1.33x
```

**33% slower** — same values, same shape, same operation. The non-contiguous stride forces the DMA engine to gather data with gaps instead of streaming it sequentially from HBM into on-chip memory. Multiply this by the hundreds of matmuls in ESM-2's 33 layers, and layout becomes a real performance factor.

---

## The memory hierarchy on Neuron

We've said "contiguous is faster" — but faster at what, exactly? At moving data through the memory hierarchy that every accelerator shares:

```
┌─────────────────────────────────────────────┐
│  On-chip SRAM (fast, small, ~tens of MB)    │
│  - Neuron: SBUF (State Buffer) + PSUM       │
├─────────────────────────────────────────────┤
│  Device memory (slow, large, ~tens of GB)   │
│  - Neuron: HBM                              │
└─────────────────────────────────────────────┘
```

Every kernel's inner loop is: load a tile from HBM into SRAM, compute, write the result back. The DMA engine that moves tiles works in contiguous bursts (as we saw in Act 4). A well-laid-out tensor means fewer, larger bursts. A poorly-laid-out tensor means many small scattered reads.

This connects back to Chapter 1's fusion story. Fusion eliminates *inter-kernel* HBM round-trips (intermediates stay in SRAM). But layout determines how efficiently each kernel does its *intra-kernel* loads. Both matter. A fused kernel on badly-laid-out data still wastes bandwidth.

---

## The dispatcher cares about layout

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
  ├── device=cpu,    layout=strided  → Intel MKL GEMM (AVX-512)
  ├── device=neuron, layout=strided  → Compile to NEFF → execute on NeuronCore tensor engine
  └── device=neuron, layout=sparse   → fallback to CPU (not yet supported)
```

The dispatcher routes based on device *and* layout. On Neuron, strided (dense) tensors compile to efficient NEFFs. Sparse layouts currently fall back to CPU — the data moves across the bus, runs on MKL, and the result moves back. We'll see this concretely in Chapter 6 when we look at fallback behavior.

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

The kernel doesn't see "a matrix." It sees a pointer, a shape, and strides. How well it can load tiles from HBM into SRAM depends entirely on how the data is laid out.

But in eager mode, each kernel makes its own local decision about layout. No kernel knows what the *next* kernel needs. If kernel A writes in layout X and kernel B needs layout Y, someone pays for the rearrangement.

*What if something could see the whole graph of operations and choose layouts globally?*

That's the compiler. Chapter 3.