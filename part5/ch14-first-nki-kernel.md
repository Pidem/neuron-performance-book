# Your first NKI kernel

*Load data, do some math, store the result. That's every kernel.*

---

## The pattern

Every NKI kernel follows the same four steps:

```
HBM (input tensors)
    │
    ▼  nl.load()
SBUF (on-chip, fast)
    │
    ▼  compute (nl.* or nisa.*)
SBUF (result)
    │
    ▼  nl.store()
HBM (output tensor)
```

That's it. You explicitly move data on-chip, compute on it, and move results back. There's no implicit caching, no magic — you control every byte.

For matmul specifically, there's one extra step: results land in PSUM (the accumulator buffer), so you must copy PSUM → SBUF before storing to HBM.

---

## Hello world: element-wise add

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit
def add_kernel(a_hbm, b_hbm):
    # Step 1: Load from HBM to SBUF
    a_sbuf = nl.load(a_hbm)
    b_sbuf = nl.load(b_hbm)
    
    # Step 2: Compute (runs on vector engine)
    result = a_sbuf + b_sbuf
    
    # Step 3: Allocate output in HBM
    out_hbm = nl.ndarray(a_hbm.shape, dtype=a_hbm.dtype, buffer=nl.hbm)
    
    # Step 4: Store from SBUF to HBM
    nl.store(out_hbm, value=result)
    return out_hbm
```

### Using it in PyTorch

```python
import torch

device = "neuron"
a = torch.randn(128, 512, device=device, dtype=torch.bfloat16)
b = torch.randn(128, 512, device=device, dtype=torch.bfloat16)

# Use the kernel like any other function
result = add_kernel(a, b)

# Verify against PyTorch
expected = a + b
assert torch.allclose(result, expected)
```

The `@nki.jit` decorator makes the kernel callable from PyTorch. The input tensors are on the Neuron device (HBM). The kernel loads them to SBUF, adds them, and stores the result back to HBM. From PyTorch's perspective, it's just a function call.

---

## The tile constraint

The inputs above are shaped `[128, 512]`. This isn't arbitrary:

- **128** = the partition dimension (128 parallel lanes). This is the maximum for the first dimension.
- **512** = the free dimension. Can be up to hardware limits per engine.

This `[128, 512]` tile fits perfectly in SBUF (128 × 512 × 2 bytes = 128 KB in BF16). No tiling needed — the entire tensor IS one tile.

What if your tensor is larger than one tile? You need to loop over tiles. That's Chapter 15.

---

## Hello matmul: the tensor engine

```python
@nki.jit
def matmul_kernel(a_t_hbm, b_hbm):
    """Simple 128x128 @ 128x512 matmul.
    
    a_t_hbm: [128, 128] — already transposed (hardware requirement)
    b_hbm:   [128, 512] — the "moving" matrix
    """
    # Load tiles to SBUF
    a_sbuf = nl.load(a_t_hbm)
    b_sbuf = nl.load(b_hbm)
    
    # Matmul — result lands in PSUM (always FP32)
    result_psum = nl.matmul(a_sbuf, b_sbuf, transpose_x=True)
    
    # Copy from PSUM to SBUF (cast FP32 → BF16)
    result_sbuf = nl.copy(result_psum, dtype=nl.bfloat16)
    
    # Allocate output and store
    out_hbm = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.hbm)
    nl.store(out_hbm, value=result_sbuf)
    return out_hbm
```

Key details:
- The left-hand side (`a_t_hbm`) must be **transposed** before feeding the tensor engine. This is a hardware requirement — the contraction axis must map to the partition dimension.
- `transpose_x=True` tells `nl.matmul` the input is already transposed. If your input isn't transposed, set this to `False` and the kernel will transpose it (at a cost).
- Results accumulate in PSUM in **FP32** — no precision loss during the multiply-accumulate. The cast to BF16 happens only when you copy out.
- Tile sizes for the tensor engine: stationary (left) is 128×128, moving (right) is 128×512.

---

## Verifying correctness: always write tests first

```python
import torch
import numpy as np

def test_matmul_kernel():
    # Ground truth in PyTorch
    a = torch.randn(128, 128, dtype=torch.bfloat16, device="neuron")
    b = torch.randn(128, 512, dtype=torch.bfloat16, device="neuron")
    expected = torch.matmul(a, b)
    
    # NKI kernel (needs transposed LHS)
    a_t = a.T.contiguous()
    result = matmul_kernel(a_t, b)
    
    # Compare with tolerance (BF16 accumulation differs from FP32)
    assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2), \
        f"Max diff: {(result - expected).abs().max()}"
    print("✓ matmul kernel matches PyTorch")

test_matmul_kernel()
```

BF16 requires relaxed tolerance (`atol=1e-2`) because the intermediate precision differs from PyTorch's default FP32 matmul. This isn't a bug — it's the precision tradeoff from Chapter 12.

---

## Profiling your first kernel

Quick latency measurement without the full Neuron Explorer:

```python
bench = nki.benchmark(matmul_kernel, a_t, b)
print(f"P50 latency: {bench.p50_latency_ms:.3f} ms")
print(f"P99 latency: {bench.p99_latency_ms:.3f} ms")
```

Use this for rapid iteration. When you need to understand *why* performance is what it is, switch to Neuron Explorer (Ch 10).

For a 128×128 @ 128×512 matmul: expect MFU in single digits. The kernel is correct but tiny — not enough work to saturate the hardware. Real workloads need tiling across larger matrices, which is where performance engineering begins.

---

## What you learned

| Concept | What it means |
|---------|--------------|
| `@nki.jit` | Makes a Python function a NeuronCore kernel |
| `nl.load()` / `nl.store()` | Explicit DMA: HBM ↔ SBUF |
| `nl.matmul()` | Tensor engine matmul, result in PSUM |
| `nl.copy()` | Move data between on-chip buffers (PSUM → SBUF), can cast dtype |
| `nl.ndarray(..., buffer=nl.hbm)` | Allocate output tensor in HBM |
| Tile constraint | First dimension ≤ 128 (partition), second dimension = free |
| Transpose requirement | Tensor engine LHS must have contraction axis on partition dim |

---

## What's missing

This kernel works on exactly one tile. Real tensors (ESM-2 weights are 1280×1280) don't fit in a single 128×512 tile. You need to:

1. **Partition** the input into tiles
2. **Loop** over tiles, accumulating partial results
3. **Index** into the larger tensor to extract each tile

This is tiling — and it requires understanding NKI's indexing system. That's next.

---

*You can load and store tiles. But how do you decompose a real operation — like a matmul — into tile-sized pieces? And which engine handles which part?*
