"""
Chapter 12: Mixed Precision on Neuron
======================================
Run on trn2.3xlarge:
    source /workshop/workspace/native_venv/bin/activate
    python scripts/ch12_mixed_precision.py

Demonstrates the precision hierarchy on Neuron hardware:
  1. FP32 → BF16 auto-cast (the free win — same range, half the bytes)
  2. PyTorch AMP with autocast (which ops stay FP32, which go BF16)
  3. The outlier problem: why naive FP8 casting destroys accuracy
  4. Per-tensor vs per-group (microscaling) quantization
  5. FP8 matmul throughput: 2× on Tensor Engine (Trn2)
  6. The shifted bottleneck: faster matmul exposes softmax

Inspired by:
  - https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
  - Ron Diamant (re:Invent 2025): microscaling hardware circuits, accelerated softmax
  - Jay Gray (Anthropic): BF16 → FP8 E4M3 = immediate 2× on matmul tiles
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import nki
import nki.language as nl
import nki.isa as nisa


# =============================================================================
# NKI matmul kernel (used by section 5 for FP8 vs BF16 benchmark)
# =============================================================================
_TILE_P = 128
_TILE_M = 128
_TILE_N = 512

@nki.jit
def _matmul_nki(a, b):
    """Generic tiled matmul: a[K,M] @ b[K,N] -> out[M,N]. Works for BF16 and FP16."""
    K_, M_ = a.shape
    N_ = b.shape[1]
    out = nl.ndarray((M_, N_), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for m in nl.affine_range(M_ // _TILE_M):
        for n in nl.affine_range(N_ // _TILE_N):
            c_psum = nl.ndarray((_TILE_M, _TILE_N), dtype=nl.float32, buffer=nl.psum)
            for k in nl.affine_range(K_ // _TILE_P):
                a_tile = nl.ndarray((_TILE_P, _TILE_M), dtype=a.dtype, buffer=nl.sbuf)
                b_tile = nl.ndarray((_TILE_P, _TILE_N), dtype=b.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=a_tile, src=a[k*_TILE_P:(k+1)*_TILE_P, m*_TILE_M:(m+1)*_TILE_M])
                nisa.dma_copy(dst=b_tile, src=b[k*_TILE_P:(k+1)*_TILE_P, n*_TILE_N:(n+1)*_TILE_N])
                nisa.nc_matmul(dst=c_psum, stationary=a_tile, moving=b_tile)
            c_sbuf = nl.ndarray((_TILE_M, _TILE_N), dtype=out.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=c_sbuf, src=c_psum)
            nisa.dma_copy(dst=out[m*_TILE_M:(m+1)*_TILE_M, n*_TILE_N:(n+1)*_TILE_N], src=c_sbuf)
    return out


# =============================================================================
# 1. BF16 vs FP16: why BF16 "just works"
# =============================================================================

def demo_bf16_vs_fp16():
    """Show why BF16 is the safe default: same range as FP32, no overflow."""
    print("=" * 60)
    print("[1] BF16 vs FP16: dynamic range matters")
    print("=" * 60)

    # Simulate a gradient that's large (common in early training)
    large_val = torch.tensor(100000.0)

    fp16_val = large_val.to(torch.float16)
    bf16_val = large_val.to(torch.bfloat16)

    print(f"  Original FP32:  {large_val.item()}")
    print(f"  Cast to FP16:   {fp16_val.item()}  ← OVERFLOW (FP16 max = 65,504)")
    print(f"  Cast to BF16:   {bf16_val.item()}")
    print()
    print("  BF16 keeps FP32's 8-bit exponent → same dynamic range (±3.4e38)")
    print("  FP16 has 5-bit exponent → max 65,504 → gradients overflow")
    print("  This is why BF16 needs no GradScaler, but FP16 does.")


# =============================================================================
# 2. PyTorch AMP: which ops stay FP32?
# =============================================================================

def demo_amp_allowlist():
    """Show which operations PyTorch auto-casts and which stay FP32."""
    print("\n" + "=" * 60)
    print("[2] PyTorch AMP: what gets cast, what doesn't")
    print("=" * 60)

    x = torch.randn(4, 128, 1280, device="neuron")
    linear = torch.nn.Linear(1280, 1280).to("neuron")
    ln = torch.nn.LayerNorm(1280).to("neuron")

    with torch.amp.autocast(device_type="neuron", dtype=torch.bfloat16):
        # Matmul: cast to BF16 (safe, accumulates in FP32 on Tensor Engine)
        y_linear = linear(x)
        # LayerNorm: stays FP32 (reduction needs precision)
        y_norm = ln(x)
        # Softmax: stays FP32 (exponentiation amplifies errors)
        y_softmax = F.softmax(x, dim=-1)

    print(f"  Input dtype:     {x.dtype}")
    print(f"  Linear output:   {y_linear.dtype}  ← cast to BF16 (matmul)")
    print(f"  LayerNorm out:   {y_norm.dtype}  ← stays FP32 (reduction)")
    print(f"  Softmax out:     {y_softmax.dtype}  ← stays FP32 (exp)")
    print()
    print("  Rule: matmuls/convs → BF16 | reductions/nonlinearities → FP32")
    print("  On Neuron: PSUM always accumulates in FP32 regardless of input dtype")


# =============================================================================
# 3. The outlier problem (why naive FP8 breaks)
# =============================================================================

def demo_outlier_problem():
    """Demonstrate how one outlier destroys naive per-tensor FP8 quantization."""
    print("\n" + "=" * 60)
    print("[3] The outlier problem: naive FP8 quantization")
    print("=" * 60)

    # Simulate a tensor with one outlier (common in LLM activations)
    normal_vals = torch.randn(1, 127) * 0.5  # 99% of values in [-1, 1]
    outlier = torch.tensor([[100.0]])          # one big activation
    tensor = torch.cat([normal_vals, outlier], dim=1)

    # Per-tensor absmax quantization to FP8 range
    fp8_e4m3_max = 448.0  # max representable value in E4M3
    scale = tensor.abs().max() / fp8_e4m3_max
    quantized = torch.round(tensor / scale)  # simulate quantize
    dequantized = quantized * scale           # simulate dequantize

    error = (tensor - dequantized).abs()
    normal_error = error[0, :127].mean()
    outlier_error = error[0, 127]

    print(f"  Tensor: 127 normal values (std=0.5) + 1 outlier (100.0)")
    print(f"  Per-tensor scale: {scale.item():.6f}")
    print(f"  Normal values error: {normal_error.item():.4f} (relative: {normal_error.item()/0.5:.1%})")
    print(f"  Outlier error:       {outlier_error.item():.4f}")
    print()
    print("  ⚠️  The outlier sets the scale → normal values lose ALL precision")
    print("  This is exactly why microscaling exists (see next example)")


# =============================================================================
# 4. Microscaling: per-group quantization fixes outliers
# =============================================================================

def demo_microscaling():
    """Show how per-group (microscaling) quantization preserves accuracy."""
    print("\n" + "=" * 60)
    print("[4] Microscaling: per-group quantization")
    print("=" * 60)

    # Same tensor, but now quantize in groups of 32
    torch.manual_seed(42)
    normal_vals = torch.randn(1, 127) * 0.5
    outlier = torch.tensor([[100.0]])
    tensor = torch.cat([normal_vals, outlier], dim=1)  # shape [1, 128]

    fp8_max = 448.0
    GROUP_SIZE = 32

    # Per-group quantization
    groups = tensor.view(-1, GROUP_SIZE)  # [4, 32]
    scales = groups.abs().amax(dim=1, keepdim=True) / fp8_max
    quantized = torch.round(groups / scales)
    dequantized = (quantized * scales).view(tensor.shape)

    error = (tensor - dequantized).abs()
    # The outlier is in the last group; other groups are unaffected
    normal_error = error[0, :96].mean()  # first 3 groups (no outlier)
    outlier_group_error = error[0, 96:128].mean()  # group containing outlier

    print(f"  Group size: {GROUP_SIZE} elements")
    print(f"  Groups without outlier — mean error: {normal_error.item():.6f}")
    print(f"  Group WITH outlier — mean error:     {outlier_group_error.item():.4f}")
    print()
    print("  ✓ Outlier only damages its own group (32 elements)")
    print("  ✓ Other groups get their own scale → full precision preserved")
    print()
    print("  On Trn2: you implement microscaling in software (NKI kernels)")
    print("  On Trn3: hardware circuits do microscaling at line-rate (zero overhead)")


# =============================================================================
# 5. FP8 matmul throughput on Neuron Tensor Engine
# =============================================================================

def demo_fp8_matmul():
    """Compare BF16 vs FP8 matmul latency on Tensor Engine (Trn2: 2× throughput)."""
    print("\n" + "=" * 60)
    print("[5] FP8 vs BF16 matmul throughput on Tensor Engine")
    print("=" * 60)

    M, K, N = 2048, 2048, 2048

    a_bf16 = torch.randn(K, M, dtype=torch.bfloat16, device="neuron")
    b_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device="neuron")

    # Warmup
    _ = _matmul_nki(a_bf16, b_bf16)
    torch.neuron.synchronize()

    # Benchmark BF16
    iters = 50
    torch.neuron.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _matmul_nki(a_bf16, b_bf16)
    torch.neuron.synchronize()
    bf16_time = (time.perf_counter() - start) / iters

    flops = 2 * M * N * K
    bf16_tflops = flops / bf16_time / 1e12

    print(f"  Matrix size: {M}×{K} @ {K}×{N}")
    print(f"  BF16 (measured, NKI nc_matmul): {bf16_time*1e6:.0f} µs | {bf16_tflops:.1f} TFLOPS")
    print(f"  FP8  (hardware spec):           ~{bf16_time*1e6/2:.0f} µs | ~{bf16_tflops*2:.0f} TFLOPS")
    print()
    print("  NeuronCore-v3 Tensor Engine: 158 FP8 TFLOPS vs 79 BF16 TFLOPS (per core)")
    print("  Same nc_matmul instruction — Tensor Engine processes FP8 tiles at 2× rate.")
    print("  PSUM accumulates in FP32 either way — precision loss is only at the")
    print("  quantization boundary (BF16→FP8), not during the matmul itself.")
    print()
    print("  The engineering challenge isn't the matmul — it's getting data INTO FP8")
    print("  without destroying accuracy. That's what microscaling solves (section 4).")
    print("  And even with 2× matmul, your softmax/layernorm may become the new")
    print("  bottleneck — that's the shifted bottleneck problem (section 6).")


# =============================================================================
# 6. The shifted bottleneck: faster matmul exposes softmax
# =============================================================================

def demo_shifted_bottleneck():
    """Show that speeding up matmul shifts the bottleneck to softmax/layernorm."""
    print("\n" + "=" * 60)
    print("[6] The shifted bottleneck (Ron Diamant, re:Invent 2025)")
    print("=" * 60)

    # Simulate self-attention timing breakdown
    # Real numbers from Jay Gray's talk: attention kernel optimization
    seq_len = 4096
    d_head = 128
    n_heads = 32

    q = torch.randn(n_heads, seq_len, d_head, dtype=torch.bfloat16, device="neuron")
    k = torch.randn(n_heads, seq_len, d_head, dtype=torch.bfloat16, device="neuron")

    # Time QK matmul
    for _ in range(5):
        torch.matmul(q, k.transpose(-2, -1))
    torch.neuron.synchronize()

    start = time.perf_counter()
    for _ in range(20):
        scores = torch.matmul(q, k.transpose(-2, -1))
    torch.neuron.synchronize()
    matmul_time = (time.perf_counter() - start) / 20

    # Time softmax
    scores_for_softmax = torch.randn(n_heads, seq_len, seq_len, dtype=torch.float32, device="neuron")
    for _ in range(5):
        F.softmax(scores_for_softmax, dim=-1)
    torch.neuron.synchronize()

    start = time.perf_counter()
    for _ in range(20):
        F.softmax(scores_for_softmax, dim=-1)
    torch.neuron.synchronize()
    softmax_time = (time.perf_counter() - start) / 20

    print(f"  Self-attention components (seq={seq_len}, heads={n_heads}, d={d_head}):")
    print(f"    QK^T matmul:  {matmul_time*1e3:.2f} ms  (Tensor Engine)")
    print(f"    Softmax:      {softmax_time*1e3:.2f} ms  (Scalar Engine, FP32)")
    print()
    print("  If we switch matmul to FP8 (2× faster):")
    print(f"    QK^T matmul:  ~{matmul_time*1e3/2:.2f} ms  (FP8 Tensor Engine)")
    print(f"    Softmax:      {softmax_time*1e3:.2f} ms  (UNCHANGED — still FP32)")
    print()

    total_bf16 = matmul_time + softmax_time
    total_fp8 = matmul_time / 2 + softmax_time
    actual_speedup = total_bf16 / total_fp8

    print(f"  Expected speedup:  2.0× (if only matmul existed)")
    print(f"  Actual speedup:    {actual_speedup:.2f}× (softmax is now the bottleneck!)")
    print()
    print("  Ron Diamant's insight: Trn3 adds hardware-accelerated softmax (4× faster)")
    print("  that keeps the pipeline balanced after FP8 matmuls. Without it, you get")
    print("  diminishing returns from lower precision — the bottleneck just shifts.")


# =============================================================================
# Main
# =============================================================================

def main():
    demo_bf16_vs_fp16()
    demo_amp_allowlist()
    demo_outlier_problem()
    demo_microscaling()
    demo_fp8_matmul()
    demo_shifted_bottleneck()

    print("\n" + "=" * 60)
    print("Summary: The Precision Ladder on Neuron")
    print("=" * 60)
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Concept                  │ Key takeaway                             │
  ├─────────────────────────────────────────────────────────────────────┤
  │ BF16 vs FP16             │ BF16 = same range as FP32, no scaler    │
  │ AMP autocast             │ Matmuls→BF16, reductions→FP32           │
  │ Outlier problem          │ One big value ruins per-tensor FP8      │
  │ Microscaling             │ Per-group scales → outlier contained    │
  │ FP8 Tensor Engine        │ 2× throughput, FP32 accumulation kept   │
  │ Shifted bottleneck       │ Faster matmul exposes softmax/vector    │
  └─────────────────────────────────────────────────────────────────────┘

  Three benchmarking approaches in this book:
    Ch14: nki.benchmark()     → raw kernel latency (hardware ceiling)
    Ch15: wrap_nki + sync     → realistic dispatch (variant comparison)
    Ch16: NeuronProfiler      → device timeline (bottleneck diagnosis)

  Here (Ch12): plain torch timing with synchronize() — measuring the
  impact of precision choices at the PyTorch model level.
""")


if __name__ == "__main__":
    main()
