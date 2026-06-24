"""
NKI Script 3: Tensor Engine — Matrix Multiplication with nisa.nc_matmul
========================================================================
Run on trn2.3xlarge:
    source /workshop/workspace/native_venv/bin/activate
    python scripts/nki_matmul_isa.py

Progressive matmul implementations from basic to tiled using nisa (ISA-level):
  1. Fixed-size single-tile matmul
  2. Tiled matmul for arbitrary dimensions
  3. Tiled matmul with hoisted loads (reuse optimization)

Key concepts:
- nisa.nc_matmul: one-to-one with hardware Tensor Engine instruction
- Stationary (LHS): must be transposed, shape [K, M] with K on partition dim
- Moving (RHS): shape [K, N] with K on partition dim
- Result always lands in PSUM in FP32
- Tile sizes: TILE_K=128 (pmax), TILE_M=128 (gemm_stationary_fmax), TILE_N=512 (gemm_moving_fmax)
"""

import numpy as np
import time
import nki
import nki.language as nl
import nki.isa as nisa


# =============================================================================
# Example 1: Single-tile matmul (128×128 @ 128×512 = 128×512)
# =============================================================================

@nki.jit
def matmul_single_tile(lhsT_hbm, rhs_hbm):
    """Minimal matmul: one tile each for stationary and moving.

    lhsT_hbm: [128, 128] — LHS transposed (K=128, M=128)
    rhs_hbm:  [128, 512] — RHS (K=128, N=512)
    result:   [128, 512]
    """
    K, M = lhsT_hbm.shape
    K_, N = rhs_hbm.shape

    result = nl.ndarray((M, N), dtype=lhsT_hbm.dtype, buffer=nl.shared_hbm)

    # Load both inputs HBM → SBUF
    lhs_tile = nl.ndarray((K, M), dtype=lhsT_hbm.dtype, buffer=nl.sbuf)
    rhs_tile = nl.ndarray((K_, N), dtype=rhs_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=lhs_tile, src=lhsT_hbm)
    nisa.dma_copy(dst=rhs_tile, src=rhs_hbm)

    # Tensor Engine: matmul → result in PSUM (FP32)
    res_psum = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=res_psum, stationary=lhs_tile, moving=rhs_tile)

    # PSUM → SBUF (cast to output dtype)
    res_sbuf = nl.ndarray((M, N), dtype=result.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=res_sbuf, src=res_psum)

    # SBUF → HBM
    nisa.dma_copy(dst=result, src=res_sbuf)
    return result


# =============================================================================
# Example 2: Tiled matmul (arbitrary M, N, K — all multiples of tile sizes)
# =============================================================================

@nki.jit
def matmul_tiled(lhsT_hbm, rhs_hbm):
    """Tiled matmul supporting arbitrary dimensions.

    lhsT_hbm: [K, M] — K and M multiples of 128
    rhs_hbm:  [K, N] — K multiple of 128, N multiple of 512
    result:   [M, N]

    The key insight: loop over (m, n) output tiles, accumulate partial products
    over the K (contraction) dimension in PSUM.
    """
    K, M = lhsT_hbm.shape
    K_, N = rhs_hbm.shape

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax                   # 128
    TILE_N = nl.tile_size.gemm_moving_fmax       # 512

    result = nl.ndarray((M, N), dtype=lhsT_hbm.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            # Accumulator in PSUM — accumulates across K tiles
            res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            for k in nl.affine_range(K // TILE_K):
                # Load tiles
                lhs_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT_hbm.dtype, buffer=nl.sbuf)
                rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs_hbm.dtype, buffer=nl.sbuf)

                nisa.dma_copy(dst=lhs_tile,
                              src=lhsT_hbm[k * TILE_K:(k + 1) * TILE_K,
                                           m * TILE_M:(m + 1) * TILE_M])
                nisa.dma_copy(dst=rhs_tile,
                              src=rhs_hbm[k * TILE_K:(k + 1) * TILE_K,
                                          n * TILE_N:(n + 1) * TILE_N])

                # Accumulate partial matmul in PSUM
                nisa.nc_matmul(dst=res_psum, stationary=lhs_tile, moving=rhs_tile)

            # Copy accumulated result PSUM → SBUF → HBM
            res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=result.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res_sbuf, src=res_psum)
            nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                                     n * TILE_N:(n + 1) * TILE_N],
                          src=res_sbuf)

    return result


# =============================================================================
# Example 3: Tiled matmul with hoisted LHS load (data reuse optimization)
# =============================================================================

@nki.jit
def matmul_hoist_lhs(lhsT_hbm, rhs_hbm):
    """Tiled matmul reusing LHS tiles across the N loop.

    Optimization: for each M tile, load ALL K tiles of LHS once,
    then reuse them across all N tiles. Reduces HBM→SBUF traffic.

    lhsT_hbm: [K, M], rhs_hbm: [K, N] → result: [M, N]
    """
    K, M = lhsT_hbm.shape
    K_, N = rhs_hbm.shape

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax                   # 128
    TILE_N = nl.tile_size.gemm_moving_fmax       # 512

    result = nl.ndarray((M, N), dtype=lhsT_hbm.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        # Hoist LHS load: load all K tiles for this M-row ONCE
        lhs_tiles = []
        for k in nl.affine_range(K // TILE_K):
            lhs_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=lhs_tile,
                          src=lhsT_hbm[k * TILE_K:(k + 1) * TILE_K,
                                       m * TILE_M:(m + 1) * TILE_M])
            lhs_tiles.append(lhs_tile)

        # Now iterate over N — reusing LHS tiles (no re-load!)
        for n in nl.affine_range(N // TILE_N):
            res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            for k in nl.affine_range(K // TILE_K):
                rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs_hbm.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=rhs_tile,
                              src=rhs_hbm[k * TILE_K:(k + 1) * TILE_K,
                                          n * TILE_N:(n + 1) * TILE_N])

                nisa.nc_matmul(dst=res_psum, stationary=lhs_tiles[k], moving=rhs_tile)

            res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=result.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res_sbuf, src=res_psum)
            nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                                     n * TILE_N:(n + 1) * TILE_N],
                          src=res_sbuf)

    return result


# =============================================================================
# Run and verify
# =============================================================================

def verify_matmul(name, kernel, lhsT, rhs, atol=1e-1):
    """Run kernel and compare against numpy reference."""
    result = kernel(lhsT, rhs)
    # numpy reference: (lhsT.T) @ rhs = M×K @ K×N = M×N
    expected = (lhsT.astype(np.float32).T @ rhs.astype(np.float32))
    expected = expected.astype(result.dtype)
    max_diff = np.abs(result.astype(np.float32) - expected.astype(np.float32)).max()
    assert max_diff < atol, f"{name}: max diff {max_diff} > {atol}"
    return max_diff


def main():
    print("=" * 60)
    print("NKI Script 3: Tensor Engine — nc_matmul")
    print("=" * 60)

    # Example 1: single tile
    print("\n[1] Single-tile matmul (128×128 @ 128×512)...")
    lhsT = np.random.randn(128, 128).astype(np.float16)
    rhs = np.random.randn(128, 512).astype(np.float16)
    diff = verify_matmul("single_tile", matmul_single_tile, lhsT, rhs)
    print(f"    ✓ max diff = {diff:.6f}")

    # Example 2: tiled matmul
    print("\n[2] Tiled matmul (512×256 @ 512×1024 → 256×1024)...")
    K, M, N = 512, 256, 1024
    lhsT = np.random.randn(K, M).astype(np.float16)
    rhs = np.random.randn(K, N).astype(np.float16)
    diff = verify_matmul("tiled", matmul_tiled, lhsT, rhs, atol=1.0)
    tiles = (M // 128) * (N // 512) * (K // 128)
    print(f"    ✓ max diff = {diff:.4f} ({tiles} tile matmuls)")

    # Example 3: hoisted LHS
    print("\n[3] Hoisted-LHS matmul (same dims, fewer HBM reads)...")
    diff = verify_matmul("hoist_lhs", matmul_hoist_lhs, lhsT, rhs, atol=1.0)
    lhs_loads_naive = (M // 128) * (N // 512) * (K // 128)
    lhs_loads_hoisted = (M // 128) * (K // 128)
    print(f"    ✓ max diff = {diff:.4f}")
    print(f"    LHS tile loads: {lhs_loads_naive} (naive) → {lhs_loads_hoisted} (hoisted)")
    print(f"    DMA traffic reduction: {1 - lhs_loads_hoisted/lhs_loads_naive:.0%}")

    print("\n" + "=" * 60)
    print("All Tensor Engine examples passed!")
    print("=" * 60)

    # =========================================================================
    # Benchmarking with wrap_nki + torch.neuron.synchronize()
    # =========================================================================
    # In Ch15 we compare kernel VARIANTS. We need to measure them the way your
    # model will actually call them — through PyTorch's dispatch path.
    #
    # wrap_nki() makes an NKI kernel callable as a PyTorch function.
    # synchronize() ensures we measure actual device completion, not just
    # how fast Python can queue work (see Ch1 async execution).
    #
    # The difference between nki.benchmark() (Ch14) and this approach:
    #   nki.benchmark() = kernel only, no dispatch overhead
    #   wrap_nki + sync = includes dispatch overhead (what your model sees)
    # =========================================================================

    print("\n" + "=" * 60)
    print("Benchmarking: wrap_nki + synchronize — comparing variants")
    print("=" * 60)

    import torch
    from torch_neuronx import wrap_nki

    K, M, N = 1024, 512, 1024
    lhsT_t = torch.randn(K, M, dtype=torch.bfloat16, device="neuron")
    rhs_t = torch.randn(K, N, dtype=torch.bfloat16, device="neuron")

    WARMUP, ITERS = 10, 100

    for name, kernel in [("tiled", matmul_tiled), ("hoist_lhs", matmul_hoist_lhs)]:
        wrapped = wrap_nki(kernel)

        # Warmup (includes first-call compilation)
        for _ in range(WARMUP):
            wrapped(lhsT_t, rhs_t)
        torch.neuron.synchronize()

        # Timed run
        start = time.perf_counter()
        for _ in range(ITERS):
            wrapped(lhsT_t, rhs_t)
        torch.neuron.synchronize()
        elapsed = time.perf_counter() - start

        avg_us = elapsed / ITERS * 1e6
        # FLOPS = 2*M*N*K (multiply-accumulate)
        flops = 2 * M * N * K
        tflops = flops / (elapsed / ITERS) / 1e12
        print(f"  {name:<12}: {avg_us:>8.1f} µs/iter | {tflops:.2f} TFLOPS")

    print("""
Why wrap_nki + synchronize?
  • Measures what your MODEL will actually experience (dispatch + execution)
  • Lets you A/B test kernel variants on the same inputs
  • synchronize() is CRITICAL — without it you only measure queue time
  • Use this in Ch15 to answer: "which tiling strategy is faster in practice?"
""")

    print("""
Key takeaways:
  • nisa.nc_matmul(dst, stationary, moving) — one Tensor Engine instruction
  • Stationary (LHS) shape: [TILE_K=128, TILE_M=128]
  • Moving (RHS) shape: [TILE_K=128, TILE_N=512]
  • Result ALWAYS in PSUM (FP32) — must tensor_copy to SBUF before store
  • Tiling: loop over M, N, K tiles; accumulate partial products in PSUM
  • Optimization: hoist loads to outer loops → reuse tiles, cut DMA traffic

Optimization progression:
  Basic tiled     → correct, but redundant loads
  Hoist LHS       → LHS tiles loaded once per M-row, reused across N
  Block free dims → load larger contiguous blocks for better DMA efficiency
  Full blocking   → block M, N, K; result-stationary in SBUF; coalesced stores
""")


if __name__ == "__main__":
    main()
