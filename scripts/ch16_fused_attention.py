"""
NKI Script 5: Fused Self-Attention — All Engines Working Together
==================================================================
Run on trn2.3xlarge:
    source /workshop/workspace/native_venv/bin/activate
    python scripts/nki_fused_attention.py

This script implements single-head self-attention (Q×K^T → softmax → ×V)
entirely at the ISA level, demonstrating how Tensor Engine, Vector Engine,
and Scalar Engine cooperate within a single kernel.

Versions:
  1. Small (d=128, seq=128): single-tile, all fits in SBUF
  2. Tiled: handles seq > 128 by tiling over sequence dimension

Data flow:
  Q[d,S], K[d,S], V[d,S] → Output[S,d]
  
  Q^T × K → scores[S,S]  (Tensor Engine, matmul)
  softmax(scores)          (Scalar/Vector Engine: max, sub, exp, sum, div)
  scores × V^T → out[S,d] (Tensor Engine, matmul)

Hardware mapping:
  ┌──────────────────────────────────────────────────────┐
  │  nc_matmul (Q^T × K)        → Tensor Engine → PSUM  │
  │  tensor_copy (PSUM → SBUF)  → Vector Engine          │
  │  tensor_reduce (max)         → → PSUM/SBUF           │
  │  tensor_scalar (subtract)    → Scalar Engine          │
  │  activation (exp)            → Scalar Engine          │
  │  tensor_reduce (sum)         → → PSUM/SBUF           │
  │  reciprocal + multiply       → Scalar Engine          │
  │  nc_transpose (scores)       → Tensor Engine → PSUM  │
  │  nc_matmul (scores × V^T)   → Tensor Engine → PSUM  │
  └──────────────────────────────────────────────────────┘
"""

import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


# =============================================================================
# Helper: softmax at ISA level
# =============================================================================

def softmax_isa(data):
    """In-place numerically stable softmax along axis=1 using ISA ops.

    data: [P, F] tensor in SBUF (float32)
    Returns: [P, F] tensor in SBUF with softmax applied.
    """
    P, F = data.shape

    # Step 1: max per row (for numerical stability)
    row_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=row_max, data=data, op=nl.maximum, axis=1)

    # Step 2: subtract max
    shifted = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=shifted, data=data, op0=nl.subtract, operand0=row_max)

    # Step 3: exp
    exp_vals = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=exp_vals, data=shifted, op=nl.exp)

    # Step 4: sum per row
    row_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=row_sum, data=exp_vals, op=nl.add, axis=1)

    # Step 5: reciprocal of sum
    inv_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_sum, data=row_sum)

    # Step 6: multiply exp by 1/sum (broadcasts scalar across free dim)
    result = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=result, data=exp_vals, op0=nl.multiply, operand0=inv_sum)

    return result


# =============================================================================
# Example 1: Single-tile attention (d=128, seq=128)
# =============================================================================

@nki.jit
def attention_single_tile(q_hbm, k_hbm, v_hbm):
    """Self-attention for one head: Q[128,128], K[128,128], V[128,128] → Out[128,128].

    Layout: [d_head, seqlen] with d_head on partition dimension.
    Output: [seqlen, d_head] — standard attention output layout.
    """
    d_head, seqlen = q_hbm.shape
    output = nl.ndarray((seqlen, d_head), dtype=q_hbm.dtype, buffer=nl.shared_hbm)

    # Load Q, K, V into SBUF
    q = nl.ndarray((d_head, seqlen), dtype=q_hbm.dtype, buffer=nl.sbuf)
    k = nl.ndarray((d_head, seqlen), dtype=k_hbm.dtype, buffer=nl.sbuf)
    v = nl.ndarray((d_head, seqlen), dtype=v_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q, src=q_hbm)
    nisa.dma_copy(dst=k, src=k_hbm)
    nisa.dma_copy(dst=v, src=v_hbm)

    # ─── Step 1: Q^T × K → scores[seqlen, seqlen] ───
    # stationary=Q[d,S], moving=K[d,S] → result[S,S]
    # Q is stationary: contraction dim (d=128) on partition, free dim (S) in output rows
    scores_psum = nl.ndarray((seqlen, seqlen), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=scores_psum, stationary=q, moving=k)

    # PSUM → SBUF for softmax
    scores = nl.ndarray((seqlen, seqlen), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=scores, src=scores_psum)

    # ─── Step 2: Softmax (Scalar + Vector Engines) ───
    probs = softmax_isa(scores)

    # ─── Step 3: Transpose for second matmul ───
    # Need scores^T[seqlen, seqlen] as stationary for scores × V^T
    probs_t_psum = nl.ndarray((seqlen, seqlen), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=probs_t_psum, data=probs)
    probs_t = nl.ndarray((seqlen, seqlen), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=probs_t, src=probs_t_psum)  # also casts to bf16

    # Transpose V: [d, S] → [S, d]
    v_t_psum = nl.ndarray((seqlen, d_head), dtype=v.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=v_t_psum, data=v)
    v_t = nl.ndarray((seqlen, d_head), dtype=v.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=v_t, src=v_t_psum)

    # ─── Step 4: scores × V^T → output[seqlen, d_head] ───
    out_psum = nl.ndarray((seqlen, d_head), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=out_psum, stationary=probs_t, moving=v_t)

    # PSUM → SBUF → HBM
    out_sbuf = nl.ndarray((seqlen, d_head), dtype=output.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out_sbuf, src=out_psum)
    nisa.dma_copy(dst=output, src=out_sbuf)

    return output


# =============================================================================
# Example 2: Tiled attention (d=128, seq=N*128)
# =============================================================================

@nki.jit
def attention_tiled(q_hbm, k_hbm, v_hbm):
    """Tiled self-attention: Q[128,S], K[128,S], V[128,S] → Out[S,128].

    Tiles over the sequence dimension. For each output row-tile:
      - Compute QK scores across all K tiles (accumulate full row in SBUF)
      - Softmax over the full row
      - Multiply with V tiles and accumulate output
    """
    d_head, seqlen = q_hbm.shape
    TILE = 128  # uniform tile size — respects gemm_stationary_fmax=128

    assert seqlen % TILE == 0
    num_tiles = seqlen // TILE

    output = nl.ndarray((seqlen, d_head), dtype=q_hbm.dtype, buffer=nl.shared_hbm)

    # Load full K and V (they fit if seq is moderate)
    k = nl.ndarray((d_head, seqlen), dtype=k_hbm.dtype, buffer=nl.sbuf)
    v = nl.ndarray((d_head, seqlen), dtype=v_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=k, src=k_hbm)
    nisa.dma_copy(dst=v, src=v_hbm)

    # Process each output row-tile
    for i_q in nl.affine_range(num_tiles):
        # Load Q tile for this row block: [d_head=128, TILE=128]
        # stationary free dim = TILE = 128 ✓ (≤ gemm_stationary_fmax)
        q_tile = nl.ndarray((d_head, TILE), dtype=q_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_tile,
                      src=q_hbm[0:d_head, i_q * TILE:(i_q + 1) * TILE])

        # Compute full scores row: Q_tile^T × K → [TILE, seqlen]
        # Tile over K's free dimension using TILE (up to 512, moving free dim)
        scores_full = nl.ndarray((TILE, seqlen), dtype=nl.float32, buffer=nl.sbuf)
        for i_k in nl.affine_range(num_tiles):
            scores_psum = nl.ndarray((TILE, TILE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=scores_psum,
                stationary=q_tile,
                moving=k[0:d_head, i_k * TILE:(i_k + 1) * TILE],
            )
            # Copy tile of scores to the right position in scores_full
            nisa.tensor_copy(
                dst=scores_full[0:TILE, i_k * TILE:(i_k + 1) * TILE],
                src=scores_psum,
            )

        # Softmax across full sequence length
        probs = softmax_isa(scores_full)

        # Multiply probs × V^T: accumulate over V tiles
        # probs[TILE, seqlen], V[d, seqlen] → out[TILE, d]
        out_accum = nl.ndarray((TILE, d_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=out_accum, value=0.0)

        for i_v in nl.affine_range(num_tiles):
            # Extract prob tile [TILE, TILE] and transpose it → [TILE, TILE]
            prob_slice = nl.ndarray((TILE, TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=prob_slice,
                src=probs[0:TILE, i_v * TILE:(i_v + 1) * TILE],
            )

            # Transpose prob_slice: need [TILE, TILE] as stationary
            # TILE=512 > stationary_fmax=128, so we keep prob_slice as moving instead
            # Rearrange: out += V_tile × prob_slice^T
            # V_tile[d=128, TILE=512] as stationary (free=TILE=512 > 128!) — won't work
            # Need to further tile or use a different approach:
            # prob_slice[TILE=128, TILE=512] as moving, need stationary with K=128
            # Split TILE into sub-tiles of 128 for the V×prob contraction

            # Sub-tile the V dimension (TILE=512 in chunks of 128)
            TILE_V_K = 128
            for i_vk in nl.affine_range(TILE // TILE_V_K):
                # prob sub-slice [TILE=128, TILE_V_K=128]
                prob_sub = nl.ndarray((TILE, TILE_V_K), dtype=q_hbm.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=prob_sub,
                    src=prob_slice[0:TILE, i_vk * TILE_V_K:(i_vk + 1) * TILE_V_K],
                )

                # V sub-tile [d=128, TILE_V_K=128]
                v_sub = nl.ndarray((d_head, TILE_V_K), dtype=v.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=v_sub,
                    src=v[0:d_head, i_v * TILE + i_vk * TILE_V_K:
                                    i_v * TILE + (i_vk + 1) * TILE_V_K],
                )

                # Matmul: V_sub[d=128, TILE_V_K=128] as stationary (free=128 ✓)
                #         prob_sub[TILE=128, TILE_V_K=128] — need transpose
                # nc_matmul: dst = stationary.T @ moving
                # We want: V_sub.T @ prob_sub_transposed = [TILE_V_K, d].T @ [TILE_V_K, TILE]
                # Actually we want output [TILE, d]:
                # = prob_sub @ V_sub.T = [TILE, TILE_V_K] @ [TILE_V_K, d]
                # In nc_matmul terms: stationary[K, M] moving[K, N] → [M, N]
                # stationary = prob_sub.T[TILE_V_K, TILE], moving = V_sub[d_head... no

                # Simpler: transpose prob_sub to [TILE_V_K, TILE]
                prob_sub_t_psum = nl.ndarray((TILE_V_K, TILE), dtype=prob_sub.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=prob_sub_t_psum, data=prob_sub)
                prob_sub_t = nl.ndarray((TILE_V_K, TILE), dtype=q_hbm.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=prob_sub_t, src=prob_sub_t_psum)

                # nc_matmul: dst[M,N] = stationary[K,M].T @ moving[K,N]
                # stationary = prob_sub_t[TILE_V_K=128, TILE=128] (K=128, M=128 ✓)
                # moving = v_sub[d_head=128, TILE_V_K=128]... no, dims wrong

                # Let's think again:
                # We want: result[TILE, d] += prob_sub[TILE, TILE_V_K] @ V_sub.T[TILE_V_K, d]
                # nc_matmul: result[M, N] = stationary[K, M].T @ moving[K, N]
                # So: K=TILE_V_K=128, M=TILE=128, N=d=128
                # stationary = prob_sub_t[TILE_V_K=128, TILE=128] ✓ (K=128, free=128)
                # moving = v_sub[d_head=128, TILE_V_K=128] — but we need [K=TILE_V_K, N=d]
                # v_sub is [d=128, TILE_V_K=128], we need [TILE_V_K=128, d=128]
                v_sub_t_psum = nl.ndarray((TILE_V_K, d_head), dtype=v_sub.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=v_sub_t_psum, data=v_sub)
                v_sub_t = nl.ndarray((TILE_V_K, d_head), dtype=v_sub.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=v_sub_t, src=v_sub_t_psum)

                # Now: stationary=prob_sub_t[128,128], moving=v_sub_t[128,128]
                partial_psum = nl.ndarray((TILE, d_head), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=partial_psum, stationary=prob_sub_t, moving=v_sub_t)

                partial_sbuf = nl.ndarray((TILE, d_head), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=partial_sbuf, src=partial_psum)
                nisa.tensor_tensor(dst=out_accum, data1=out_accum, data2=partial_sbuf, op=nl.add)

        # Store output tile
        out_cast = nl.ndarray((TILE, d_head), dtype=output.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_cast, src=out_accum)
        nisa.dma_copy(dst=output[i_q * TILE:(i_q + 1) * TILE, 0:d_head],
                      src=out_cast)

    return output


# =============================================================================
# NumPy reference
# =============================================================================

def attention_numpy(q, k, v):
    """Reference: Q[d,S], K[d,S], V[d,S] → Out[S,d]."""
    q_f32 = q.astype(np.float32)
    k_f32 = k.astype(np.float32)
    v_f32 = v.astype(np.float32)

    scores = q_f32.T @ k_f32  # [S, S]
    # Numerically stable softmax
    row_max = scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores - row_max)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    out = probs @ v_f32.T  # [S, d]
    return out


# =============================================================================
# Run and verify
# =============================================================================

def main():
    print("=" * 60)
    print("NKI Script 5: Fused Self-Attention (All Engines)")
    print("=" * 60)

    # Example 1: Single tile
    print("\n[1] Single-tile attention (d=128, seq=128)...")
    d, s = 128, 128
    q = np.random.randn(d, s).astype(np.float16)
    k = np.random.randn(d, s).astype(np.float16)
    v = np.random.randn(d, s).astype(np.float16)

    result = attention_single_tile(q, k, v)
    expected = attention_numpy(q, k, v)
    max_diff = np.abs(result.astype(np.float32) - expected.astype(np.float32)).max()
    assert max_diff < 0.5, f"Single-tile attention: max diff {max_diff}"
    print(f"    ✓ max diff = {max_diff:.4f}")
    print("    Engines used: Tensor(2×matmul + 2×transpose) + Scalar(softmax) + Vector(copy)")

    # Example 2: Tiled
    print("\n[2] Tiled attention (d=128, seq=2048)...")
    d, s = 128, 2048
    q = np.random.randn(d, s).astype(np.float16)
    k = np.random.randn(d, s).astype(np.float16)
    v = np.random.randn(d, s).astype(np.float16)

    result = attention_tiled(q, k, v)
    expected = attention_numpy(q, k, v)
    max_diff = np.abs(result.astype(np.float32) - expected.astype(np.float32)).max()
    assert max_diff < 1.0, f"Tiled attention: max diff {max_diff}"
    tiles = s // 512
    print(f"    ✓ max diff = {max_diff:.4f} ({tiles} seq tiles)")

    print("\n" + "=" * 60)
    print("All Attention examples passed!")
    print("=" * 60)

    # =========================================================================
    # Profiling with NeuronProfiler — full device timeline
    # =========================================================================
    # In Ch16 we need to see WHY a kernel is slow — not just how slow.
    # NeuronProfiler captures a device-level trace showing:
    #   - Which engines are active at each point in time
    #   - Where the Tensor Engine is idle (waiting for data or softmax)
    #   - DMA transfer overlaps with compute
    #
    # This is what motivates pipelining: if the profiler shows Tensor Engine
    # idle during softmax, you know you can overlap them.
    #
    # The output is a directory you open in neuron-explorer for timeline analysis.
    #
    # Three benchmarking approaches across Part V:
    #   Ch14: nki.benchmark()        → raw kernel latency, no framework
    #   Ch15: wrap_nki + sync        → realistic dispatch, compare variants
    #   Ch16: NeuronProfiler         → device timeline, diagnose bottlenecks
    # =========================================================================

    print("\n" + "=" * 60)
    print("Profiling: NeuronProfiler — device timeline for neuron-explorer")
    print("=" * 60)

    import os
    import torch
    from torch_neuronx import wrap_nki
    from torch.profiler import profile, ProfilerActivity
    from torch_neuronx.profiling import NeuronConfig, ProfileMode, NeuronProfiler

    PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ch16_profile_output")
    os.makedirs(PROFILE_DIR, exist_ok=True)

    d, s = 128, 128
    q_t = torch.randn(d, s, dtype=torch.bfloat16, device="neuron")
    k_t = torch.randn(d, s, dtype=torch.bfloat16, device="neuron")
    v_t = torch.randn(d, s, dtype=torch.bfloat16, device="neuron")

    wrapped_attn = wrap_nki(attention_single_tile)

    # Warmup (triggers compilation)
    wrapped_attn(q_t, k_t, v_t)
    torch.neuron.synchronize()

    # Capture device profile
    neuron_config = NeuronConfig(
        modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME],
        profile_output_dir=PROFILE_DIR,
    )

    print("  Capturing device trace...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        experimental_config=neuron_config,
    ) as prof:
        wrapped_attn(q_t, k_t, v_t)
        torch.neuron.synchronize()

    print(f"  ✓ Profile saved to: {PROFILE_DIR}/")
    print("    Open in neuron-explorer to see the timeline:")
    print("    → Which engine is active at each point")
    print("    → Where Tensor Engine is IDLE (optimization opportunity)")
    print("    → DMA overlap with compute")

    print("""
Why NeuronProfiler?
  • Wall clock time tells you HOW SLOW — the profiler tells you WHY
  • Shows engine-level parallelism (or lack thereof)
  • Motivates pipelining: "TE idle during softmax" → overlap them
  • Use this in Ch16 to answer: "which engine is my bottleneck?"
""")

    print("""
Engine utilization in fused attention:
  ┌──────────────────────────────────────────────────────────────┐
  │ Operation          │ ISA call            │ Engine             │
  ├──────────────────────────────────────────────────────────────┤
  │ Q^T × K            │ nc_matmul           │ Tensor Engine      │
  │ max(scores)        │ tensor_reduce(max)  │ → PSUM/SBUF        │
  │ scores - max       │ tensor_scalar(sub)  │ Scalar Engine      │
  │ exp(shifted)       │ activation(exp)     │ Scalar Engine      │
  │ sum(exp)           │ tensor_reduce(add)  │ → PSUM/SBUF        │
  │ 1/sum              │ reciprocal          │ Scalar Engine      │
  │ exp * (1/sum)      │ tensor_scalar(mul)  │ Scalar Engine      │
  │ transpose scores   │ nc_transpose        │ Tensor Engine      │
  │ scores × V^T       │ nc_matmul           │ Tensor Engine      │
  └──────────────────────────────────────────────────────────────┘

Next optimization steps (see Ch 16 Advanced NKI):
  1. Fuse the two matmuls with softmax in a single tiled loop
  2. Software pipeline: tile I+1 matmul overlaps tile I softmax
  3. Delay division to the end (divide smaller output instead)
  4. Downcast scores to BF16 before transpose (halves data movement)
""")


if __name__ == "__main__":
    main()
