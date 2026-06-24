"""
NKI Script 1: DMA Basics — Explicit Memory Movement
====================================================
Run on trn2.3xlarge:
    source /workshop/workspace/native_venv/bin/activate
    python scripts/ch14_dma_basics.py

This script teaches explicit data movement between HBM (off-chip) and SBUF (on-chip).

Key concepts:
- HBM = High Bandwidth Memory (off-chip, large, slow)
- SBUF = State Buffer (on-chip SRAM, small, fast)
- Every kernel must explicitly move data: HBM → SBUF → compute → SBUF → HBM
- The partition dimension (first axis) ≤ 128
- nl.load / nl.store are the high-level DMA primitives
- nisa.dma_copy is the ISA-level equivalent (1:1 with hardware instruction)

Two execution modes:
- @nki.jit + torch tensors on device='neuron': correctness testing
- @nki.benchmark (decorator): latency measurement only (output is NOT valid)
"""

import torch
import nki
import nki.language as nl
import nki.isa as nisa


# =============================================================================
# Example 1: Simple copy — load one tile, store it back unchanged
# =============================================================================

@nki.jit
def dma_identity(input_hbm):
    """Load a tile from HBM to SBUF, then store it back. The simplest possible kernel."""
    P, F = input_hbm.shape
    output_hbm = nl.ndarray((P, F), dtype=input_hbm.dtype, buffer=nl.shared_hbm)

    # HBM → SBUF
    tile = nl.load(input_hbm[0:P, 0:F])

    # SBUF → HBM
    nl.store(output_hbm[0:P, 0:F], value=tile)

    return output_hbm


# =============================================================================
# Example 2: Tiled copy — handle tensors larger than one tile
# =============================================================================

@nki.jit
def dma_tiled_copy(input_hbm):
    """Copy a large tensor tile-by-tile. Demonstrates slicing for DMA.

    input_hbm shape: [M, N] where M is a multiple of 128, N multiple of 512.
    """
    M, N = input_hbm.shape
    TILE_P = 128  # partition dimension max
    TILE_F = 512  # free dimension (arbitrary, but 512 is a good default)

    output_hbm = nl.ndarray((M, N), dtype=input_hbm.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_P):
        for n in nl.affine_range(N // TILE_F):
            # Load one tile from HBM → SBUF
            tile = nl.load(input_hbm[m * TILE_P:(m + 1) * TILE_P,
                                     n * TILE_F:(n + 1) * TILE_F])

            # Store tile from SBUF → HBM
            nl.store(output_hbm[m * TILE_P:(m + 1) * TILE_P,
                                n * TILE_F:(n + 1) * TILE_F], value=tile)

    return output_hbm


# =============================================================================
# Example 3: ISA-level DMA with dtype cast
# =============================================================================

@nki.jit
def dma_cast(input_f32):
    """Load FP32 data, cast to BF16 on-chip, store back.
    Shows nisa.dma_copy (ISA-level) + nisa.tensor_copy for casting."""
    P, F = input_f32.shape
    output_bf16 = nl.ndarray((P, F), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # Load as FP32 using ISA-level DMA
    tile_f32 = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile_f32, src=input_f32[0:P, 0:F])

    # Cast FP32 → BF16 in SBUF (Vector Engine handles the cast)
    tile_bf16 = nl.ndarray((P, F), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tile_bf16, src=tile_f32)

    # Store the BF16 result
    nisa.dma_copy(dst=output_bf16[0:P, 0:F], src=tile_bf16)

    return output_bf16


# =============================================================================
# Example 4: memset — initialize SBUF without loading from HBM
# =============================================================================

@nki.jit
def dma_memset_and_accumulate(input_hbm):
    """Zero-initialize an accumulator in SBUF, then add input to it.

    Shows nisa.memset — useful for initializing accumulators before reduction loops.
    """
    P, F = input_hbm.shape
    output_hbm = nl.ndarray((P, F), dtype=input_hbm.dtype, buffer=nl.shared_hbm)

    # Initialize accumulator to zero (no HBM read needed)
    accum = nl.ndarray((P, F), dtype=input_hbm.dtype, buffer=nl.sbuf)
    nisa.memset(dst=accum, value=0.0)

    # Load input
    tile = nl.ndarray((P, F), dtype=input_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=input_hbm[0:P, 0:F])

    # Add input to accumulator (element-wise on Vector Engine)
    nisa.tensor_tensor(dst=accum, data1=accum, data2=tile, op=nl.add)

    # Store result
    nisa.dma_copy(dst=output_hbm[0:P, 0:F], src=accum)

    return output_hbm


# =============================================================================
# Run and verify all examples (correctness via torch tensors)
# =============================================================================

def main():
    torch.manual_seed(42)
    device = torch.device('neuron')

    print("=" * 60)
    print("NKI Script 1: DMA Basics")
    print("=" * 60)

    # Example 1: Identity copy
    print("\n[1] Identity copy (128x512 BF16)...")
    x = torch.randn(128, 512, dtype=torch.bfloat16, device=device)
    result = dma_identity(x)
    max_diff = (result.float() - x.float()).abs().max().item()
    assert max_diff < 1e-5, f"Identity failed! max_diff={max_diff}"
    print(f"    ✓ Data survives HBM → SBUF → HBM round-trip (max_diff={max_diff})")

    # Example 2: Tiled copy (larger than one tile)
    print("\n[2] Tiled copy (512x1024 BF16)...")
    x = torch.randn(512, 1024, dtype=torch.bfloat16, device=device)
    result = dma_tiled_copy(x)
    max_diff = (result.float() - x.float()).abs().max().item()
    assert max_diff < 1e-5, f"Tiled copy failed! max_diff={max_diff}"
    print(f"    ✓ 512x1024 copied tile-by-tile (4×2 = 8 tiles, max_diff={max_diff})")

    # Example 3: DMA with cast
    print("\n[3] FP32 → BF16 cast during transfer...")
    x = torch.randn(128, 512, dtype=torch.float32, device=device)
    result = dma_cast(x)
    expected = x.bfloat16()
    max_diff = (result.float() - expected.float()).abs().max().item()
    assert max_diff < 1e-2, f"Cast failed! max_diff={max_diff}"
    print(f"    ✓ FP32→BF16 cast on-chip (max_diff={max_diff:.6f})")

    # Example 4: memset + accumulate
    print("\n[4] memset zero + accumulate...")
    x = torch.randn(128, 512, dtype=torch.bfloat16, device=device)
    result = dma_memset_and_accumulate(x)
    max_diff = (result.float() - x.float()).abs().max().item()
    assert max_diff < 1e-3, f"memset+accum failed! max_diff={max_diff}"
    print(f"    ✓ 0 + x = x (memset initialized accumulator correctly, max_diff={max_diff:.6f})")

    # =========================================================================
    print("\n" + "=" * 60)
    print("All DMA examples passed!")
    print("=" * 60)
    print("""
Key takeaways:
  • nl.load() / nl.store() move data between HBM and SBUF
  • nisa.dma_copy() is the ISA-level equivalent (same hardware instruction)
  • Tiles are [P, F] where P ≤ 128 (partition dim), F = free dim
  • For large tensors: loop over tiles with nl.affine_range
  • nisa.memset() initializes SBUF without touching HBM
  • All memory movement is EXPLICIT — no caching, no magic
""")


if __name__ == "__main__":
    main()
