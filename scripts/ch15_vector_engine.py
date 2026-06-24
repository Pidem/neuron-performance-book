"""
NKI Script 2: Vector & Scalar Engine Operations
================================================
Run on trn2.3xlarge:
    source /workshop/workspace/native_venv/bin/activate
    python scripts/nki_vector_engine.py

This script teaches element-wise and reduction operations using nisa ISA calls.
These map to the Vector Engine (element-wise) and Scalar Engine (reductions,
activations, scalar math) on NeuronCore.

Key concepts:
- nisa.tensor_tensor: element-wise binary ops (add, mul, sub) → Vector Engine
- nisa.tensor_scalar: element-wise op with a scalar constant → Scalar Engine
- nisa.tensor_reduce: reduction along an axis (sum, max) → produces PSUM result
- nisa.activation: transcendentals (exp, rsqrt, tanh, sigmoid) → Scalar Engine
- PSUM: the accumulator buffer (FP32 only), result of reductions and matmuls
"""

import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


# =============================================================================
# Example 1: Element-wise add (Vector Engine)
# =============================================================================

@nki.jit
def vector_add(a_hbm, b_hbm):
    """Element-wise addition: a + b using nisa.tensor_tensor."""
    P, F = a_hbm.shape
    output = nl.ndarray((P, F), dtype=a_hbm.dtype, buffer=nl.shared_hbm)

    a = nl.ndarray((P, F), dtype=a_hbm.dtype, buffer=nl.sbuf)
    b = nl.ndarray((P, F), dtype=b_hbm.dtype, buffer=nl.sbuf)
    c = nl.ndarray((P, F), dtype=a_hbm.dtype, buffer=nl.sbuf)

    nisa.dma_copy(dst=a, src=a_hbm)
    nisa.dma_copy(dst=b, src=b_hbm)

    # Vector Engine: element-wise binary operation
    nisa.tensor_tensor(dst=c, data1=a, data2=b, op=nl.add)

    nisa.dma_copy(dst=output, src=c)
    return output


# =============================================================================
# Example 2: Scale by constant (Scalar Engine — tensor_scalar)
# =============================================================================

@nki.jit
def scalar_multiply(x_hbm, scale: float):
    """Multiply every element by a scalar constant using nisa.tensor_scalar.

    tensor_scalar supports chaining two operations:
      dst = op1(op0(data, operand0), operand1)
    Here we just use op0=multiply with no op1.
    """
    P, F = x_hbm.shape
    output = nl.ndarray((P, F), dtype=x_hbm.dtype, buffer=nl.shared_hbm)

    x = nl.ndarray((P, F), dtype=x_hbm.dtype, buffer=nl.sbuf)
    y = nl.ndarray((P, F), dtype=x_hbm.dtype, buffer=nl.sbuf)

    nisa.dma_copy(dst=x, src=x_hbm)

    # Scalar Engine: multiply all elements by a constant
    nisa.tensor_scalar(dst=y, data=x, op0=nl.multiply, operand0=scale)

    nisa.dma_copy(dst=output, src=y)
    return output


# =============================================================================
# Example 3: Fused multiply-add with tensor_scalar (two ops in one instruction)
# =============================================================================

@nki.jit
def fused_scale_shift(x_hbm, scale: float, shift: float):
    """Compute x * scale + shift in ONE instruction.

    nisa.tensor_scalar can chain: dst = op1(op0(data, operand0), operand1)
    This avoids a separate add instruction.
    """
    P, F = x_hbm.shape
    output = nl.ndarray((P, F), dtype=x_hbm.dtype, buffer=nl.shared_hbm)

    x = nl.ndarray((P, F), dtype=x_hbm.dtype, buffer=nl.sbuf)
    y = nl.ndarray((P, F), dtype=x_hbm.dtype, buffer=nl.sbuf)

    nisa.dma_copy(dst=x, src=x_hbm)

    # One instruction: y = (x * scale) + shift
    nisa.tensor_scalar(
        dst=y, data=x,
        op0=nl.multiply, operand0=scale,
        op1=nl.add, operand1=shift,
    )

    nisa.dma_copy(dst=output, src=y)
    return output


# =============================================================================
# Example 4: Reduction — sum along free dimension (axis=1)
# =============================================================================

@nki.jit
def reduce_sum_axis1(x_hbm):
    """Sum reduction along axis=1 (free dimension).

    tensor_reduce produces result in PSUM (always FP32).
    We then copy PSUM → SBUF to store.
    """
    P, F = x_hbm.shape
    output = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    x = nl.ndarray((P, F), dtype=x_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=x_hbm)

    # Reduce: result lands in PSUM (FP32)
    sum_psum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.tensor_reduce(dst=sum_psum, data=x, op=nl.add, axis=1)

    # PSUM → SBUF (required before storing to HBM)
    sum_sbuf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sum_sbuf, src=sum_psum)

    nisa.dma_copy(dst=output, src=sum_sbuf)
    return output


# =============================================================================
# Example 5: Activation functions (Scalar Engine)
# =============================================================================

@nki.jit
def activation_rsqrt(x_hbm):
    """Compute 1/sqrt(x) using nisa.activation.

    Available activations: nl.exp, nl.rsqrt, nl.sigmoid, nl.gelu, etc.
    These run on the Scalar Engine.
    """
    P, F = x_hbm.shape
    output = nl.ndarray((P, F), dtype=x_hbm.dtype, buffer=nl.shared_hbm)

    x = nl.ndarray((P, F), dtype=x_hbm.dtype, buffer=nl.sbuf)
    y = nl.ndarray((P, F), dtype=x_hbm.dtype, buffer=nl.sbuf)

    nisa.dma_copy(dst=x, src=x_hbm)

    # Scalar Engine: compute rsqrt element-wise
    nisa.activation(dst=y, data=x, op=nl.rsqrt)

    nisa.dma_copy(dst=output, src=y)
    return output


# =============================================================================
# Example 6: Combining ops — softmax numerator (exp(x - max))
# =============================================================================

@nki.jit
def exp_shifted(x_hbm):
    """Compute exp(x - max(x)) along axis=1. Numerically stable softmax numerator.

    Shows the pattern: reduce → broadcast scalar → subtract → activation.
    """
    P, F = x_hbm.shape
    output = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm)

    x = nl.ndarray((P, F), dtype=x_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=x_hbm)

    # Step 1: max reduction along free dim → PSUM
    max_psum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.tensor_reduce(dst=max_psum, data=x, op=nl.maximum, axis=1)

    # Step 2: copy PSUM → SBUF
    max_sbuf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=max_sbuf, src=max_psum)

    # Step 3: subtract max (broadcasts scalar across free dim)
    # tensor_scalar with operand0 as a tensor broadcasts P-dim scalar to all F positions
    shifted = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=shifted, data=x, op0=nl.subtract, operand0=max_sbuf)

    # Step 4: exp
    result = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=result, data=shifted, op=nl.exp)

    nisa.dma_copy(dst=output, src=result)
    return output


# =============================================================================
# Run and verify
# =============================================================================

def main():
    print("=" * 60)
    print("NKI Script 2: Vector & Scalar Engine Operations")
    print("=" * 60)

    P, F = 128, 512

    # Example 1
    print("\n[1] Element-wise add...")
    a = np.random.randn(P, F).astype(np.float16)
    b = np.random.randn(P, F).astype(np.float16)
    result = vector_add(a, b)
    expected = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16)
    assert np.allclose(result, expected, atol=1e-2), f"Max diff: {np.abs(result - expected).max()}"
    print("    ✓ nisa.tensor_tensor(op=nl.add)")

    # Example 2
    print("\n[2] Scalar multiply (x * 2.5)...")
    x = np.random.randn(P, F).astype(np.float16)
    result = scalar_multiply(x, 2.5)
    expected = (x.astype(np.float32) * 2.5).astype(np.float16)
    assert np.allclose(result, expected, atol=1e-2)
    print("    ✓ nisa.tensor_scalar(op0=nl.multiply)")

    # Example 3
    print("\n[3] Fused scale+shift (x * 0.5 + 1.0) in one instruction...")
    result = fused_scale_shift(x, 0.5, 1.0)
    expected = (x.astype(np.float32) * 0.5 + 1.0).astype(np.float16)
    assert np.allclose(result, expected, atol=1e-2)
    print("    ✓ nisa.tensor_scalar(op0=mul, op1=add) — single instruction!")

    # Example 4
    print("\n[4] Sum reduction along axis=1...")
    x = np.random.randn(P, F).astype(np.float16)
    result = reduce_sum_axis1(x)
    expected = x.astype(np.float32).sum(axis=1, keepdims=True)
    assert np.allclose(result, expected, atol=1.0), f"Max diff: {np.abs(result - expected).max()}"
    print("    ✓ nisa.tensor_reduce(op=nl.add, axis=1) → PSUM → SBUF")

    # Example 5
    print("\n[5] Activation: rsqrt...")
    x = np.abs(np.random.randn(P, F)).astype(np.float16) + 0.1  # positive inputs
    result = activation_rsqrt(x)
    expected = (1.0 / np.sqrt(x.astype(np.float32))).astype(np.float16)
    assert np.allclose(result, expected, atol=5e-2), f"Max diff: {np.abs(result - expected).max()}"
    print("    ✓ nisa.activation(op=nl.rsqrt)")

    # Example 6
    print("\n[6] exp(x - max(x)) — softmax building block...")
    x = np.random.randn(P, F).astype(np.float16)
    result = exp_shifted(x)
    x_f32 = x.astype(np.float32)
    expected = np.exp(x_f32 - x_f32.max(axis=1, keepdims=True))
    assert np.allclose(result, expected, atol=1e-2), f"Max diff: {np.abs(result - expected).max()}"
    print("    ✓ reduce → broadcast subtract → exp (3 engines cooperating)")

    print("\n" + "=" * 60)
    print("All Vector/Scalar Engine examples passed!")
    print("=" * 60)
    print("""
Key takeaways:
  • nisa.tensor_tensor → Vector Engine (binary element-wise)
  • nisa.tensor_scalar → Scalar Engine (op with constant, can fuse 2 ops)
  • nisa.tensor_reduce → reduces along axis, result in PSUM (FP32)
  • nisa.activation → transcendentals (exp, rsqrt, sigmoid...) on Scalar Engine
  • PSUM is FP32-only — must copy to SBUF before storing to HBM
  • Scalar broadcasting: a [P,1] tensor broadcasts across the F dimension
""")


if __name__ == "__main__":
    main()
