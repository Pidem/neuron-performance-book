"""
SchNet with matmul-reformulated message passing
================================================
Replaces scatter_add and fancy indexing with dense matmul operations
that run natively on the Neuron tensor engine.

Run on trn2.3xlarge:
    source /workshop/workspace/native_venv/bin/activate
    python scripts/ch7_matmul_schnet.py
"""

import torch
import time
import types
import schnetpack.properties as properties
from schnetpack.representation import SchNet
from schnetpack.representation.schnet import SchNetInteraction
from schnetpack.nn.radial import GaussianRBF
from schnetpack.nn.cutoff import CosineCutoff

DEVICE = "neuron"


def matmul_forward(self, x, f_ij, idx_i, idx_j, rcut_ij):
    """Drop-in replacement for SchNetInteraction.forward using matmul."""
    x = self.in2f(x)
    Wij = self.filter_network(f_ij) * rcut_ij[:, None]
    x_j = self.G @ x        # gather via matmul
    x_ij = x_j * Wij
    x = self.A @ x_ij       # scatter via matmul
    x = self.f2out(x)
    return x


def patch_schnet_with_matmul(model, idx_i, idx_j, n_atoms, n_pairs, device):
    """Monkey-patch a SchNet model to use matmul for gather/scatter."""
    dtype = torch.float32

    G = torch.zeros(n_pairs, n_atoms, dtype=dtype, device=device)
    G[torch.arange(n_pairs, device=device), idx_j] = 1.0

    A = torch.zeros(n_atoms, n_pairs, dtype=dtype, device=device)
    A[idx_i, torch.arange(n_pairs, device=device)] = 1.0

    for interaction in model.interactions:
        interaction.register_buffer("G", G)
        interaction.register_buffer("A", A)
        interaction.forward = types.MethodType(matmul_forward, interaction)

    return model


def build_inputs(n_atoms, n_neighbors, device):
    n_pairs = n_atoms * n_neighbors
    return {
        properties.Z: torch.randint(1, 10, (n_atoms,), device=device),
        properties.Rij: torch.randn(n_pairs, 3, device=device),
        properties.idx_i: torch.randint(0, n_atoms, (n_pairs,), device=device),
        properties.idx_j: torch.randint(0, n_atoms, (n_pairs,), device=device),
    }, n_pairs


def bench(fn, n_iter=200, device="neuron"):
    for _ in range(20):
        fn()
    if device == "neuron":
        torch.neuron.synchronize()
    start = time.time()
    for _ in range(n_iter):
        fn()
    if device == "neuron":
        torch.neuron.synchronize()
    return (time.time() - start) / n_iter * 1000


# ============================================================
print("=" * 70)
print("MATMUL-REFORMULATED SCHNET vs ORIGINAL")
print("=" * 70)

# Correctness check on CPU first
print("\n--- Correctness verification (CPU) ---")
model = SchNet(
    n_atom_basis=128, n_interactions=6,
    radial_basis=GaussianRBF(n_rbf=20, cutoff=5.0),
    cutoff_fn=CosineCutoff(cutoff=5.0),
).eval()

inputs, n_p = build_inputs(100, 20, "cpu")
with torch.no_grad():
    ref = model(inputs)["scalar_representation"].clone()

# Patch in-place and re-run
patch_schnet_with_matmul(
    model, inputs[properties.idx_i], inputs[properties.idx_j],
    100, n_p, "cpu"
)
with torch.no_grad():
    patched = model(inputs)["scalar_representation"]

max_diff = (ref - patched).abs().max().item()
print(f"  Max diff: {max_diff:.6f} (should be ~0)")
assert max_diff < 1e-5, f"Correctness failed: {max_diff}"
print("  ✓ Matmul reformulation is numerically exact")

# ============================================================
print(f"\n--- Benchmark ---")
configs = [
    (50, 20, "small molecule (aspirin-sized)"),
    (200, 30, "medium fragment"),
    (500, 40, "small protein"),
]

print(f"\n  {'Config':<35} {'CPU':>8} {'Orig+Compile':>12} {'Matmul+Compile':>14} {'Speedup':>8}")
print(f"  {'-'*35} {'-'*8} {'-'*12} {'-'*14} {'-'*8}")

for n_atoms, n_neighbors, label in configs:
    # --- CPU baseline ---
    model_cpu = SchNet(
        n_atom_basis=128, n_interactions=6,
        radial_basis=GaussianRBF(n_rbf=20, cutoff=5.0),
        cutoff_fn=CosineCutoff(cutoff=5.0),
    ).eval()
    inputs_cpu, _ = build_inputs(n_atoms, n_neighbors, "cpu")
    with torch.no_grad():
        t_cpu = bench(lambda: model_cpu(inputs_cpu), n_iter=100, device="cpu")

    # --- Original compiled on Neuron ---
    torch._dynamo.reset()
    model_orig = SchNet(
        n_atom_basis=128, n_interactions=6,
        radial_basis=GaussianRBF(n_rbf=20, cutoff=5.0),
        cutoff_fn=CosineCutoff(cutoff=5.0),
    ).eval().to(DEVICE)
    inputs_neuron, n_p = build_inputs(n_atoms, n_neighbors, DEVICE)

    compiled_orig = torch.compile(model_orig, backend="neuron")
    with torch.no_grad():
        for _ in range(3):
            compiled_orig(inputs_neuron)
        torch.neuron.synchronize()
    t_orig = bench(lambda: compiled_orig(inputs_neuron), n_iter=100)

    # --- Matmul version compiled on Neuron ---
    torch._dynamo.reset()
    model_matmul = SchNet(
        n_atom_basis=128, n_interactions=6,
        radial_basis=GaussianRBF(n_rbf=20, cutoff=5.0),
        cutoff_fn=CosineCutoff(cutoff=5.0),
    ).eval().to(DEVICE)
    patch_schnet_with_matmul(
        model_matmul,
        inputs_neuron[properties.idx_i], inputs_neuron[properties.idx_j],
        n_atoms, n_p, DEVICE,
    )

    compiled_matmul = torch.compile(model_matmul, backend="neuron")
    with torch.no_grad():
        for _ in range(3):
            compiled_matmul(inputs_neuron)
        torch.neuron.synchronize()
    t_matmul = bench(lambda: compiled_matmul(inputs_neuron), n_iter=100)

    speedup_vs_cpu = t_cpu / t_matmul
    speedup_vs_orig = t_orig / t_matmul

    print(f"  {label:<35} {t_cpu:>7.2f}ms {t_orig:>11.2f}ms {t_matmul:>13.2f}ms {speedup_vs_cpu:>6.1f}x cpu")
    print(f"  {'':35} {'':>8} {'':>12} {'':>14} {speedup_vs_orig:>6.1f}x orig")

# ============================================================
print(f"\n--- Memory cost ---")
for n_atoms, n_neighbors, label in configs:
    n_pairs = n_atoms * n_neighbors
    mem_mb = 2 * (n_pairs * n_atoms) * 4 / 1024 / 1024
    print(f"  {label:<35} G+A = {mem_mb:.1f} MB")
