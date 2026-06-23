"""
Chapter 7: Getting SchNet running on Neuron
============================================
Run on trn2.3xlarge:
    source /workshop/workspace/native_venv/bin/activate
    cd /workshop/neuron-performance-book/schnetpack && pip install -e .
    cd /workshop/neuron-performance-book
    python scripts/ch7_schnet_example.py
"""

import torch
import time
import glob
import os
import schnetpack.properties as properties
from schnetpack.representation import SchNet
from schnetpack.nn.radial import GaussianRBF
from schnetpack.nn.cutoff import CosineCutoff

DEVICE = "neuron"
N_INTERACTIONS = 6
N_ATOM_BASIS = 128


def build_model():
    return SchNet(
        n_atom_basis=N_ATOM_BASIS,
        n_interactions=N_INTERACTIONS,
        radial_basis=GaussianRBF(n_rbf=20, cutoff=5.0),
        cutoff_fn=CosineCutoff(cutoff=5.0),
    ).eval()


def build_inputs(n_atoms, n_neighbors, device="cpu"):
    n_pairs = n_atoms * n_neighbors
    return {
        properties.Z: torch.randint(1, 10, (n_atoms,), device=device),
        properties.Rij: torch.randn(n_pairs, 3, device=device),
        properties.idx_i: torch.randint(0, n_atoms, (n_pairs,), device=device),
        properties.idx_j: torch.randint(0, n_atoms, (n_pairs,), device=device),
    }


def time_cpu(model, inputs, n_iter=100):
    with torch.no_grad():
        for _ in range(10):
            _ = model(inputs)
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iter):
            _ = model(inputs)
    return (time.time() - start) / n_iter


def time_neuron(model, inputs, n_iter=100):
    with torch.no_grad():
        for _ in range(10):
            _ = model(inputs)
    torch.neuron.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iter):
            _ = model(inputs)
    torch.neuron.synchronize()
    return (time.time() - start) / n_iter


# ============================================================
# Part 1: Basic correctness — does SchNet run on Neuron?
# ============================================================
print("=" * 70)
print("PART 1: CORRECTNESS CHECK")
print("=" * 70)

model = build_model()
inputs_cpu = build_inputs(50, 20, device="cpu")

with torch.no_grad():
    out_cpu = model(inputs_cpu)
print(f"CPU output shape: {out_cpu['scalar_representation'].shape}")

model_neuron = model.to(DEVICE)
inputs_neuron = {k: v.to(DEVICE) for k, v in inputs_cpu.items()}

with torch.no_grad():
    out_neuron = model_neuron(inputs_neuron)
torch.neuron.synchronize()

correct = torch.allclose(
    out_cpu["scalar_representation"],
    out_neuron["scalar_representation"].cpu(),
    atol=1e-3,
)
print(f"Neuron output matches CPU: {correct}")
print(f"Max diff: {(out_cpu['scalar_representation'] - out_neuron['scalar_representation'].cpu()).abs().max():.6f}")


# ============================================================
# Part 2: Profile eager mode — identify CPU fallbacks
# ============================================================
print("\n" + "=" * 70)
print("PART 2: CPU FALLBACK ANALYSIS (EAGER MODE)")
print("=" * 70)

# Warmup
with torch.no_grad():
    for _ in range(5):
        _ = model_neuron(inputs_neuron)
torch.neuron.synchronize()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True,
) as prof:
    with torch.no_grad():
        _ = model_neuron(inputs_neuron)
    torch.neuron.synchronize()

print("\nTop 15 ops by CPU time:")
print(f"  {'Op':<50} {'Calls':<6} {'Total ms':>10}")
print(f"  {'-'*50} {'-'*6} {'-'*10}")
for e in sorted(prof.key_averages(), key=lambda e: e.cpu_time_total, reverse=True)[:15]:
    print(f"  {e.key:<50} {e.count:<6} {e.cpu_time_total/1000:>10.2f}")

# Identify fallback ops specifically
fallback_ops = []
for e in prof.key_averages():
    if "copy" in e.key.lower() or e.key in ["aten::to", "aten::_to_copy"]:
        fallback_ops.append(e)

total_fallback_ms = sum(e.cpu_time_total for e in fallback_ops) / 1000
total_ms = sum(e.cpu_time_total for e in prof.key_averages()) / 1000
print(f"\nFallback overhead (to/copy ops): {total_fallback_ms:.2f} ms / {total_ms:.2f} ms total")
print(f"Fallback fraction: {total_fallback_ms/total_ms*100:.1f}%")


# ============================================================
# Part 3: torch.compile — what improves, what doesn't
# ============================================================
print("\n" + "=" * 70)
print("PART 3: torch.compile ANALYSIS")
print("=" * 70)

torch._dynamo.reset()
compiled_model = torch.compile(model_neuron, backend="neuron")

# First call (compilation)
start = time.time()
with torch.no_grad():
    _ = compiled_model(inputs_neuron)
torch.neuron.synchronize()
print(f"Compilation time: {time.time()-start:.1f}s")

# Check NEFFs
for path in ["/tmp/neff_cache", os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR", "")]:
    if path:
        neffs = glob.glob(f"{path}/**/*.neff", recursive=True)
        if neffs:
            print(f"NEFFs in {path}: {len(neffs)}")

# Dynamo explanation
print("\nDynamo graph analysis:")
torch._dynamo.reset()
compiled_model2 = torch.compile(model_neuron, backend="neuron")
explanation = torch._dynamo.explain(compiled_model2)(inputs_neuron)
print(explanation)


# ============================================================
# Part 4: Scaling — find the crossover point
# ============================================================
print("\n" + "=" * 70)
print("PART 4: SCALING ANALYSIS — WHERE DOES NEURON WIN?")
print("=" * 70)

configs = [
    (50, 20, "small molecule"),
    (200, 30, "medium protein fragment"),
    (500, 50, "small protein"),
    (1000, 50, "medium protein"),
    (2000, 50, "large protein"),
]

print(f"\n  {'Config':<25} {'Atoms':>6} {'Pairs':>8} {'CPU ms':>8} {'Neuron ms':>10} {'Speedup':>8}")
print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")

results = []
for n_atoms, n_neighbors, label in configs:
    n_pairs = n_atoms * n_neighbors

    # CPU
    model_cpu = build_model()
    inputs_c = build_inputs(n_atoms, n_neighbors, device="cpu")
    t_cpu = time_cpu(model_cpu, inputs_c, n_iter=50)

    # Neuron compiled
    torch._dynamo.reset()
    model_n = build_model().to(DEVICE).eval()
    inputs_n = build_inputs(n_atoms, n_neighbors, device=DEVICE)
    compiled_n = torch.compile(model_n, backend="neuron")

    # Warmup
    try:
        with torch.no_grad():
            for _ in range(5):
                _ = compiled_n(inputs_n)
        torch.neuron.synchronize()
        t_neuron = time_neuron(compiled_n, inputs_n, n_iter=50)
        speedup = t_cpu / t_neuron
    except Exception as e:
        t_neuron = float("inf")
        speedup = 0.0
        print(f"  {label:<25} {n_atoms:>6} {n_pairs:>8} {t_cpu*1000:>8.2f} {'FAILED':>10} {'N/A':>8}")
        print(f"    Error: {e}")
        continue

    print(f"  {label:<25} {n_atoms:>6} {n_pairs:>8} {t_cpu*1000:>8.2f} {t_neuron*1000:>10.2f} {speedup:>8.2f}x")
    results.append((label, n_atoms, n_pairs, t_cpu, t_neuron, speedup))

# Summary
print("\n" + "-" * 70)
crossover = next((r for r in results if r[5] > 1.0), None)
if crossover:
    print(f"Crossover point: ~{crossover[1]} atoms ({crossover[0]})")
    print(f"  At this scale, Neuron compute outweighs fallback overhead.")
else:
    print("No crossover found — CPU wins at all tested scales.")
    print("  The fallback cost (index + scatter_add) dominates.")


# ============================================================
# Part 5: The scatter-as-matmul reformulation
# ============================================================
print("\n" + "=" * 70)
print("PART 5: SCATTER-AS-MATMUL REFORMULATION")
print("=" * 70)

n_atoms = 50
n_neighbors = 20
n_pairs = n_atoms * n_neighbors

# Build adjacency matrix on CPU, move once
idx_i_cpu = torch.randint(0, n_atoms, (n_pairs,))
A = torch.zeros(n_atoms, n_pairs, dtype=torch.bfloat16)
A[idx_i_cpu, torch.arange(n_pairs)] = 1.0

# Test equivalence: scatter_add vs matmul
x_ij = torch.randn(n_pairs, N_ATOM_BASIS)
scatter_result = torch.zeros(n_atoms, N_ATOM_BASIS).index_add(0, idx_i_cpu, x_ij)
matmul_result = A.float() @ x_ij

print(f"scatter_add vs A @ x_ij — max diff: {(scatter_result - matmul_result).abs().max():.6f}")
print(f"Mathematically equivalent: {torch.allclose(scatter_result, matmul_result, atol=1e-4)}")

# Size analysis
print(f"\nAdjacency matrix A: {n_atoms}×{n_pairs} = {A.numel() * 2 / 1024:.1f} KB (BF16)")
print(f"Fits in SBUF (28 MB): easily")
print(f"\nFor 500 atoms × 25000 pairs: {500 * 25000 * 2 / 1024 / 1024:.1f} MB — still fits")
print(f"For 2000 atoms × 100000 pairs: {2000 * 100000 * 2 / 1024 / 1024:.1f} MB — too large, need sparse")

# Time comparison on Neuron
A_neuron = A.to(DEVICE)
x_ij_neuron = torch.randn(n_pairs, N_ATOM_BASIS, device=DEVICE, dtype=torch.bfloat16)
idx_i_neuron = idx_i_cpu.to(DEVICE)

# scatter_add path (has fallback)
torch.neuron.synchronize()
start = time.time()
for _ in range(100):
    tmp = torch.zeros(n_atoms, N_ATOM_BASIS, device=DEVICE, dtype=torch.bfloat16)
    _ = tmp.index_add(0, idx_i_neuron, x_ij_neuron)
torch.neuron.synchronize()
t_scatter = (time.time() - start) / 100

# matmul path (native on Neuron)
torch.neuron.synchronize()
start = time.time()
for _ in range(100):
    _ = A_neuron @ x_ij_neuron
torch.neuron.synchronize()
t_matmul = (time.time() - start) / 100

print(f"\nscatter_add (with fallback): {t_scatter*1000:.3f} ms")
print(f"matmul (native on Neuron):   {t_matmul*1000:.3f} ms")
print(f"Speedup from reformulation:  {t_scatter/t_matmul:.1f}x")
