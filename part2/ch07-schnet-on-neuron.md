# Example 1: Getting SchNet running on Neuron

*You have a molecular GNN. You want it on Neuron. What breaks, what works, and what do you do about it?*

---

## Introducing SchNet

- SchNet: message-passing neural network for predicting molecular properties (energies, forces)
- Architecture: continuous-filter convolutions on atom positions, radial basis functions, interaction blocks with residual connections
- HCLS-relevant: drug discovery (binding affinity prediction), materials science, molecular dynamics
- Available via `schnetpack` (already cloned in this repo under `schnetpack/`)
- Key characteristic for this chapter: uses scatter/gather operations heavily (message aggregation in GNNs = sum messages from neighbors into each atom)

---

## Move to Neuron — it just works (mostly)

- Load a pretrained SchNet model from schnetpack
- `model.to("neuron")`, prepare inputs, run forward pass
- It runs! Produces numerically correct output
- No code changes required — PyTorch Native backend handles dispatch automatically
- But... first call is slow (op-by-op JIT compilation, each op → small NEFF)
- Subsequent calls hit NEFF cache, much faster
- Key insight from the Neuron team: "The model RUNS correctly — performance is the only issue"

---

## The fallback list — what landed on CPU?

- Some ops have no Neuron kernel implementation → dispatcher falls back to CPU automatically
- Mechanically: tensor copies from HBM → CPU memory, op executes on CPU, result copies back to HBM
- Each fallback = two PCIe round-trips (one in, one out) + CPU compute
- SchNet fallback ops (expected): `scatter_reduce`, `index_select`, possibly `scatter_add`
- These are fundamental to GNNs — message passing requires gathering/scattering along neighbor indices
- Show the fallback list: `torch_neuronx.get_fallback_ops()` or examine profiler trace for `copy_` patterns
- Compare with EquiformerV3 findings: 5 CPU fallbacks (`atan2`, `linalg_cross`, `acos`, `scatter_reduce`, `uniform_`) caused 87× slowdown

---

## Measuring the cost

- Profile the eager forward pass on Neuron
- Identify in the trace: compute blocks (on NeuronCore) vs `copy_` blocks (PCIe transfers)
- Quantify: what fraction of wall-clock time is actual compute vs data movement to/from CPU?
- For GNNs, the scatter ops are in the inner loop (every message-passing layer) — high frequency fallbacks
- Small tensor fallbacks (e.g., one-time shape computation) = negligible cost
- Large tensor fallbacks in the hot path (e.g., scatter_reduce on all atom features every layer) = devastating
- The profiler makes this immediately obvious: gaps between NeuronCore activity = CPU fallback round-trips

---

## The decision framework

- For each fallback op, ask three questions:
  1. **Is it in the hot path?** (every forward pass, every layer) or one-time (model init, preprocessing)
  2. **How much data moves?** (small scalar = cheap; full activation tensor = expensive)
  3. **Is there a Neuron-native alternative?** (reformulate as matmul, use a different algorithm, write NKI kernel)
- Decision matrix:
  - One-time + small data → accept on CPU (negligible cost)
  - Hot path + large data + reformulable → reformulate now
  - Hot path + large data + not reformulable → accept for now, write NKI kernel later (Part V)
- Real customer examples that motivated NKI:
  - Mobilai: Max Pooling not lowered properly by compiler → huge bottleneck → NKI kernel was 100× faster
  - Autodesk: trilinear interpolation not supported → compiler broke entirely → NKI kernel unlocked the model
  - "Before NKI, the only option for unsupported operators or performance issues was to file a ticket and wait. Now you can implement solutions yourself" (Emily, Neuron team)

---

## When eager is enough

- Research/debugging: eager gives you Python-level stack traces, `print()` works, `pdb` works
- Small models / prototyping: compilation overhead may exceed runtime savings
- Rule of thumb: if your forward pass is < 10ms in eager, `torch.compile` overhead may not pay off
- The two-persona workflow:
  - **Research persona:** eager mode, fast iteration, instant feedback
  - **Production persona:** compiled mode, max throughput, cached NEFFs
- Eager is your development environment; compile is your deployment environment

*Question raised → "Eager works but it's slow. Can the compiler help with these fallbacks?"*

---

## Compiling SchNet — what improves, what doesn't

- `torch.compile(model, backend="neuron")` — the compiler sees the full graph, fuses ops, generates one NEFF
- What improves: all the supported ops between fallbacks get fused (matmuls + activations + norms → fewer HBM round-trips)
- What does NOT improve: fallback ops still fall back. The compiler can't fuse what doesn't have a Neuron implementation
- The compiler optimizes everything *around* the unsupported ops, but can't eliminate them
- Profile compiled SchNet: the gaps (CPU fallbacks) are still there, but the NeuronCore bursts between them are denser/faster
- Net speedup depends on ratio of supported vs unsupported compute
- "Two compilers being developed: one for NKI kernels, one for everything else. The NKI compiler is not yet available to customers — currently all goes through the monolithic Neuron compiler" (Emily, Neuron team)

---

## Eager vs compiled — the two personas

- **Research persona:** eager mode, fast iteration, `print()` and `pdb` work, instant feedback
- **Production persona:** compiled mode, max throughput, cached NEFFs
- Rule of thumb: if your forward pass is < 10ms in eager, `torch.compile` overhead may not pay off
- The transition workflow: develop in eager → validate correctness → compile → profile → deploy
- When NOT to compile: dynamic control flow, heavy Python interop, active debugging
- Bucketing for variable-length inputs (protein sequences vary from 30 to 2000+ residues)

*Question raised → "Compile helped with the fused ops, but the fallback gaps remain. Where exactly is the time going?"*

*Next: [Chapter 8](../part3/ch09-profiler) — The profiler (Part III).*
