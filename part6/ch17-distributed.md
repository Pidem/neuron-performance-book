# Multi-core, multi-chip

*Your model doesn't fit on one chip. Now what?*

---

## The scale-up domain

A single Trainium 2 chip has 8 NeuronCores and 96 GiB HBM. A `trn2.48xlarge` instance gives you 16 chips (128 NeuronCores, 1.5 TiB HBM) connected by NeuronLink at 1.28 TB/s per chip.

But frontier models don't fit on one instance either. That's where the **ultra server** — a full rack of tightly-integrated chips — comes in.

### Trainium 3 Ultra Server

| Component | Spec | vs Trn2 Ultra Server |
|-----------|------|---------------------|
| Chips per scale-up domain | 144 | — |
| Compute (microscaled FP8) | 360 PFLOPS | 4.4× |
| HBM capacity | 20 TB | 3.4× |
| HBM bandwidth | 700 TB/s | 3.9× |
| Interconnect | NeuronSwitch (full-mesh) | 2× bandwidth |

The key architectural change: **NeuronSwitch** replaces Trn2's 2D torus. Every compute sled connects to every other sled within a single hop via dedicated switch sleds in the middle of the rack.

```{admonition} Why full-mesh matters for MoE
:class: note
In Mixture-of-Expert models, tokens are routed to arbitrary experts at runtime. If experts live on different chips, you need all-to-all communication — every chip potentially talking to every other chip simultaneously. A 2D torus forces multi-hop routing for distant pairs. Full-mesh NeuronSwitch gives single-hop latency for *any* pair, making MoE expert parallelism dramatically more efficient.
```

### Trainium 4 (in development)

Targets announced (likely to exceed):
- 6× FP4 performance uplift
- 4× memory bandwidth
- 2× memory capacity

---

## Parallelism strategies

When a model is too large for one chip, you split it. The standard strategies all work unchanged on Neuron via PyTorch's native distributed APIs:

| Strategy | What's split | When to use | PyTorch API |
|----------|-------------|-------------|-------------|
| **Data parallelism (DP)** | Batch across replicas | Model fits on one chip, want more throughput | `DistributedDataParallel` |
| **FSDP** | Weights sharded, gathered per-op | Model *barely* fits — shard weights to save memory | `FullyShardedDataParallel` |
| **Tensor parallelism (TP)** | Weight matrices split across chips | Single matmul too large for one chip's SBUF | `DTensor` |
| **Pipeline parallelism (PP)** | Layers across chips | Very deep models, hide communication in bubbles | `torch.distributed.pipelining` |
| **Expert parallelism (EP)** | Experts across chips | MoE models | Custom routing |

```python
# This is real — same code as GPU
import torch.distributed as dist

dist.init_process_group(backend="nccl")  # Works on Neuron via collective cores
model = MyModel().to("neuron")
model = torch.nn.parallel.DistributedDataParallel(model)
```

There's no `NxD` modeling layer, no custom Neuron parallelism API. Standard PyTorch distributed works because Neuron implements the `ProcessGroup` backend.

---

## Communication overlap: why collectives are "free"

Neuron's 16 CC-Cores (collective compute cores) per chip are physically independent from the tensor/vector/scalar engines. They can fire all-reduce, all-gather, and reduce-scatter operations **while the compute engines work on the next layer**.

```
Tensor engine:  [compute layer N  ][compute layer N+1][compute layer N+2]
CC-Cores:       [all-reduce grad N-1][all-reduce grad N][all-reduce grad N+1]
```

As long as compute time ≥ communication time, collectives add zero latency. This is why Neuron scales well without explicit overlap tuning — the hardware handles it architecturally.

For latency-sensitive inference (token decode), Neuron supports **direct SRAM-to-SRAM collectives** that skip HBM entirely — the result goes straight from one chip's SBUF to another's without the SBUF→HBM→collective→HBM→SBUF round-trip. Anthropic uses this for minimum-latency decode.

---

## The communication roofline

From Chapter 9, recall that sharded matmuls introduce a communication bound:

$$D > \frac{2 \times \text{FLOPs/s per chip}}{\text{NeuronLink bandwidth}}$$

For Trn2: threshold ≈ 130,000 (model dimension). Since D is typically 4096–16384, tensor parallelism across many chips becomes communication-bound quickly.

The mitigation:
- Keep TP degree small (2–4 chips) — use FSDP for the rest
- NeuronSwitch's higher bandwidth on Trn3 raises the threshold
- Overlap communication with the next operation's compute

---

## Scale in practice

- **Project Rainier:** 500,000+ Trainium 2 chips deployed for Anthropic — one of the world's largest AI compute clusters
- Over **1 million Trainium chips** running customer workloads in production today
- Standard training libraries (TorchTitan, HuggingFace Accelerate) work out of the box on Neuron with >50% MFU on dense models

```{admonition} The 5× gen-over-gen efficiency
:class: note
Trainium 3 delivers **5× more tokens per second per megawatt** compared to Trainium 2 on real inference workloads (benchmarked on GPT-OSS 120B). This is 1.5 years between generations — a pace unlike typical CPU improvements (~20% per generation).
```

---

*You can distribute your model. But how do you package it for real users — fine-tuning, serving, and continuous deployment?*

*Next: [Chapter 18](ch18-production) — Putting it in production.*
