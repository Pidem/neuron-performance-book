# Next-gen chips and distributed training

*You've mastered a single NeuronCore. The silicon keeps getting faster, and the network turns many chips into one.*

Everything in this book has lived on a single chip. One instance, one Trainium2 device, a handful of NeuronCores working together under the runtime's control. Two things are changing at once: each generation of NeuronCore gets architecturally smarter (not wider), and the interconnect between chips improves fast enough that multi-chip parallelism keeps up with the compute.

This chapter covers where the performance ladder goes from here. The principles you learned (tiling, engine overlap, memory hierarchy) stay the same even as the numbers change.

---

## The chip keeps getting smarter

In Part V, you wrote NKI kernels targeting NeuronCore-v3 (Trainium2). NeuronCore-v4 (Trainium3) adds architectural features that eliminate bottlenecks you had to work around by hand.

```{figure} ../assets/trn3_improvements.png
:align: center
:width: 90%

NeuronCore-v4 improvements over v3. Each upgrade targets a specific bottleneck: the Tensor Engine gains 4× throughput via microscaled formats, the Vector Engine absorbs softmax with a dedicated fast-exponential unit, the Scalar Engine doubles BF16 throughput, and DMA gets near-memory accumulation plus indirect addressing. The net effect is fewer round-trips between engines and fewer cycles spent on data reformatting.
```

### What changes for NKI programmers

| Feature | What it does | Why it matters |
|---------|-------------|----------------|
| **Fast exponential** (`nisa.exponential`) | Computes `exp(x - max) + accumulate` on VectorE at 4× the throughput of ScalarE `activation` | Softmax, the critical path in long-context attention, runs 4× faster without leaving VectorE |
| **Background transpose** | TensorE runs a transpose in parallel with the next matmul | Chains of transpose+matmul (common in attention) overlap instead of serializing |
| **SBUF indirect access** | Gather/scatter along the free dimension in a single instruction | Sparse access patterns (MoE routing, token selection) no longer need multiple instructions |
| **SBUF Read-Add-Write** | DMA accumulates into SBUF: `B += A` near-memory | Gradient accumulation and residual additions skip the compute engines |
| **Quad MXFP8/MXFP4** | 4× matmul throughput via microscaled formats (OCP standard) | 315 TFLOPS per core in MXFP8, up from 158 FP8 on v3 |
| **VectorE MX quantization** | `nisa.quantize_mx` produces TensorE-ready MXFP8 layout in one instruction | No manual scale computation or data packing |
| **Scalar performance mode** | ScalarE runs `tensor_scalar` and `tensor_copy` at 2× BF16 throughput | Offload from VectorE or balance load between engines |
| **DMA traffic shaping** | 4 priority classes for DMA bandwidth allocation | Fine-grained control when compute and collectives compete for the bus |

Each feature moves work closer to where the data already lives. Fast exponential keeps softmax on VectorE instead of bouncing to ScalarE. Read-Add-Write keeps gradient accumulation in the DMA path instead of occupying a compute engine. Indirect access eliminates gather loops. Each generation removes a scheduling constraint that the previous generation forced on you.

### Raw numbers: NeuronCore-v4 vs v3

| | NeuronCore-v3 (Trn2) | NeuronCore-v4 (Trn3) |
|---|---|---|
| SBUF | 28 MiB | 32 MiB |
| Tensor Engine (BF16) | 79 TFLOPS | 79 TFLOPS |
| Tensor Engine (MXFP8) | 158 TFLOPS | 315 TFLOPS |
| HBM per chip | 96 GiB | 144 GiB |
| HBM bandwidth per chip | 2.9 TB/s | 4.7 TB/s |
| CC-Cores per chip | 16 | 20 |

BF16 compute stays flat. The throughput gain comes from narrower formats and smarter data movement. The chip isn't getting wider; it's getting better at feeding itself.

### Trainium 4 (announced)

Targets (likely to exceed):
- 6× FP4 performance uplift
- 4× memory bandwidth
- 2× memory capacity

---

## From one chip to many

You learned in Chapter 4 that a single NeuronCore has dedicated engines (Tensor, Vector, Scalar, DMA) all running in parallel. The same separation applies at the system level: dedicated Collective Compute Cores (CC-Cores) handle inter-chip communication while compute engines keep working on the next layer.

### CC-Cores: communication that doesn't steal from compute

CC-Cores (16 per Trn2 chip, 20 per Trn3) sit physically separate from the tensor/vector/scalar engines. They fire all-reduce, all-gather, and reduce-scatter while the Tensor Engine computes the next layer:

```
Tensor Engine:  [compute layer N  ][compute layer N+1][compute layer N+2]
CC-Cores:       [all-reduce grad N-1][all-reduce grad N][all-reduce grad N+1]
```

When compute time ≥ communication time, collectives add zero wall-clock latency. You don't write overlap code. The hardware handles it.

```{admonition} SRAM-to-SRAM collectives
:class: note
The default collective path is: SBUF → DMA → HBM → collective → HBM → DMA → SBUF. Three memory hops each way. For training this is fine — compute dominates. But in latency-sensitive decode (streaming tokens one at a time), those extra hops dominate.

Neuron hardware supports **direct SRAM-to-SRAM collectives** that skip HBM entirely. In the profiler, you can see the difference: without direct collectives, the GPSIMD engine spends time writing DMA descriptors for intermediate memory movements between SBUF and HBM. With direct collectives, those descriptors disappear — the collective fires straight from on-chip memory to the neighboring chip's on-chip memory, and per-token latency drops dramatically.

Anthropic uses this in their LLM decode kernels to minimize per-token latency on sharded models.
```

### The communication roofline

As models scale, weight matrices grow too large for a single chip's memory. The solution is tensor parallelism (TP): split each weight matrix across P chips so each chip computes a slice of the matmul in parallel, then aggregate the results. The catch: if NeuronLink bandwidth can't deliver the partial results fast enough, chips sit idle waiting for each other.

Consider a matrix multiplication Y = X × W in a transformer layer with hidden dimension D=4096, batch B=8, sharded across P=3 chips. Each chip holds the full input X [B, D] (already present from the previous layer's all-gather) and one permanent weight shard Wᵢ [D, D/3]. No master chip coordinates this: every chip runs the same program (SPMD), knows its rank, and indexes into its own shard.

```{figure} ../assets/tp_matmul_distributed.svg
:align: center
:width: 90%

Tensor parallelism for one matmul. Each chip computes its local slice, then partial outputs Y₀, Y₁, Y₂ flow over NeuronLink so every chip can assemble the full Y for the next layer.
```

Only the partial outputs flow over NeuronLink. Weights stay resident on each chip. Input X is already present from the previous layer. The all-gather sends each chip's Yᵢ [B, D/3] to the other P−1 chips. In our example: each Yᵢ is 8 × 1365 × 2 bytes ≈ 22 KB, so each chip receives about 44 KB per layer.

TP stays efficient when compute takes longer than communication. Otherwise chips finish their slice and wait:

$$\text{Compute: } \frac{2 \cdot B \cdot D \cdot (D/P)}{\text{FLOPs/s}} \quad > \quad \text{Comm: } \frac{(P-1)/P \cdot B \cdot D \cdot \text{bytes per elem}}{\text{NeuronLink BW}}$$

Cancel B·D from both sides and you get the minimum D for TP to be compute-bound:

$$D > (P-1) \times \frac{\text{bytes per element} \times \text{FLOPs/s per chip}}{\text{NeuronLink bandwidth}}$$

On Trn2 (632 BF16 TFLOPS per chip, 1.28 TB/s NeuronLink, 2 bytes per BF16 element):

| TP degree (P) | D must exceed | Llama-7B (D=4096) | Llama-70B (D=8192) |
|---------------|---------------|--------------------|--------------------|
| 2 | 988 | ✓ compute-bound | ✓ compute-bound |
| 4 | 2,962 | ✓ compute-bound | ✓ compute-bound |
| 8 | 6,912 | ✗ comm-bound | ✓ compute-bound |

At TP degree 2–4, most models have enough hidden dimension to stay efficient. Push to 8 chips and smaller models hit the communication wall. The fix: keep TP small, shard weights across more chips with FSDP instead. NeuronSwitch on Trn3 doubles interconnect bandwidth, doubling these thresholds.

### Standard PyTorch, unchanged

```python
import torch.distributed as dist

dist.init_process_group(backend="nccl")  # Works on Neuron via CC-Cores
model = MyModel().to("neuron")
model = torch.nn.parallel.DistributedDataParallel(model)
```

No custom parallelism API. DP, FSDP, TP, PP, EP all work through standard PyTorch distributed because Neuron implements the `ProcessGroup` backend. Your distributed code doesn't change when you move from GPU to Neuron.

---

## The network: EFA and NeuronSwitch

Within a rack, chips talk via NeuronLink (1.28 TB/s per chip on Trn2). On Trn3, NeuronSwitch replaces the 2D torus with a full-mesh fabric where every chip connects to every other chip in one hop. This matters for Mixture-of-Expert models, where tokens route to arbitrary experts at runtime and all-to-all patterns can't be localized.

Between racks, the Elastic Fabric Adapter (EFA) provides 28.8 Tbps of aggregate scale-out bandwidth per UltraServer. Purpose-built networking, co-designed with the chip. Every packet distributes across all available paths to eliminate congestion. This is how Trainium scales from one rack to millions of chips in a non-blocking, petabit-scale fabric.

| Scale | Interconnect | Topology |
|-------|-------------|----------|
| Within chip (8 cores) | On-die | Shared SBUF/HBM |
| Within rack (144 chips, Trn3) | NeuronSwitch | Full-mesh, single-hop |
| Across racks (UltraClusters) | EFA | Multi-path, non-blocking |

---

## Scale in practice

Project Rainier put over 500,000 Trainium 2 chips in production for Anthropic, making it one of the world's largest AI compute clusters. Across all customers, over a million Trainium chips run production workloads today. TorchTitan and HuggingFace Accelerate work out of the box on these clusters and reach above 50% MFU on dense models.

Trainium 3 produces 5× more tokens per second per megawatt than Trainium 2 on GPT-OSS 120B inference. That's 1.5 years between chip generations. When you're paying for thousands of chips, that pace compounds.

---

## The ladder continues

This book took you from `model(x)` to custom NKI kernels on a single core. The next rungs (multi-chip parallelism, UltraServer-scale training, serving millions of tokens per second) use the same principles you already know: minimize data movement, keep engines busy, overlap what you can.

---

*How do you package an optimized model for real users? Next: [Chapter 18](ch18-production).*
