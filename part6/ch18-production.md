# From kernel to community

*You learned the general skill. The ecosystem is catching up.*

This book teaches one thing: take a PyTorch model, compile it to Neuron hardware, profile it, and push its performance with NKI. That workflow works for any architecture — standard or custom, published or proprietary. You're not waiting for anyone to add support for your model.

But you're not working alone. HuggingFace, vLLM, and PyTorch's own TorchTitan each independently invested in Neuron support. They integrated torch_neuronx for their most popular architectures, wrote fused NKI kernels for attention and MLP layers, and validated performance on Trainium. If your model happens to be Llama or Mistral or Qwen, someone already did the optimization work for you.

This chapter maps that ecosystem — not as a tutorial, but so you understand how your skills connect to production tooling.

---

## The stack

| Layer | What it does | Who built it |
|-------|-------------|--------------|
| **torch_neuronx** | Compiles any PyTorch graph to Neuron hardware | AWS Neuron team |
| **optimum-neuron** | Fine-tuning API for HuggingFace models on Neuron | HuggingFace |
| **vLLM Neuron plugin** | High-throughput LLM serving on Neuron | vLLM community + AWS |
| **TorchTitan** | Distributed training (FSDP, TP, PP) | PyTorch team |

Each layer builds on the one above it. optimum-neuron calls torch_neuronx internally. vLLM's Neuron backend uses the same compilation path you learned in Chapter 6. The NKI attention kernel inside vLLM uses the same APIs from Chapters 12–15.

The key insight: these tools cover *common* architectures. When your model doesn't fit — a novel attention pattern, a custom activation, an architecture no one has ported yet — you fall back to the general skill this book teaches.

---

## Fine-tuning: optimum-neuron

[optimum-neuron](https://huggingface.co/docs/optimum-neuron) wraps the HuggingFace Transformers API. You write standard LoRA fine-tuning code, swap `Trainer` for `NeuronTrainer`, and the library handles graph compilation and BF16 mixed precision underneath.

The compiled graphs use the same operator fusion and tiling you saw in Chapters 3–5. When you profile a fine-tuning run with Neuron Explorer, you see the same NEFF execution and DMA patterns from Chapter 10. The abstraction is thin — one layer above the torch_neuronx path you already know.

---

## Serving: vLLM on Neuron

[vLLM](https://docs.vllm.ai/) handles LLM inference with continuous batching, paged KV cache, and speculative decoding. The Neuron plugin provides:

- **Continuous batching** — new requests join mid-generation without waiting
- **Paged KV cache** — virtual-memory management avoids HBM fragmentation
- **Speculative decoding** — draft model proposes tokens, main model verifies in one forward pass
- **Fused NKI kernels** — attention and MLP operators written in NKI

That last point matters. vLLM's Neuron attention kernel is NKI code — the same language from Parts IV–V. When you read the [Neuron Kernel Library source](https://github.com/aws-neuron/neuron-kernel-library), you'll see `nki.language` APIs, tile arithmetic, and SBUF/PSUM patterns from this book.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    tensor_parallel_size=2,  # shard across 2 chips (Ch17's column-parallel split)
)
outputs = llm.generate(["Summarize:"], SamplingParams(max_tokens=256))
```

vLLM supports specific architectures — Llama, Mistral, Qwen, DeepSeek, and others. If your model isn't on that list, vLLM won't serve it out of the box. That's where contribution comes in.

---

## Contributing back

The ecosystem tools are open source. If you optimize a model for Neuron, you can upstream it:

**Adding a model to vLLM.** vLLM maintains its own model implementations (separate from HuggingFace's `modeling_*.py`). To add Neuron support for a new architecture, you write a model class that uses vLLM's abstractions (paged attention, tensor-parallel linear layers) and register it in vLLM's architecture registry. Your NKI profiling skills tell you where fused kernels are needed; your kernel-writing skills let you implement them.

**Contributing kernels to Neuron Kernel Library.** NKL is the collection of pre-optimized NKI kernels that vLLM and optimum-neuron use internally. If you write a faster attention variant, a fused activation, or a custom normalization kernel, it can land in NKL and benefit every serving deployment that uses it.

**Publishing optimized checkpoints.** After fine-tuning with optimum-neuron, you can push compiled model artifacts to HuggingFace Hub. Other Neuron users skip the compilation step entirely — they download your pre-compiled checkpoint and serve it directly.

The path from "I optimized this for my workload" to "everyone on Neuron benefits" is a pull request.

---

## Where your skills plug in

The production stack handles plumbing. Your job as a performance engineer is to fix the bottlenecks generic tools can't:

**Profiling a serving workload.** vLLM gives you throughput numbers. Neuron Explorer tells you *why* throughput plateaus. If prefill latency dominates, you'll see it in the attention operator's compute utilization. If decode is memory-bound, you'll see low arithmetic intensity in the MLP layers — the roofline from Chapter 9.

**Custom kernels in production.** vLLM's NKI attention kernel handles standard multi-head attention. Models with non-standard patterns (sliding window + global tokens, cross-document boundaries, mixture-of-experts routing) need custom implementations. You write them with the NKI workflow from Chapters 12–15, then register them as custom ops in vLLM's model class.

**Number format choices.** Chapter 11 showed how BF16, FP8, and MX formats trade accuracy for throughput. In production, you pick the format per-layer: FP8 for dense MLP projections where quantization error is tolerable, BF16 for attention logits where precision matters. The profiler confirms whether your choice saturates compute or stays memory-bound.

---

## The lifecycle

At each stage, a different part of this book applies:

```
Train or fine-tune (optimum-neuron, TorchTitan)
       │
       │  Chapters 3–5: compilation, graph breaks, operator fusion
       ▼
Profile (Neuron Explorer)
       │
       │  Chapters 9–10: roofline, profiler interpretation
       ▼
Optimize hotspots (torch.compile → NKI if needed)
       │
       │  Chapters 12–15: custom kernels, tiling, number formats
       ▼
Deploy (vLLM, tensor parallelism across chips)
       │
       │  Chapter 17: TP, all-gather, communication roofline
       ▼
Monitor, optimize, contribute upstream
```

Most models never need custom NKI kernels. The compiled defaults handle standard architectures. But when you hit a wall — or when you want to bring a new architecture to Neuron — you have the skills to do it yourself, and an open-source ecosystem ready to accept the result.

---

*You started this book treating `model(x)` as a black box. Now you can trace execution from Python through the compiler to individual tensor engine instructions. The tools transfer from GPUs, the ecosystem is open, and the path from personal optimization to community contribution is a pull request.*
