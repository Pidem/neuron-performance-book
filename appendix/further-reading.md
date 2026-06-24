# Further reading

*Curated resources for going deeper on Neuron, NKI, and performance engineering.*

---

The best place to start is obviously the AWS Neuron documentation available [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/index.html)

This book is only a guided intro to familiarize yourself with the concepts. 

## Neuron SDK documentation

- [Neuron SDK main page](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/)
- [NKI Programming Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)
- [NKI Tutorials (including 98% HFU matmul)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/tutorials/index.html)
- [NKI Performance Optimizations Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/nki_perf_guide.html)
- [Neuron Explorer (profiler)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/index.html)
- [Neuron Data Types](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/data-types.html)
- [Trn2 Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trn2-arch.html)
- [NeuronCore-v3 Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuroncores/neuroncore-v3.html)
- [PyTorch Native on Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/programming-guide/native-pytorch/index.html)

---

## Code samples

The AWS Neuron team and the community of architects continuously share some code samples that are a great way to familiarize yourself with the Neuron software stack: In general, you can look at the repositories hosted on [GitHub](https://github.com/aws-neuron), here are the main repositories worht checking out: 

- [NKI Library](https://github.com/aws-neuron/nki-library) — production-ready kernels (flash attention, RMSNorm-Quant, MoE, fused Adam, etc.)
- [NKI Samples](https://github.com/aws-neuron/nki-samples) — tutorial-style kernels with explanations
- [torch-neuronx](https://github.com/aws-neuron/torch-neuronx) — PyTorch Native backend source (open source)

---

## re:Invent talks (YouTube videos)

- **[AIM335: Trn3 UltraServers and Anthropic Kernel Optimization](https://www.youtube.com/watch?v=c_1FhdXNUSE)** (Ron Diamont + Jay Gray) — NeuronCore architecture deep dive, NKI in production at Anthropic, the RMSNorm-Quant kernel design
- **[AIM201: Breakthrough AI Performance with Trainium*](https://www.youtube.com/watch?v=5SgZ3zRJFAY&t=1s)** (Colin Brace + Poolside + DART) — 80% MFU achieved by keeping activations in SRAM, DART's SRAM-to-SRAM collectives
- **[AIM414 Performance engineering on Neuron: How to optimize your LLM with NKI](https://www.youtube.com/watch?v=e09mW8e8G-s)** (Scott Perry, Sadaf Rasool) 
- **[AIM351 End-to_end Foundation Model Lifecyle](https://www.youtube.com/watch?v=M5ClIj4wamg&t=1s) (Matt McClean)
---

## 3rd party books and and blogposts: 

- [How To Scale Your Model](https://jax-ml.github.io/scaling-book/) (Austin et al., Google DeepMind, 2025) — rooflines, sharding, training at scale. The closest analog to this book for TPUs/GPUs.
  - [Part 1: Rooflines](https://jax-ml.github.io/scaling-book/roofline/)
  - [Part 2: TPUs](https://jax-ml.github.io/scaling-book/tpus/)
  - [Part 3: Sharding](https://jax-ml.github.io/scaling-book/sharding/)
  - [Part 12: GPUs](https://jax-ml.github.io/scaling-book/gpus/)
- [KV Caching in LLM Inference](https://medium.com/@prathamgrover777/kv-caching-attention-optimization-from-o-n%C2%B2-to-o-n-8b605f0d4072) — why decode is memory-bound

**PyTorch related:**
- [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/) (Edward Z. Yang) — tensor storage, strides, the dispatcher
- [Tensor Puzzles](https://github.com/srush/Tensor-Puzzles) (Sasha Rush) — exercises for building intuition about tensor operations and indexing
- [Anatomy of a PT2 Compilation](https://aditvenk.substack.com/p/anatomy-of-a-pt2-compilation) (Aditya Venkataraman) — how torch.compile works under the hood (Dynamo, guards, AOTAutograd, Inductor)
**Precision related:**
- [What Every User Should Know About Mixed Precision Training](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/) (PyTorch Foundation) — AMP, GradScaler, BF16 vs FP16 tradeoffs
- [4-bit LLM Training and Primer on Precision](https://vizuara.substack.com/p/4-bit-llm-training-and-primer-on) (Siddhant Rai) — FP4 training, DGEs, quantization fundamentals
- [Aleksa Gordić: Matrix Multiplication](https://www.aleksagordic.com/blog/matmul) — matmul from first principles to hardware

---

## Academic references

- [MIT 6.5940: EfficientML](https://www.youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB) — Song Han's course on efficient deep learning (quantization, pruning, distillation, hardware-aware design)
- [Stanford CS149: Parallel Computing](https://www.youtube.com/watch?v=V1tINV2-9p4&list=PLoROMvodv4rMp7MTFr4hQsDEcX7Bx6Odp) 


---

## GPU content

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) — useful for understanding what Neuron does differently
- [Cornell GPU Architecture Guide](https://cvw.cac.cornell.edu/gpu-architecture) — GPU fundamentals for comparison
- [LLM Inference from Scratch (CUDA)](https://andrewkchan.dev/posts/yalm.html) — end-to-end GPU inference implementation
