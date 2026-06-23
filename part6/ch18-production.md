# Putting it in production

*Fine-tune Llama 3.2 1B on PubMedQA, deploy with vLLM on Neuron.*

```{admonition} TODO
:class: warning
Draft content.
```

```{admonition} Notes from Neuron team talks (2026)
:class: note
**vLLM on Neuron:**
- Integration via native PyTorch plugin format
- Flash attention, fused QKV, speculative decoding pre-integrated
- Continuous batching, paged attention for KV cache management
- Disaggregated inference: building blocks ready, first integration with Bedrock/Mantle
- Advanced features coming: prefix caching, chunked/segmented prefill, lower serving latency

**HuggingFace:**
- Transformers V5 porting underway — HF team "very excited" about ease of native PyTorch integration
- Early integration shows "easy onboarding with minimal frictions"
- optimum-neuron for fine-tuning (LoRA, HF API)

**TorchTitan:**
- Standard PyTorch training library, 10+ models work out of the box on Neuron
- FSDP, TP, PP, CP, EP all supported
- Upstream contributions for portability (no Neuron-specific code)
- GPToss 20B training demo on trn2.48xlarge
- Dense models reach >50% MFU

**Two inference modes (from Poolside talk):**
- Latency-optimized (interactive serving)
- Throughput-optimized (batch/RL workloads)

**Agentic tools:**
- AWS Transform for porting CUDA code to Neuron
- Kiro skills + Cloud Code skills for kernel authoring
- Descartes feedback: "tasks that took weeks now take days to hours"
```
