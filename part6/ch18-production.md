# Putting it in production

*Fine-tune, deploy, serve. The last mile from optimized model to live traffic.*

---

## The deployment stack

You've optimized your model through Parts I–V. Now you need to package it for real users. Neuron's production stack mirrors the GPU ecosystem — same libraries, different backend:

| Stage | Library | What it does |
|-------|---------|-------------|
| Fine-tuning | **optimum-neuron** | HuggingFace Transformers API, LoRA, minimal code changes |
| Serving | **vLLM Neuron plugin** | High-throughput LLM inference with continuous batching |
| Training at scale | **TorchTitan** | Standard PyTorch distributed training (FSDP, TP, PP) |

---

## Fine-tuning with optimum-neuron

[optimum-neuron](https://huggingface.co/docs/optimum-neuron) wraps the HuggingFace Transformers API with Neuron-optimized kernels underneath. The workflow:

**Step 1: Load and format your dataset**

```python
from datasets import load_dataset

dataset = load_dataset("your-org/your-data")
# Format for instruction tuning (chat template)
```

**Step 2: Fine-tune with LoRA**

```python
from optimum.neuron import NeuronTrainer, NeuronTrainingArguments
from peft import LoraConfig

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])

training_args = NeuronTrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    bf16=True,
)

trainer = NeuronTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    peft_config=lora_config,
)
trainer.train()
```

The code is nearly identical to standard Transformers fine-tuning — `NeuronTrainer` replaces `Trainer`, `NeuronTrainingArguments` replaces `TrainingArguments`. Under the hood, optimum-neuron integrates Neuron's compiled kernels for attention, MLP, and normalization layers.

**Step 3: Consolidate LoRA adapters**

```bash
optimum-cli neuron consolidate --model ./results --output ./merged-model
```

**Step 4 (optional): Push to HuggingFace Hub**

```python
model.push_to_hub("your-org/your-model-neuron")
```

---

## Serving with vLLM

[vLLM](https://docs.vllm.ai/) is the standard open-source library for high-throughput LLM serving. The Neuron plugin provides:

- **Flash attention** — fused NKI kernel for the attention computation
- **Fused QKV** — single kernel for query/key/value projection
- **Speculative decoding** — draft model generates candidates, main model verifies in batch
- **Continuous batching** — new requests join mid-generation without waiting
- **Paged attention** — efficient KV cache management (no fragmentation)

Supported models: Llama, Qwen, GPTOSS, Mistral, DeepSeek (and growing).

```python
from vllm import LLM, SamplingParams

sampling = SamplingParams(temperature=0.7, max_tokens=512)

llm = LLM(
    model="your-org/your-model-neuron",
    tensor_parallel_size=2,  # Shard across 2 NeuronCores
)

outputs = llm.generate(["Summarize the mechanism of action of..."], sampling)
```

That's it. Same API as GPU vLLM. The `tensor_parallel_size` parameter controls how many NeuronCores share the model weights — same concept as GPU tensor parallelism.

---

## Training at scale with TorchTitan

For pre-training or full fine-tuning at scale, [TorchTitan](https://github.com/pytorch/torchtitan) is a standard PyTorch training library that works on Neuron out of the box:

- 10+ model architectures supported (Llama, GPT, etc.)
- FSDP, TP, PP, CP, EP — all via standard PyTorch APIs
- No Neuron-specific code — upstream contributions for portability
- Dense models reach >50% MFU on Trn2

---

## The two inference modes

For production serving, you choose between two optimization targets:

| Mode | Optimize for | Use case |
|------|-------------|----------|
| **Latency-optimized** | Time to first token, tokens/second per user | Interactive chat, coding assistants |
| **Throughput-optimized** | Total tokens/second across all users | Batch processing, RL reward models, offline eval |

The hardware is the same — the difference is in batching strategy, KV cache allocation, and scheduling policy. vLLM handles this via configuration.

---

## The full lifecycle

```
Select model (open-weight, HuggingFace)
    │
    ▼
Fine-tune (optimum-neuron, LoRA, Trn2)
    │
    ▼
Profile (Neuron Explorer — is attention optimized?)
    │
    ▼
Optimize (torch.compile → NKI if needed)
    │
    ▼
Deploy (vLLM Neuron plugin, tensor parallelism)
    │
    ▼
Monitor & iterate (latency, throughput, cost)
```

```{admonition} The open-source commitment
:class: note
The entire Neuron developer stack is being open-sourced: NKI compiler, torch-neuronx backend, Neuron Kernel Library (pre-optimized production kernels), and all vLLM/HuggingFace plugins. You're never locked into a proprietary toolchain.
```

---

*This concludes the book. You started with `model(x)` as a black box and ended with instruction-level control over custom silicon. The path from data scientist to performance engineer is a ladder — each rung gives you more control and more performance. Climb as high as your workload demands.*
