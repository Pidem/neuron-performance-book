# Example 2: Getting OpenCLIP running on Neuron

*A model that fits the hardware well. What does "good" look like — and where do you go from here?*

---

## Introducing OpenCLIP

- OpenCLIP: open-source reproduction of CLIP (Contrastive Language-Image Pretraining)
- Two encoders: Vision Transformer (ViT) + Text Transformer, trained with contrastive loss
- Architecture is almost entirely matmuls + attention + LayerNorm — the ideal Neuron workload
- HCLS-relevant: BiomedCLIP (Microsoft) uses this architecture for medical image-text retrieval
- Available via `open_clip` package: https://github.com/mlfoundations/open_clip
- Good contrast with SchNet: this model *should* work well on Neuron — the question becomes "how well?" not "does it work?"

---

## Move to Neuron — clean compilation

- Load a pretrained OpenCLIP model (e.g., ViT-B/32 with text encoder)
- `model.to("neuron")` → eager runs cleanly, minimal or no fallback ops
- `torch.compile(model, backend="neuron")` → single fused NEFF, fast compilation
- Profile: dense tensor engine activity, no CPU fallback gaps
- This is what a well-matched workload looks like — the engines stay busy

---

## Profiling "good" — what the timeline should look like

- Neuron Explorer shows: tightly packed matmul waves (QKV projections, FFN layers)
- Vector/Scalar activity interleaved (LayerNorm, softmax) without idle gaps
- DMA engines pipelining weight loads behind compute
- Compare with SchNet profile: no PCIe round-trip gaps, high tensor engine utilization
- Read the summary tab: HFU, MFU, arithmetic intensity — establish baseline numbers
- "This is what you're aiming for when optimizing a kernel" — reference point for Part III

---

## The contrastive loss — shape dynamics

- Forward pass produces embeddings from both encoders: image_features, text_features
- Similarity matrix: `image_features @ text_features.T` — batch_size × batch_size
- As batch size grows, this matrix grows quadratically — memory pressure
- Interesting tradeoff: larger batch = better contrastive learning, but higher memory cost
- Profile at different batch sizes: when does SBUF pressure cause spilling?
- This introduces the reader to batch size as a performance lever (preview of roofline in Part III)

---

## Precision as the next frontier

- ViT is mostly matmuls — prime candidate for reduced precision (BF16 → FP8)
- Text encoder may be more sensitive to precision (embeddings, softmax over vocabulary)
- Experiment: cast ViT encoder to FP8, keep text encoder at BF16
- Measure: accuracy on retrieval task vs inference speed
- The asymmetric precision strategy: not all encoders need the same precision
- Foreshadow Part IV: "In Chapter 11, we'll study exactly which number formats work where and why"

---

## When the hardware fits — what's left to optimize?

- Model works, compiles cleanly, high MFU — is there anything left?
- Yes: batch size selection (memory-bound vs compute-bound), precision choices, serving configuration
- The optimization ladder continues: even a well-matched model benefits from roofline analysis and precision tuning
- Key insight: SchNet's problem was *functionality* (ops not supported). OpenCLIP's problem is *efficiency* (how close to peak can we get?)
- "Don't try to optimize something that already works well — profile first" (Neuron team advice)

*Question raised → "I can see the profiler says 60% MFU. How do I know what's achievable? What's the theoretical peak?"*

*Next: [Chapter 9](../part3/ch09-profiler) — The profiler (Part III).*
