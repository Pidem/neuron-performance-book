# From Data Scientist to Performance Engineer

*Making the most of your PyTorch models on Neuron chips*

## Who is this for?

You're a scientist at a life sciences company. You've trained protein language models, molecular property predictors, or clinical NLP systems on GPUs. You're comfortable with PyTorch and Python. But you've never thought about *why* your training takes the time it does, or what's actually happening on the hardware underneath.

This book takes you from "I can train a model" to "I understand why my kernel is memory-bound on NeuronCore-v3" — through a structured mental model ladder.

## The journey

We follow **ESM-2** (a protein language model) from a black-box `model(x)` call all the way down to hand-written NKI kernels. Each chapter peels back one layer of abstraction. Each chapter ends with a question that the next one answers.

```{admonition} The optimization ladder
:class: tip
**Eager** (get it running) → **torch.compile** (graph-level perf) → **NKI** (kernel-level perf)
```

## How to use this book

- **Read linearly** — chapters build on each other
- **Run the notebooks** — every concept has runnable code on `trn2.xlarge`
- **Skip to your level** — if you already know PyTorch internals, start at Part II

## Table of contents

```{tableofcontents}
```
