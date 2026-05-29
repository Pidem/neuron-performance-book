# What actually happens when you call `model(x)`

*Opens with: ESM-2 predicting masked amino acids in a protein sequence.*

> You call `model(x)` and get predictions. But what actually happens between those parentheses?

## Setup

```python
import torch
from transformers import EsmModel, EsmTokenizer

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

# A simple protein sequence (insulin)
sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT"
inputs = tokenizer(sequence, return_tensors="pt")
output = model(**inputs)
```

What just happened? Let's trace it.

## Eager mode and the dispatcher

```{admonition} TODO
:class: warning
Draft content for this chapter.
```

## The dispatch table

## The computational graph (implicit)

## Why should I care?

```{admonition} Why should I care?
:class: tip
This dispatch mechanism is exactly how your code will run on Neuron without changes. Understanding it now means the GPU→Neuron migration in Part II will feel obvious.
```

---

*Question raised → "These ops operate on tensors. But what IS a tensor in memory?"*
