"""
Chapter 2: Tensors aren't just arrays
======================================
Exploring memory layout, strides, views, and contiguity
using ESM-2's embedding table and attention matrices.
"""

import torch
from transformers import EsmModel, EsmTokenizer

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
model.eval()

# --- 1. The embedding table: how is it stored? ---
embed = model.embeddings.word_embeddings.weight
print(f"=== ESM-2 Embedding Table ===")
print(f"Shape: {embed.shape}")  # [vocab_size, hidden_dim] = [33, 1280]
print(f"Dtype: {embed.dtype}")
print(f"Memory: {embed.nelement() * embed.element_size() / 1e6:.1f} MB")
print(f"Strides: {embed.stride()}")  # (1280, 1) — row-major, contiguous
print(f"Contiguous: {embed.is_contiguous()}")
print(f"Data pointer: {embed.data_ptr()}")

# --- 2. Strides: the key to understanding memory layout ---
print(f"\n=== Strides Explained ===")
# A stride tells you: how many elements to skip to get to the next index in that dimension
# For a [33, 1280] tensor with strides (1280, 1):
#   - To go to the next row (next token): skip 1280 elements
#   - To go to the next column (next feature): skip 1 element
print(f"To access embed[5, 100]:")
print(f"  Memory offset = 5 * {embed.stride()[0]} + 100 * {embed.stride()[1]} = {5 * embed.stride()[0] + 100 * embed.stride()[1]} elements")
print(f"  = {(5 * embed.stride()[0] + 100 * embed.stride()[1]) * embed.element_size()} bytes from start")

# --- 3. Views vs copies ---
print(f"\n=== Views vs Copies ===")
# A view shares memory with the original tensor (zero-copy)
row = embed[5]  # a view — no copy!
print(f"embed[5] is a view: {row.data_ptr() == embed.data_ptr() + 5 * embed.stride()[0] * embed.element_size()}")
print(f"embed[5] shape: {row.shape}, strides: {row.stride()}")

# Transpose creates a view with swapped strides — no data movement!
embed_t = embed.T
print(f"\nembed.T shape: {embed_t.shape}")
print(f"embed.T strides: {embed_t.stride()}")  # (1, 1280) — columns are now contiguous
print(f"embed.T is contiguous: {embed_t.is_contiguous()}")  # False!
print(f"Same memory: {embed_t.data_ptr() == embed.data_ptr()}")  # True — just a view

# --- 4. When reshape copies vs doesn't ---
print(f"\n=== Reshape: copy or view? ===")
# Reshape on a contiguous tensor → view (no copy)
reshaped = embed.reshape(-1)  # flatten
print(f"Flatten contiguous tensor: is view = {reshaped.data_ptr() == embed.data_ptr()}")

# Reshape on a non-contiguous tensor → COPY
reshaped_t = embed_t.reshape(-1)  # must copy because strides don't allow it
print(f"Flatten transposed tensor: is view = {reshaped_t.data_ptr() == embed_t.data_ptr()}")
print("^ This triggered a memory copy! Performance implication.")

# --- 5. Attention matrices: where layout matters ---
print(f"\n=== Attention Matrix Layout ===")
sequence = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"
inputs = tokenizer(sequence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# Attention weights shape: [batch, heads, seq_len, seq_len]
attn = outputs.attentions[0]  # first layer
print(f"Attention shape: {attn.shape}")
print(f"Attention strides: {attn.stride()}")
print(f"Memory: {attn.nelement() * attn.element_size() / 1e3:.1f} KB")

# Accessing one head's attention pattern:
head_0 = attn[0, 0]  # [seq_len, seq_len] — is this contiguous?
print(f"\nSingle head attention:")
print(f"  Shape: {head_0.shape}")
print(f"  Strides: {head_0.stride()}")
print(f"  Contiguous: {head_0.is_contiguous()}")

# --- 6. Performance implications ---
print(f"\n=== Performance Implications ===")

# Contiguous access is fast, strided access is slow
import time

big_tensor = torch.randn(4096, 4096)

# Row access (contiguous) vs column access (strided)
start = time.perf_counter()
for _ in range(1000):
    _ = big_tensor[0].sum()  # row — contiguous
row_time = time.perf_counter() - start

start = time.perf_counter()
for _ in range(1000):
    _ = big_tensor[:, 0].sum()  # column — strided
col_time = time.perf_counter() - start

print(f"Row access (contiguous):  {row_time*1000:.2f} ms for 1000 iterations")
print(f"Column access (strided):  {col_time*1000:.2f} ms for 1000 iterations")
print(f"Ratio: {col_time/row_time:.1f}x slower")

print("""
=== Key Takeaway ===
Tensors are not just "arrays of numbers." They are:
- A data pointer (where the bytes live)
- A shape (logical dimensions)
- Strides (how to navigate memory)

This means:
- Transpose is free (just swap strides) but makes the tensor non-contiguous
- Non-contiguous tensors force copies when you reshape
- Memory access patterns (contiguous vs strided) directly affect performance
- On Neuron, DMA engines load tiles from HBM — contiguous tiles load faster

Next chapter: Can a compiler optimize these access patterns for us?
""")
