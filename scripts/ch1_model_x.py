"""
Chapter 1: What actually happens when you call model(x)
========================================================
Run on trn2.3xlarge:
    source /workshop/workspace/native_venv/bin/activate
    pip install transformers
    python scripts/ch1_model_x.py
"""

import torch
import torch.nn.functional as F
import time
from transformers import EsmModel, EsmTokenizer

print(f"Torch version: {torch.__version__}")

# ─── ESM-2 forward pass ───────────────────────────────────────────────────────

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
model.eval()

sequence = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"
inputs = tokenizer(sequence, return_tensors="pt")

print(f"Input shape: {inputs['input_ids'].shape}")
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ─── Profile CPU forward pass ─────────────────────────────────────────────────

print("\n--- CPU profiling (top 10 ops) ---")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True,
) as prof:
    with torch.no_grad():
        output = model(**inputs)

events = prof.key_averages()
for e in sorted(events, key=lambda e: e.cpu_time_total, reverse=True)[:10]:
    print(f"  {e.key:<50} called {e.count:>4} times, total {e.cpu_time_total/1000:>7.1f}ms")

# ─── Move to Neuron ───────────────────────────────────────────────────────────

print("\n--- Running on Neuron ---")
model_neuron = model.to("neuron")
inputs_neuron = {k: v.to("neuron") for k, v in inputs.items()}

with torch.no_grad():
    output_neuron = model_neuron(**inputs_neuron)

print(f"Output device: {output_neuron.last_hidden_state.device}")
print(f"Output shape:  {output_neuron.last_hidden_state.shape}")
print(f"Same result:   {torch.allclose(output.last_hidden_state, output_neuron.last_hidden_state.cpu(), atol=1e-3)}")

# ─── Dispatcher: CPU vs Neuron ─────────────────────────────────────────────────

print("\n--- Dispatcher demo: same aten::mm, different backend ---")
A = torch.randn(32, 64)
B = torch.randn(64, 128)

print("CPU:")
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as p:
    C_cpu = torch.matmul(A, B)
for e in p.key_averages():
    if 'mm' in e.key:
        print(f"  {e.key}  cpu_time={e.cpu_time_total:.0f}µs")

A_n, B_n = A.to("neuron"), B.to("neuron")
print("Neuron first pass (with compilation):")
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as p:
    C_neuron = torch.matmul(A_n, B_n)
for e in p.key_averages():
    if 'mm' in e.key or 'neuron' in e.key.lower():
        print(f"  {e.key}  cpu_time={e.cpu_time_total:.0f}µs")

print("Neuron second pass (cached NEFF):")
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as p:
    C_neuron2 = torch.matmul(A_n, B_n)
for e in p.key_averages():
    if 'mm' in e.key or 'neuron' in e.key.lower():
        print(f"  {e.key}  cpu_time={e.cpu_time_total:.0f}µs")

print(f"Same result: {torch.allclose(C_cpu, C_neuron.cpu(), atol=1e-3)}")

# ─── Fusion: manual vs SDPA ───────────────────────────────────────────────────

print("\n--- Op fusion: manual attention vs fused SDPA ---")


class ESMAttentionManual(torch.nn.Module):
    def __init__(self, d_model=1280, n_heads=20):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x, attention_mask=None):
        B, L, D = x.shape
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(context)


class ESMAttentionFused(torch.nn.Module):
    def __init__(self, d_model=1280, n_heads=20):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x, attention_mask=None):
        B, L, D = x.shape
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        context = F.scaled_dot_product_attention(Q, K, V, attn_mask=attention_mask)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(context)


manual_attn = ESMAttentionManual().eval()
fused_attn = ESMAttentionFused().eval()
x = torch.randn(1, 32, 1280)

attn_core_ops = ['bmm', 'matmul', 'softmax', '_softmax', 'scaled_dot_product', 'masked_fill']

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p1:
    with torch.no_grad():
        manual_attn(x)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p2:
    with torch.no_grad():
        fused_attn(x)

print("Manual path (attention core ops):")
for e in p1.key_averages():
    if any(op in e.key for op in attn_core_ops):
        print(f"  {e.key:<45} ×{e.count}")
print("Fused path (attention core ops):")
for e in p2.key_averages():
    if any(op in e.key for op in attn_core_ops):
        print(f"  {e.key:<45} ×{e.count}")

# ─── Async execution demo ─────────────────────────────────────────────────────

print("\n--- Async execution: dispatch vs actual device time ---")
A = torch.randn(4096, 4096, device="neuron", dtype=torch.bfloat16)
B = torch.randn(4096, 4096, device="neuron", dtype=torch.bfloat16)

start = time.time()
C = torch.matmul(A, B)
dispatch_time = time.time() - start

start = time.time()
C = torch.matmul(A, B)
torch.neuron.synchronize()                 # ⚠️  Always synchronize before taking timestamps!
exec_time = time.time() - start

print(f"Dispatch only: {dispatch_time*1000:.1f}ms")
print(f"With sync:     {exec_time*1000:.1f}ms")

