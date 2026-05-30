"""
Chapter 1: The PyTorch Dispatcher — Same Math, Different Machine Code
=====================================================================

Thesis: A transformer is just ~5 operations composed 33 times.
        The DISPATCHER decides HOW each op actually executes.
        Same model code → different kernels depending on hardware + context.

We'll prove this in 4 acts:
  Act 1: Profile ESM-2, identify the actual ops
  Act 2: Build a naked model with exactly those ops
  Act 3: Show the SAME op dispatching to DIFFERENT implementations
  Act 4: Show fusion — multiple ops collapsing into one kernel
"""

import torch
import torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer

# ============================================================
# ACT 1: What ops does ESM-2 actually call?
# ============================================================

print("=" * 70)
print("ACT 1: Profiling ESM-2 — what's really happening?")
print("=" * 70)

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model.eval()

# Human insulin B chain
sequence = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"
inputs = tokenizer(sequence, return_tensors="pt")
print(f"\nInput: {sequence}")
print(f"Tokenized shape: {inputs['input_ids'].shape}")  # [1, 32]
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Model parameters: {sum(p.numel() for p in esm_model.parameters()):,}")

# Profile the forward pass
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True,
) as prof:
    with torch.no_grad():
        output = esm_model(**inputs)

# Find the attention-related ops
print(f"\nOutput: {output.last_hidden_state.shape}")
print(f"\n{'Op Name':<55} {'Count':>6} {'Total ms':>10}")
print("-" * 75)

target_ops = ['matmul', 'addmm', 'masked_fill', 'softmax', 'scaled_dot_product',
              'layer_norm', 'gelu', 'linear', 'mul', 'add']
events = prof.key_averages()
found_ops = []
for e in sorted(events, key=lambda e: e.cpu_time_total, reverse=True):
    if any(op in e.key.lower() for op in target_ops):
        found_ops.append(e)
        print(f"  {e.key:<53} {e.count:>6} {e.cpu_time_total/1000:>9.1f}")
    if len(found_ops) >= 15:
        break

print("""
┌─────────────────────────────────────────────────────────────────┐
│ OBSERVATION: ESM-2 (650M params, 33 layers) decomposes into     │
│ just a handful of ATen ops repeated hundreds of times.          │
│                                                                 │
│ The "intelligence" is in the WEIGHTS, not the ops.              │
│ The ops are generic linear algebra.                             │
└─────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# ACT 1.5: EAGER EXECUTION — Why there's no compilation
# ============================================================

print("=" * 70)
print("ACT 1.5: Eager Execution — Python is the runtime")
print("=" * 70)

print("""
What does "eager" mean?

  EAGER MODE (PyTorch default):
    Python executes your code LINE BY LINE.
    Each operation runs IMMEDIATELY and returns a result.
    There is NO "graph" being built behind the scenes.
    There is NO compilation step before execution.
    Python IS the executor.

  Think of it like a calculator:
    You type "2 + 3" → you get 5 immediately.
    You don't type all your equations first and then press "compile & run."

  This is DIFFERENT from:
    - TensorFlow 1.x (build a graph first, then sess.run())
    - torch.compile (trace the code, build a graph, optimize, then run)
    - XLA/Neuron (trace lazy ops, compile to hardware-specific binary)

Let's PROVE it with a live experiment.
""")

# --- Proof 1: Operations execute immediately ---
print("--- PROOF 1: Operations execute immediately ---\n")

import time

a = torch.randn(1000, 1000)
b = torch.randn(1000, 1000)

print("About to do matmul...")
t0 = time.perf_counter()
c = torch.matmul(a, b)  # This RUNS right now. Not "scheduled." Not "queued." DONE.
t1 = time.perf_counter()
print(f"matmul finished in {(t1-t0)*1000:.2f}ms")
print(f"Result already exists: c.shape = {c.shape}, c[0,0] = {c[0,0]:.4f}")
print(f"No .run(), no .execute(), no .compile(). It just... happened.\n")


# --- Proof 2: We can inspect intermediate values at any time ---
print("--- PROOF 2: We can inspect intermediates (impossible in graph mode) ---\n")

# In a compiled/graph framework, you CAN'T do this — intermediates don't exist yet
x = torch.randn(1, 32, 1280)  # Pretend this is ESM-2's hidden state

# Step 1: Layer norm
ln = torch.nn.LayerNorm(1280)
x_normed = ln(x)
print(f"After layer_norm: mean={x_normed.mean():.6f}, std={x_normed.std():.4f}")
# ^ We can print this BETWEEN operations. The value is REAL. It's computed.

# Step 2: Linear
linear = torch.nn.Linear(1280, 1280)
x_proj = linear(x_normed)
print(f"After linear: shape={x_proj.shape}, first value={x_proj[0,0,0]:.4f}")
# ^ Again — we can inspect it. It's a real tensor with real numbers.

print("""
In a graph/compiled framework, these intermediate tensors wouldn't have
values yet — they'd be PLACEHOLDERS waiting for the graph to execute.
In eager mode, every tensor has a value the instant it's created.
""")


# --- Proof 3: Hooks — spying on execution order ---
print("--- PROOF 3: Hooks — watching execution happen in real time ---\n")

print("""
What is a "hook"?

  A hook is a callback function you attach to a module.
  PyTorch calls it at a specific moment during execution:

    FORWARD HOOK:  called AFTER module.forward() finishes
    FORWARD PRE-HOOK: called BEFORE module.forward() starts

  Think of it like a security camera at a door:
    - You put a camera on the door
    - Every time someone walks through, the camera records it
    - The camera doesn't CHANGE anything — it just observes

  Why does this prove eager execution?
    Because hooks fire IN ORDER, ONE AT A TIME, as Python walks through
    the model. If there were a graph, we'd see all hooks fire at once
    or in an optimized/shuffled order.
""")

# Let's put "cameras" on ESM-2's layers and watch them fire
execution_log = []

def make_hook(layer_name):
    """Factory that creates a hook which logs when this layer runs."""
    def hook_fn(module, input, output):
        # This function is called by PyTorch AFTER the module runs
        execution_log.append({
            'step': len(execution_log) + 1,
            'name': layer_name,
            'output_shape': output[0].shape if isinstance(output, tuple) else output.shape
        })
    return hook_fn

# Attach hooks to the first 3 transformer layers of ESM-2
# Each transformer layer has: attention → intermediate → output
hook_handles = []
for i in range(3):
    layer = esm_model.encoder.layer[i]
    
    # Hook on the self-attention
    h = layer.attention.self.register_forward_hook(make_hook(f"Layer {i} → Self-Attention"))
    hook_handles.append(h)
    
    # Hook on the attention output (residual + layernorm)
    h = layer.attention.output.register_forward_hook(make_hook(f"Layer {i} → Attention Output"))
    hook_handles.append(h)
    
    # Hook on the feed-forward intermediate
    h = layer.intermediate.register_forward_hook(make_hook(f"Layer {i} → FFN Intermediate"))
    hook_handles.append(h)
    
    # Hook on the layer output
    h = layer.output.register_forward_hook(make_hook(f"Layer {i} → FFN Output"))
    hook_handles.append(h)

# Run the model — hooks will fire as Python steps through each module
execution_log.clear()
with torch.no_grad():
    _ = esm_model(**inputs)

print(f"Execution trace (first 3 layers, {len(execution_log)} events):\n")
print(f"  {'Step':<6} {'Module':<35} {'Output Shape'}")
print(f"  {'-'*6} {'-'*35} {'-'*20}")
for entry in execution_log:
    print(f"  {entry['step']:<6} {entry['name']:<35} {entry['output_shape']}")

# Clean up hooks (important! hooks persist otherwise)
for h in hook_handles:
    h.remove()

print("""
┌─────────────────────────────────────────────────────────────────┐
│ Notice the order: Layer 0 fully completes, then Layer 1, etc.   │
│                                                                 │
│ This IS Python's execution order. No scheduler. No optimizer.   │
│ No graph. Python calls forward(), which calls sub-forward()s,   │
│ which call ATen ops, which hit the dispatcher.                  │
│                                                                 │
│ The call stack right now:                                       │
│                                                                 │
│   model(**inputs)                                               │
│     → model.forward()             [Python method call]          │
│       → embeddings(input_ids)     [Python method call]          │
│       → encoder(hidden_states)    [Python method call]          │
│         → layer[0](x)            [Python method call]           │
│           → attention.self(x)    [Python method call]           │
│             → torch.matmul(Q,K)  [ATen op → Dispatcher → MKL]  │
│             → masked_fill(...)   [ATen op → Dispatcher → CPU]   │
│             → softmax(...)       [ATen op → Dispatcher → CPU]   │
│             → torch.matmul(a,V)  [ATen op → Dispatcher → MKL]  │
│           ← returns              [back to Python]               │
│           → attention.output(x)  [Python method call]           │
│             → layer_norm(...)    [ATen op → Dispatcher → CPU]   │
│           ...and so on                                          │
│                                                                 │
│ Every indentation = a Python function call.                     │
│ Every ATen op = an immediate computation. No delay.             │
│                                                                 │
│ THAT'S EAGER MODE.                                              │
└─────────────────────────────────────────────────────────────────┘
""")

print("""
WHY does this matter?

  PROS of eager mode:
    ✓ Easy to debug (print anything, set breakpoints anywhere)
    ✓ Dynamic control flow (if/else, loops that depend on tensor values)
    ✓ Immediate error messages (crash on the exact line that's wrong)
    ✓ Natural Python — no special "graph language" to learn

  CONS of eager mode:
    ✗ Python overhead on every single op (GIL, interpreter, etc.)
    ✗ The dispatcher can't see the big picture (no global optimization)
    ✗ Can't fuse ops across module boundaries without help
    ✗ Can't pre-compile for specialized hardware (Neuron, TPU)

  torch.compile() (Chapter 3) fixes the cons by TRACING eager code
  into a graph — but your code stays eager. You write eager, the
  compiler sees a graph. Best of both worlds.
""")

# ============================================================
# ACT 2: Rebuild attention from the exact same ops
# ============================================================

print("=" * 70)
print("ACT 2: Naked attention — same ops, no framework magic")
print("=" * 70)

class ESMAttentionManual(torch.nn.Module):
    """
    This does EXACTLY what one ESM-2 attention layer does,
    using the same ATen ops we saw in the profiler.
    """
    def __init__(self, d_model=1280, n_heads=20):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # 64
        self.scale = self.d_head ** -0.5
        
        # These trigger: aten::addmm (or aten::linear)
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
    
    def forward(self, x, attention_mask=None):
        B, L, D = x.shape
        
        # --- Op 1: aten::addmm (linear projection) ---
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        # Q, K, V: [B, n_heads, L, d_head]
        
        # --- Op 2: aten::matmul (Q @ K^T) ---
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores: [B, n_heads, L, L]
        
        # --- Op 3: aten::masked_fill (mask padding tokens) ---
        if attention_mask is not None:
            # attention_mask: [B, 1, 1, L] with 0s where we want to mask
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # --- Op 4: aten::softmax ---
        attn_weights = torch.softmax(scores, dim=-1)
        
        # --- Op 5: aten::matmul (attn @ V) ---
        context = torch.matmul(attn_weights, V)
        # context: [B, n_heads, L, d_head]
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        
        # --- Op 6: aten::addmm (output projection) ---
        return self.out_proj(context)


# Run with same dimensions as ESM-2
manual_attn = ESMAttentionManual(d_model=1280, n_heads=20)
manual_attn.eval()

x = torch.randn(1, 32, 1280)  # Same shape as ESM-2's hidden states
mask = torch.ones(1, 1, 1, 32)
mask[:, :, :, -2:] = 0  # Simulate 2 padding tokens

# Profile our manual version
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True,
) as prof2:
    with torch.no_grad():
        out_manual = manual_attn(x, mask)

print(f"\nManual attention output: {out_manual.shape}")
print(f"\n{'Op Name':<55} {'Count':>6}")
print("-" * 65)
for e in sorted(prof2.key_averages(), key=lambda e: e.count, reverse=True)[:10]:
    if any(op in e.key.lower() for op in target_ops):
        print(f"  {e.key:<53} {e.count:>6}")

print("""
┌─────────────────────────────────────────────────────────────────┐
│ SAME OPS. Our 30-line model triggers the exact same ATen ops    │
│ as a 650M-parameter protein language model.                     │
│                                                                 │
│ The dispatcher doesn't know it's doing biology vs. a toy.       │
│ It just sees matmul, masked_fill, softmax, matmul.              │
└─────────────────────────────────────────────────────────────────┘
""")


# ============================================================
# ACT 3: Same operation, different dispatch
# ============================================================

print("=" * 70)
print("ACT 3: The dispatcher in action — same op, different kernel")
print("=" * 70)

print("""
The dispatcher's job: given an op + tensor metadata, pick the right kernel.

    torch.matmul(A, B)
         │
         ▼
    ┌─────────────┐
    │  DISPATCHER  │ ← looks at: dtype, device, shape, layout
    └──────┬──────┘
           │
    ┌──────┼──────────────────────────┐
    │      │      │         │         │
    ▼      ▼      ▼         ▼         ▼
   MKL   cuBLAS  Neuron   Metal    oneDNN
  (CPU)  (CUDA)  (NKI)   (MPS)   (Intel)
""")

# Demonstrate: same matmul, different dispatch keys
A = torch.randn(32, 64)
B = torch.randn(64, 128)

# CPU dispatch
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p_cpu:
    C_cpu = torch.matmul(A, B)

print("CPU matmul dispatch:")
for e in p_cpu.key_averages():
    if 'matmul' in e.key or 'mm' in e.key:
        print(f"  → {e.key}")

print(f"\nA's device: {A.device}, dtype: {A.dtype}")

# If CUDA available, show the difference
if torch.cuda.is_available():
    A_cuda = A.cuda()
    B_cuda = B.cuda()
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, 
                    torch.profiler.ProfilerActivity.CUDA]
    ) as p_cuda:
        C_cuda = torch.matmul(A_cuda, B_cuda)
    
    print("\nCUDA matmul dispatch:")
    for e in p_cuda.key_averages():
        if 'matmul' in e.key or 'mm' in e.key or 'gemm' in e.key.lower():
            print(f"  → {e.key}")
    
    print(f"\n  Same operation, same result: {torch.allclose(C_cpu, C_cuda.cpu(), atol=1e-5)}")
    print(f"  But underneath: MKL (CPU) vs cuBLAS (CUDA) — completely different machine code.")
else:
    print("\n  [No CUDA available — but the principle holds:]")
    print("  torch.matmul dispatches to MKL/OpenBLAS on CPU, cuBLAS on CUDA,")
    print("  NKI on Neuron, BNNS/Metal on Apple Silicon — all via the SAME Python call.")


# ============================================================
# ACT 4: Fusion — collapsing ops into one kernel
# ============================================================

print("\n" + "=" * 70)
print("ACT 4: Fusion — the dispatcher's optimization trick")
print("=" * 70)

print("""
Manual attention:                     Fused SDPA:
  matmul(Q, K^T)                        ┐
  multiply by scale                     │
  masked_fill(mask, -inf)               ├→ ONE kernel call
  softmax                               │   (scaled_dot_product_attention)
  matmul(attn, V)                       ┘

5 op dispatches → 1 op dispatch. Same math. Fewer kernel launches.
""")

class ESMAttentionFused(torch.nn.Module):
    """Same computation as ESMAttentionManual, but using the fused SDPA op."""
    
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
        
        # ONE OP replaces matmul + scale + mask + softmax + matmul
        context = F.scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=attention_mask,  # mask built in
            scale=self.d_head ** -0.5
        )
        
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(context)


# Copy weights so we can compare outputs
fused_attn = ESMAttentionFused(d_model=1280, n_heads=20)
fused_attn.load_state_dict(manual_attn.state_dict())
fused_attn.eval()

# Profile both
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p_manual:
    with torch.no_grad():
        out_m = manual_attn(x, mask)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p_fused:
    with torch.no_grad():
        # SDPA expects mask shape [B, n_heads, L, L] or broadcastable
        sdpa_mask = mask.expand(1, 20, 32, 32).bool()
        out_f = fused_attn(x, sdpa_mask)

# Count ops
def count_target_ops(profiler, targets):
    count = 0
    for e in profiler.key_averages():
        if any(t in e.key.lower() for t in targets):
            count += e.count
    return count

attn_ops = ['matmul', 'masked_fill', 'softmax', 'scaled_dot_product', 'mul']
manual_count = count_target_ops(p_manual, attn_ops)
fused_count = count_target_ops(p_fused, attn_ops)

print(f"Manual path — attention-related op dispatches: {manual_count}")
print(f"Fused path  — attention-related op dispatches: {fused_count}")
print(f"Outputs match: {torch.allclose(out_m, out_f, atol=1e-4)}")

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│ SAME MATH. Fewer dispatches. That's fusion.                     │
│                                                                 │
│ The fused kernel does matmul+mask+softmax+matmul in ONE shot:   │
│ • Fewer kernel launches (less overhead)                         │
│ • Data stays in fast memory (no round-trips to DRAM)            │
│ • On GPU: FlashAttention under the hood                         │
│                                                                 │
│ The dispatcher picks the fused path automatically on GPU.       │
│ On CPU it falls back to the manual decomposition.               │
└─────────────────────────────────────────────────────────────────┘
""")


# ============================================================
# EPILOGUE: The full picture
# ============================================================

print("=" * 70)
print("THE BIG PICTURE")
print("=" * 70)
print("""
    ESM-2("FVNQHLCGSHLVEALYLVCGERGFFYTPKT")
         │
         ▼
    ┌──────────────────────────────────┐
    │  Python: model.forward()         │  ← Eager mode (Act 1-2)
    │  Each line triggers an ATen op   │
    └──────────────┬───────────────────┘
                   │
         ┌─────────┼─────────┐
         ▼         ▼         ▼
    aten::matmul  aten::softmax  aten::masked_fill  ...
         │         │         │
         ▼         ▼         ▼
    ┌──────────────────────────────────┐
    │  DISPATCHER                      │  ← Act 3: routes by device/dtype
    │  Checks: device? dtype? layout?  │
    └──────────────┬───────────────────┘
                   │
    ┌──────────────┼──────────────────────────────┐
    │              │              │                │
    ▼              ▼              ▼                ▼
  ┌─────┐     ┌───────┐    ┌────────┐     ┌──────────┐
  │ MKL │     │cuBLAS │    │ Neuron │     │  Metal   │
  │(CPU)│     │(CUDA) │    │ (NKI)  │     │ (Apple)  │
  └─────┘     └───────┘    └────────┘     └──────────┘
    │              │              │                │
    ▼              ▼              ▼                ▼
  x86 asm      PTX/SASS       NEFF           GPU shader

Fusion (Act 4) collapses multiple ops into one BEFORE dispatching:
  [matmul + mask + softmax + matmul] → [scaled_dot_product_attention]
  Fewer dispatches = fewer kernel launches = faster.

YOUR MODEL CODE NEVER CHANGES. Only the dispatch target does.
That's the abstraction. That's why "just change the device" works.

Next: Chapter 2 — What ARE these tensors? (memory layout, strides, views)
""")
