"""
OpenCLIP on Neuron — Investigation Script
==========================================
Unlike SchNet (Ch7), CLIP is a pure transformer model. The question:
does it "just work" on Neuron, or are there hidden bottlenecks?

Run on trn2.3xlarge:
    source /workshop/workspace/native_venv/bin/activate
    pip install -e open_clip
    python scripts/ch8_openclip_example.py
"""

import torch
import time
import sys
sys.path.insert(0, "open_clip/src")

import open_clip

DEVICE = "neuron"

# ============================================================
print("=" * 70)
print("PART 1: MOVING OPENCLIP TO NEURON")
print("=" * 70)

# Create ViT-B/32 (smallest standard CLIP) with random weights
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained=None, precision="fp32"
)
model = model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Synthetic inputs
batch_size = 8
images = torch.randn(batch_size, 3, 224, 224)
text = tokenizer(["a photo of a cat"] * batch_size)

# CPU reference
with torch.no_grad():
    img_feat_cpu = model.encode_image(images, normalize=True)
    txt_feat_cpu = model.encode_text(text, normalize=True)
print(f"\nCPU output shapes: image={img_feat_cpu.shape}, text={txt_feat_cpu.shape}")

# Move to Neuron
model_neuron = model.to(DEVICE)
images_neuron = images.to(DEVICE)
text_neuron = text.to(DEVICE)

# Vision encoder
with torch.no_grad():
    img_feat_neuron = model_neuron.encode_image(images_neuron, normalize=True)
img_diff = (img_feat_cpu - img_feat_neuron.cpu()).abs().max().item()
print(f"\nVision encoder on Neuron: ✓ (max diff={img_diff:.6f})")

# Text encoder — has argmax pooling which uses fancy indexing
try:
    with torch.no_grad():
        txt_feat_neuron = model_neuron.encode_text(text_neuron, normalize=True)
    txt_diff = (txt_feat_cpu - txt_feat_neuron.cpu()).abs().max().item()
    print(f"Text encoder on Neuron:   ✓ (max diff={txt_diff:.6f})")
    text_works = True
except Exception as e:
    print(f"Text encoder on Neuron:   ✗ FAILED")
    print(f"  Error: {type(e).__name__}: {str(e)[:200]}")
    print(f"  Root cause: text_global_pool uses x[arange, text.argmax()] — fancy indexing")
    text_works = False

# ============================================================
print("\n" + "=" * 70)
print("PART 2: PROFILING — WHAT FALLS BACK TO CPU?")
print("=" * 70)

# Skip NeuronDebugContext — use compile graph breaks instead
print("\nUsing torch.compile to detect graph breaks...")
torch._dynamo.reset()
import torch._dynamo as dynamo
explanation = dynamo.explain(model_neuron.encode_image)(images_neuron, normalize=True)
print(f"  Graph count: {explanation.graph_count}")
print(f"  Graph break count: {explanation.graph_break_count}")
if explanation.graph_break_count > 0:
    print(f"  Break reasons: {explanation.break_reasons[:5]}")

# ============================================================
print("\n" + "=" * 70)
print("PART 3: TORCH.COMPILE")
print("=" * 70)

def bench(fn, n_iter=100, warmup=20):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.neuron.synchronize()
        start = time.time()
        for _ in range(n_iter):
            fn()
        torch.neuron.synchronize()
    return (time.time() - start) / n_iter * 1000

def bench_cpu(fn, n_iter=30, warmup=5):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        start = time.time()
        for _ in range(n_iter):
            fn()
    return (time.time() - start) / n_iter * 1000

# Eager baseline (vision only — the reliable path)
t_eager_img = bench(lambda: model_neuron.encode_image(images_neuron, normalize=True))
print(f"\nVision encoder (batch={batch_size}):")
print(f"  Eager:    {t_eager_img:.2f}ms")

# Compiled
torch._dynamo.reset()
compiled_encode_image = torch.compile(model_neuron.encode_image, backend="neuron")
with torch.no_grad():
    for _ in range(3):
        compiled_encode_image(images_neuron, normalize=True)
torch.neuron.synchronize()

t_compiled_img = bench(lambda: compiled_encode_image(images_neuron, normalize=True))
print(f"  Compiled: {t_compiled_img:.2f}ms")
print(f"  Speedup:  {t_eager_img/t_compiled_img:.1f}×")

# CPU comparison
model_cpu, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
model_cpu = model_cpu.eval()
t_cpu_img = bench_cpu(lambda: model_cpu.encode_image(images, normalize=True), n_iter=50)
print(f"  CPU:      {t_cpu_img:.2f}ms")
print(f"  Neuron compiled vs CPU: {t_cpu_img/t_compiled_img:.1f}×")

# ============================================================
print("\n" + "=" * 70)
print("PART 4: SCALING WITH BATCH SIZE")
print("=" * 70)

print(f"\n  {'Batch':>6} {'Eager(ms)':>10} {'Compiled(ms)':>12} {'CPU(ms)':>8} {'vs Eager':>9} {'vs CPU':>8}")
print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*8} {'-'*9} {'-'*8}")

for bs in [1, 4, 8, 16, 32, 64]:
    imgs = torch.randn(bs, 3, 224, 224, device=DEVICE)
    imgs_cpu = torch.randn(bs, 3, 224, 224)

    t_eager = bench(lambda: model_neuron.encode_image(imgs, normalize=True), n_iter=50)

    torch._dynamo.reset()
    compiled = torch.compile(model_neuron.encode_image, backend="neuron")
    with torch.no_grad():
        for _ in range(3):
            compiled(imgs, normalize=True)
    torch.neuron.synchronize()
    t_comp = bench(lambda: compiled(imgs, normalize=True), n_iter=50)

    t_cpu = bench_cpu(lambda: model_cpu.encode_image(imgs_cpu, normalize=True), n_iter=20)

    print(f"  {bs:>6} {t_eager:>10.2f} {t_comp:>12.2f} {t_cpu:>8.2f} {t_eager/t_comp:>8.1f}× {t_cpu/t_comp:>7.1f}×")

# ============================================================
print("\n" + "=" * 70)
print("PART 5: LARGER MODELS (vision encoder, batch=8)")
print("=" * 70)

for model_name in ["ViT-B-32", "ViT-B-16", "ViT-L-14"]:
    torch._dynamo.reset()
    m, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=None)
    m = m.eval()

    # CPU
    imgs_cpu = torch.randn(8, 3, 224, 224)
    t_cpu = bench_cpu(lambda: m.encode_image(imgs_cpu, normalize=True), n_iter=20)

    # Neuron compiled
    m_n = m.to(DEVICE)
    imgs_n = torch.randn(8, 3, 224, 224, device=DEVICE)
    compiled_enc = torch.compile(m_n.encode_image, backend="neuron")
    with torch.no_grad():
        for _ in range(3):
            compiled_enc(imgs_n, normalize=True)
    torch.neuron.synchronize()
    t_neuron = bench(lambda: compiled_enc(imgs_n, normalize=True), n_iter=50)

    n_params = sum(p.numel() for p in m.visual.parameters()) / 1e6
    print(f"  {model_name:<12} ({n_params:.0f}M vis params)  CPU={t_cpu:.1f}ms  Neuron={t_neuron:.1f}ms  Speedup={t_cpu/t_neuron:.1f}×")

# ============================================================
print("\n" + "=" * 70)
print("PART 6: FIXING THE TEXT ENCODER")
print("=" * 70)
print("\nThe text encoder fails in EAGER mode because:")
print("  1. aten::add with shape [8,77,512] fails NEFF compilation (seq_len=77 tiling issue)")
print("  2. text_global_pool uses x[arange, text.argmax()] — fancy indexing")
print("\nAttempting torch.compile (the compiler may handle these differently)...")

torch._dynamo.reset()
try:
    compiled_txt = torch.compile(model_neuron._encode_text, backend="neuron")
    with torch.no_grad():
        txt_feat_neuron = compiled_txt(text_neuron, normalize=True)
    txt_diff = (txt_feat_cpu - txt_feat_neuron.cpu()).abs().max().item()
    print(f"  torch.compile text encoder: ✓ (max diff={txt_diff:.6f})")

    t_comp_txt = bench(lambda: compiled_txt(text_neuron, normalize=True))
    t_cpu_txt = bench_cpu(lambda: model_cpu.encode_text(text, normalize=True), n_iter=50)
    print(f"  Compiled: {t_comp_txt:.2f}ms")
    print(f"  CPU:      {t_cpu_txt:.2f}ms")
    print(f"  Speedup:  {t_cpu_txt/t_comp_txt:.1f}×")
except Exception as e:
    print(f"  torch.compile text encoder: ✗ FAILED")
    print(f"  Error: {str(e)[:300]}")
    print(f"\n  The text transformer with seq_len=77 hits a Neuron compiler limitation.")
    print(f"  The causal attention mask + non-power-of-2 sequence length causes NEFF creation to fail.")
    print(f"\n  Workarounds:")
    print(f"    1. Pad text to 128 tokens (power-of-2 aligned)")
    print(f"    2. Run text encoder on CPU (it's lightweight compared to vision)")
    print(f"    3. Use a different pooling (cls token at position 0)")

    # Show that text encoder is lightweight compared to vision anyway
    t_cpu_txt = bench_cpu(lambda: model_cpu.encode_text(text, normalize=True), n_iter=100)
    t_cpu_img = bench_cpu(lambda: model_cpu.encode_image(images, normalize=True), n_iter=30)
    print(f"\n  CPU text encoder: {t_cpu_txt:.2f}ms (vs vision: {t_cpu_img:.2f}ms)")
    print(f"  Text is {t_cpu_img/t_cpu_txt:.0f}× cheaper than vision — CPU fallback is acceptable.")

