# From Data Scientist to Performance Engineer

*Making the most of your PyTorch models on Neuron chips*

## Who is this for?

You're a scientist or ML engineer working with biological foundation models. You train protein language models, molecular property predictors, or clinical NLP systems. You're comfortable with PyTorch and Python, but you've never thought about *why* your training takes the time it does, or what's actually happening on the hardware underneath.

This book is for ML engineers who want to close that gap on AWS Neuron chips. The examples use examples from the Healthcare & Life Sciences space, but the concepts apply to any deep learning workload: LLMs, vision models, diffusion, multimodal. 

No hardware background required. 

## What you'll learn

Every chapter comes with code you can run on a Trainium instance. You start by profiling a model you already know, then peel back layers: how the compiler organizes your ops, how data moves through memory, how instructions land on hardware. At each step, understanding the hardware a little better lets you write slightly better software — until you're writing kernels that extract 80% of the chip's theoretical peak.

| Level | What you write | What changes | Typical MFU |
|-------|---------------|--------------|-------------|
| **Eager** | `model.to("neuron")` | use `device='neuron'` | ~30% |
| **torch.compile** | `torch.compile(model, backend="neuron")` | add one line; compiler fuses and tiles | 50–60% |
| **Custom kernel (NKI)** | `@nki.jit` attention function | You control engines, tiling, pipelining | 80%+ |

*MFU (Model FLOPs Utilization) measures what fraction of the chip's compute capacity your model actually uses, higher is better. A "kernel" here means a small program that runs directly on the accelerator hardware. NKI (Neuron Kernel Interface) is the Python DSL for writing them.*

Each level requires more knowledge but delivers more performance. This book takes you through all three.

## How to use this book

- **Parts I–II** — PyTorch internals and Neuron hardware (start here)
- **Part III** — Measuring performance: roofline models, the profiler
- **Parts IV–V** — Optimizing: number formats, then custom NKI kernels
- **Part VI** — Scaling out and production

## Hardware setup

All code in this book has been tested on a single **trn2.3xlarge** instance (1 Trainium2 chip, 32GB HBM).

```{figure} assets/trn2_chip.png
:alt: Trainium2 chip
:width: 400px
:align: center

The AWS Trainium2 chip — your workstation for this book.
```
### Provisioning a trn2.3xlarge via Capacity Blocks

Capacity Blocks let you reserve Trainium instances for a fixed duration at a predictable cost (~$2.20/hr). At the time of writing, `trn2.3xlarge` Capacity Blocks are available in:

- Melbourne (`ap-southeast-4`)
- São Paulo (`sa-east-1`)

**Step 1: Find available blocks**

```bash
aws ec2 describe-capacity-block-offerings \
  --instance-type trn2.3xlarge \
  --capacity-duration-hours 24 \
  --instance-count 1 \
  --region ap-southeast-4
```

**Step 2: Purchase a block**

```bash
aws ec2 purchase-capacity-block \
  --capacity-block-offering-id cb-XXXXXXXXXXXXXXXXX \
  --instance-platform Linux/UNIX \
  --region ap-southeast-4
```

**Step 3: Launch your instance**

Once the block's start time arrives, launch an instance into it using the [Neuron Deep Learning AMI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/index.html):

```bash
aws ec2 run-instances \
  --instance-type trn2.3xlarge \
  --image-id ami-XXXXXXXXXXXXXXXXX \
  --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=cr-XXXXXXXXX}" \
  --key-name your-key \
  --region ap-southeast-4
```

**Step 4: Connect and start JupyterLab**

```bash
ssh -i your-key.pem ubuntu@<instance-ip> -L 8888:localhost:8888
# On the instance:
source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate
jupyter lab --no-browser --port=8888
```

Then open `http://localhost:8888` in your browser.

```{admonition} Cost estimate
:class: note
A 24-hour Capacity Block for `trn2.3xlarge` costs approximately \$54. A 10-hour block costs approximately \$22. This is enough time to work through several chapters.
```

## Table of contents

```{tableofcontents}
```
