# From Data Scientist to Performance Engineer

*Making the most of your PyTorch models on Neuron chips*

## Who is this for?

You're a scientist or ML engineer working with biological foundation models. You train protein language models, molecular property predictors, or clinical NLP systems. You're comfortable with PyTorch and Python — but you've never thought about *why* your training takes the time it does, or what's actually happening on the hardware underneath.

You want to make sure you're using your Neuron chip to its full potential. This book teaches you how.

## What you'll learn

This book takes you from "I can call `model(x)`" to "I understand why my kernel is memory-bound on NeuronCore-v3 and I know how to fix it." We do this incrementally — each chapter peels back one layer of abstraction, and each chapter ends with a question that the next one answers.

```{admonition} The optimization ladder
:class: tip
**Eager** (get it running) → **torch.compile** (graph-level optimization) → **NKI** (kernel-level control)
```

We follow **ESM-2** (a 650M-parameter protein language model) from a black-box forward pass all the way down to hand-written NKI kernels. By the end, you'll have a complete mental model of what happens between `model(x)` and the transistors that execute it.

## How to use this book

- **Read linearly** — chapters build on each other
- **Run the notebooks** — every concept has runnable code on a `trn2.3xlarge`
- **Skip to your level** — if you already know PyTorch internals, start at Part II

## Hardware setup

All code in this book runs on a single **trn2.3xlarge** instance (1 Trainium2 chip, 32GB HBM). No GPU required — the entire book runs on Neuron from Chapter 1.

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
