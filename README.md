# From Data Scientist to Performance Engineer

Making the most of your PyTorch models on Neuron chips.

## Build locally

```bash
pip install -r requirements.txt
jupyter-book build .
open _build/html/index.html
```

## Deploy

Pushes to `main` auto-deploy to GitHub Pages via the workflow in `.github/workflows/deploy.yml`.

Live at: https://pidem.github.io/neuron-performance-book/
