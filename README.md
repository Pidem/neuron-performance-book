# From Data Scientist to Performance Engineer

Making the most of your PyTorch models on Neuron chips.



## Build the book and read locally

```bash
pip install -r requirements.txt
jupyter-book build .
open _build/html/index.html
```

## To deploy trn2 instance reach out to @pidemal
run `workshop-stack.yml` and use the AMI provided by pierre. 

## Book Progress
- [X] Intro 
- [X] Part I
- [ ] Part II
- [ ] Part III
- End to end examples with nki.language and nki.isa?
- Repos of existing kernels
- Multi-core/multichip patterns?


## Deploy

Pushes to `main` auto-deploy to GitHub Pages via the workflow in `.github/workflows/deploy.yml`.

Live at: https://pidem.github.io/neuron-performance-book/
