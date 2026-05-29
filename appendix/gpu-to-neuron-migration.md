# Migrating from GPU to Neuron

*The 5-minute migration and what comes after.*

- Change `cuda` → `neuron`, run, check fallback ops
- Common blockers: complex dtypes, unsupported ops, CUDA-specific code paths
- The optimization path: eager → profile → compile → NKI where needed

```{admonition} TODO
:class: warning
Draft content.
```
