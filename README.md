# Missile Defense Simulation

A vectorized, GPU-resident 3D interception sandbox. This stage covers the physics core:
batched tensor dynamics over thousands of parallel environments, fixed-timestep
integration, and batch equivalence verified against an independent reference
implementation. Built as a student research project on public, approximate data.

## Stack

PyTorch with the Intel XPU backend (`torch.xpu`), Windows-native. State tensors carry a
leading `N_envs` dimension and every layer is written batched, so one step advances all
environments on device. `torch.compile` is supported on Windows XPU from PyTorch 2.7
(not yet enabled here). The backend resolver falls back `xpu -> cuda -> cpu` and
reports which one it selected.

## Install

Requires Python 3.11 or newer.

```
python -m venv .venv
.venv/Scripts/activate      # Windows; use .venv/bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/xpu
pip install -e .
```

The XPU wheel must come from PyTorch's own index. The default PyPI wheel is CPU-only on
Windows, and `torch.xpu.is_available()` will report False.

## Test

```
pytest
python scripts/validate_batching.py
```

The batch-equivalence gate compares the tensor engine against an independent NumPy
reference implementation at machine precision.

`test_engine_matches_across_devices` compares CPU against the resolved accelerator; on
a machine with no accelerator it skips, since there is nothing to compare against.

---

*Educational simulation. Not a targeting aid; models no classified performance. All
weapon statistics are published public estimates, labeled as a model.*
