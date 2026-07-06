"""Steps/sec vs N_envs on each backend: evidence the step is actually batched."""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from mdsim.core.config import load_config, resolve_device  # noqa: E402
from mdsim.core.state import make_initial  # noqa: E402
from mdsim.envs.engine import EngineParams, step  # noqa: E402

# Extends past the configured n_envs on purpose: the Phase 0 step is a handful of
# elementwise kernels, so the accelerator stays launch-bound until the batch is large
# enough to cover the dispatch. Stopping at 4096 would report a loss and hide that.
BATCH_SIZES = (1, 64, 1024, 4096, 16384, 65536)
WARMUP_STEPS = 20
TIMED_STEPS = 200


def _synchronize(device: str) -> None:
    """Device work is queued asynchronously; timing without this measures dispatch."""
    backend = getattr(torch, device, None)
    if backend is not None and hasattr(backend, "synchronize"):
        backend.synchronize()


def measure(config, device: str, n_envs: int) -> float:
    """Steps per second for one batch size. One step advances all n_envs."""
    config = replace(config, sim=replace(config.sim, n_envs=n_envs))
    # The full loop, not physics alone: sensing and guidance are most of the step's
    # work, so timing physics only would flatter the accelerator crossover.
    params = EngineParams.from_config(config)
    state = make_initial(config, device)

    for _ in range(WARMUP_STEPS):
        state = step(state, params)
    _synchronize(device)

    start = time.perf_counter()
    for _ in range(TIMED_STEPS):
        state = step(state, params)
    _synchronize(device)
    return TIMED_STEPS / (time.perf_counter() - start)


def main() -> int:
    config = load_config(ROOT / "configs")
    selected = resolve_device(config.sim.device)

    devices = ["cpu"] if selected == "cpu" else ["cpu", selected]
    results: dict[str, dict[int, float]] = {}

    for device in devices:
        results[device] = {n: measure(config, device, n) for n in BATCH_SIZES}

    header = f"{'N_envs':>8}" + "".join(f"{d + ' steps/s':>18}" for d in devices)
    header += f"{'env-steps/s (' + devices[-1] + ')':>26}"
    print(f"\n{header}")
    print("-" * len(header))
    for n in BATCH_SIZES:
        row = f"{n:>8}"
        for device in devices:
            row += f"{results[device][n]:>18,.0f}"
        row += f"{n * results[devices[-1]][n]:>26,.0f}"
        print(row)
    print("-" * len(header))

    if len(devices) == 1:
        print("Only CPU available; no accelerator comparison to make.")
        return 0

    accelerator = devices[-1]
    ratios = {
        n: (n * results[accelerator][n]) / (n * results["cpu"][n]) for n in BATCH_SIZES
    }
    crossover = next((n for n in BATCH_SIZES if ratios[n] > 1.0), None)
    largest = BATCH_SIZES[-1]

    print(f"At N={largest}: {accelerator} is {ratios[largest]:.2f}x CPU on env-steps/s.")
    if crossover is None:
        print(
            f"WARNING: {accelerator} never beats CPU across {BATCH_SIZES}. The step is "
            "too small to cover kernel launch overhead, so the batch is not paying "
            "for the device."
        )
    else:
        print(
            f"{accelerator} overtakes CPU at N={crossover}; below that the step is "
            "launch-bound, and its steps/s barely moves with N."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
