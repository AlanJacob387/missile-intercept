"""Run a single-threat Phase 0 engagement and report outcome metrics."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: no display on this box

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from mdsim.core.config import load_config, resolve_device  # noqa: E402
from mdsim.core.state import make_initial  # noqa: E402
from mdsim.envs.engine import EngineParams, step  # noqa: E402
from mdsim.eval.metrics import summarize  # noqa: E402

N_ENVS = 4096
NOISE_LEVELS = (0.5, 1.0, 2.0)

# From burnout the threat is airborne for ~239 s at dt=0.05, reaching the ground near
# step 4786; the interceptor round resolves near step 2660. The old 2600 bound predates
# the burnout initial condition and truncated the sweep before the engagement finished,
# so it must stay above the ground-impact step, not above the launch step.
MAX_STEPS = 5000
IDLE_CHECK_EVERY = 100

FIGURE_PATH = ROOT / "artifacts" / "figures" / "pk_vs_noise.png"


def run_once(config, device: str, noise_scale: float) -> dict[str, float | None]:
    """One batched engagement at a given radar noise multiplier."""
    params = EngineParams.from_config(config, engage=True, noise_scale=noise_scale)
    state = make_initial(config, device, dtype=torch.float32)

    for index in range(MAX_STEPS):
        state = step(state, params)
        if (index + 1) % IDLE_CHECK_EVERY == 0:
            # Once nothing is in flight the outcome is settled; the sync costs one
            # readback per 100 steps rather than one per step.
            committed = bool(state.interceptor_committed.all())
            flying = bool(state.interceptor_alive.any())
            if committed and not flying:
                break

    return summarize(state)


def save_plot(noise_levels: tuple[float, ...], pk_values: list[float]) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(6.0, 4.0))
    axes.plot(noise_levels, pk_values, marker="o", linewidth=2.0, color="#2b6cb0")
    axes.set_xlabel("Radar noise multiplier (x nominal sigmas)")
    axes.set_ylabel("Kill probability")
    axes.set_title(f"Phase 0: Pk vs radar noise ({N_ENVS} envs per point)")
    axes.set_ylim(0.0, 1.05)
    axes.grid(True, alpha=0.3)
    for x, y in zip(noise_levels, pk_values):
        axes.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                      ha="center", fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=150)
    plt.close(figure)


def main() -> int:
    config = load_config(ROOT / "configs")
    device = resolve_device(config.sim.device)
    config = replace(config, sim=replace(config.sim, n_envs=N_ENVS))

    results = [run_once(config, device, scale) for scale in NOISE_LEVELS]
    pk_values = [r["pk"] for r in results]

    header = (
        f"{'noise x':>8}{'Pk':>10}{'leakage':>10}{'committed':>12}"
        f"{'kills':>10}{'cmt/kill*':>11}{'tracks held':>14}"
    )
    print(f"\nPhase 0 kill probability vs radar noise  ({N_ENVS} envs, device={device})")
    print(header)
    print("-" * len(header))
    for scale, result in zip(NOISE_LEVELS, results):
        ratio = result["committed_per_kill_batch_level"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.2f}"
        print(
            f"{scale:>8.1f}{result['pk']:>10.4f}{result['leakage']:>10.4f}"
            f"{result['interceptors_committed']:>12,.0f}"
            f"{result['kills']:>10,.0f}"
            f"{ratio_text:>11}"
            f"{result['tracks_held']:>14.3f}"
        )
    print("-" * len(header))
    print(
        "* cmt/kill is a BATCH-LEVEL ratio (total committed / total killed), not a "
        "salvo size. With one interceptor per env it is just 1/Pk."
    )

    save_plot(NOISE_LEVELS, pk_values)
    print(f"wrote {FIGURE_PATH.relative_to(ROOT)}")

    monotonic = all(a >= b for a, b in zip(pk_values, pk_values[1:]))
    if not monotonic:
        print(
            "WARNING: Pk does not degrade monotonically with radar noise. Either the "
            "noise is not reaching the tracker or the engagement is insensitive to it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
