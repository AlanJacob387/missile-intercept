"""Sweep inventory against raid size and emit the saturation surface."""

from __future__ import annotations

import sys
import time
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

# Slot counts. Every cell carries these tensor shapes; the sweep varies how many
# slots are active, so per-step cost is flat across the grid and the cells are
# directly comparable.
THREAT_SLOTS = 12
INTERCEPTOR_SLOTS = 12

THREAT_COUNTS = (2, 4, 8, 12)
INVENTORIES = (0, 1, 2, 4, 8, 12)

# The headline 1D slice. Mid-grid: large enough that inventory has to be rationed,
# small enough that a full magazine can still cover the raid.
SLICE_THREAT_COUNT = 8

# Threats fly from burnout to ground over roughly 4,150 steps at dt=0.05, and the
# engagement resolves shortly before that. The bound must clear ground impact, not
# merely the launch step: a short horizon reports threats as neither killed nor
# leaked and silently reads as zero leakage.
MAX_STEPS = 5000
IDLE_CHECK_EVERY = 100

FIGURE_DIR = ROOT / "docs" / "figures"
SURFACE_PATH = FIGURE_DIR / "saturation_surface.png"
SLICE_PATH = FIGURE_DIR / "leakage_vs_inventory.png"


def run_cell(
    config, params: EngineParams, device: str, threat_count: int, inventory: int
) -> dict[str, float | None]:
    """One batched raid at a given raid size and magazine depth."""
    state = make_initial(
        config,
        device,
        dtype=torch.float32,
        n_active_threats=threat_count,
        inventory=inventory,
    )

    for index in range(MAX_STEPS):
        state = step(state, params)
        if (index + 1) % IDLE_CHECK_EVERY == 0:
            # Every threat is down once it is killed, leaked, or grounded, and the
            # outcome is settled from there. Interceptor state is the wrong exit
            # test here: at inventory 0 nothing is ever committed.
            if not bool(state.threat_alive.any()):
                break

    result = summarize(state)
    result["steps_run"] = float(index + 1)
    return result


def print_surface(grid: dict[tuple[int, int], dict[str, float | None]]) -> None:
    header = f"{'threats':>9}" + "".join(f"{inv:>9}" for inv in INVENTORIES)
    print("\nLeakage surface  (fraction of active threats reaching a city)")
    print(f"{'':>9}{'inventory':>{9 * len(INVENTORIES)}}")
    print(header)
    print("-" * len(header))
    for count in THREAT_COUNTS:
        row = f"{count:>9}"
        for inventory in INVENTORIES:
            row += f"{grid[(count, inventory)]['leakage']:>9.4f}"
        print(row)
    print("-" * len(header))


def print_slice(grid: dict[tuple[int, int], dict[str, float | None]]) -> None:
    header = (
        f"{'inventory':>10}{'leakage':>10}{'kills':>10}{'committed':>12}"
        f"{'unfired':>10}{'value lost':>12}{'steps':>8}"
    )
    print(f"\nHeadline slice  (raid size {SLICE_THREAT_COUNT}, {N_ENVS} envs per point)")
    print(header)
    print("-" * len(header))
    for inventory in INVENTORIES:
        cell = grid[(SLICE_THREAT_COUNT, inventory)]
        print(
            f"{inventory:>10}{cell['leakage']:>10.4f}{cell['kills']:>10,.0f}"
            f"{cell['interceptors_committed']:>12,.0f}"
            f"{cell['inventory_remaining']:>10.2f}"
            f"{cell['value_destroyed']:>12.3f}{cell['steps_run']:>8,.0f}"
        )
    print("-" * len(header))


def save_surface(grid: dict[tuple[int, int], dict[str, float | None]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    values = [
        [grid[(count, inv)]["leakage"] for inv in INVENTORIES] for count in THREAT_COUNTS
    ]

    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    image = axes.imshow(
        values, cmap="magma_r", vmin=0.0, vmax=1.0, aspect="auto", origin="upper"
    )
    axes.set_xticks(range(len(INVENTORIES)), [str(i) for i in INVENTORIES])
    axes.set_yticks(range(len(THREAT_COUNTS)), [str(c) for c in THREAT_COUNTS])
    axes.set_xlabel("Interceptor inventory")
    axes.set_ylabel("Raid size (active threats)")
    axes.set_title(f"Phase 1: leakage vs inventory and raid size ({N_ENVS} envs per cell)")

    for row, count in enumerate(THREAT_COUNTS):
        for col, inventory in enumerate(INVENTORIES):
            leak = grid[(count, inventory)]["leakage"]
            axes.text(
                col, row, f"{leak:.2f}", ha="center", va="center", fontsize=9,
                color="white" if leak > 0.55 else "black",
            )

    figure.colorbar(image, ax=axes, label="Leakage fraction")
    figure.tight_layout()
    figure.savefig(SURFACE_PATH, dpi=150)
    plt.close(figure)


def save_slice(grid: dict[tuple[int, int], dict[str, float | None]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    leakages = [grid[(SLICE_THREAT_COUNT, inv)]["leakage"] for inv in INVENTORIES]

    figure, axes = plt.subplots(figsize=(6.0, 4.0))
    axes.plot(INVENTORIES, leakages, marker="o", linewidth=2.0, color="#b02b2b")
    axes.set_xlabel("Interceptor inventory")
    axes.set_ylabel("Leakage fraction")
    axes.set_title(
        f"Phase 1: leakage vs inventory, raid size {SLICE_THREAT_COUNT} "
        f"({N_ENVS} envs per point)"
    )
    axes.set_ylim(-0.05, 1.05)
    axes.grid(True, alpha=0.3)
    for x, y in zip(INVENTORIES, leakages):
        axes.annotate(
            f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(SLICE_PATH, dpi=150)
    plt.close(figure)


def main() -> int:
    config = load_config(ROOT / "configs")
    device = resolve_device(config.sim.device)
    config = replace(
        config,
        sim=replace(config.sim, n_envs=N_ENVS),
        scenario=replace(
            config.scenario,
            n_threats=THREAT_SLOTS,
            n_interceptors=INTERCEPTOR_SLOTS,
        ),
    )
    params = EngineParams.from_config(config)

    grid: dict[tuple[int, int], dict[str, float | None]] = {}
    started = time.perf_counter()
    for count in THREAT_COUNTS:
        for inventory in INVENTORIES:
            cell_started = time.perf_counter()
            grid[(count, inventory)] = run_cell(config, params, device, count, inventory)
            print(
                f"  threats={count:>3} inventory={inventory:>3}  "
                f"leakage={grid[(count, inventory)]['leakage']:.4f}  "
                f"({time.perf_counter() - cell_started:.0f} s)"
            )
    print(f"\nswept {len(grid)} cells in {time.perf_counter() - started:.0f} s")

    print_surface(grid)
    print_slice(grid)

    save_surface(grid)
    save_slice(grid)
    print(f"\nwrote {SURFACE_PATH.relative_to(ROOT)}")
    print(f"wrote {SLICE_PATH.relative_to(ROOT)}")

    slice_values = [grid[(SLICE_THREAT_COUNT, inv)]["leakage"] for inv in INVENTORIES]
    if max(slice_values) - min(slice_values) < 1e-6:
        print(
            f"\nWARNING: leakage is flat at {slice_values[0]:.4f} across the whole "
            "inventory axis. Adding interceptors changes nothing, so the engagement "
            "is not resolving: check assignment, launch, and inventory before "
            "reading the curve."
        )
    elif not all(a >= b for a, b in zip(slice_values, slice_values[1:])):
        print(
            "\nWARNING: leakage does not fall monotonically with inventory. More "
            "rounds should never let more threats through."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
