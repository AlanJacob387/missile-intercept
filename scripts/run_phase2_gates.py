"""Phase 2 gates: IMM vs single-KF, Hungarian vs greedy, salvo vs single-shot.

Prints the three comparison tables and saves one figure per comparison to
docs/figures/. Numbers are reported as measured; if a gate does not show the
expected direction, it prints a mechanism note to diagnose rather than a forced pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from mdsim.core.config import load_config  # noqa: E402
from mdsim.eval.compare import (  # noqa: E402
    hungarian_vs_greedy_damage,
    imm_vs_kf_tracking_error,
    salvo_vs_single_shot_leakage,
)

FIGURE_DIR = ROOT / "docs" / "figures"


def _bar(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(5.0, 4.0))
    colors = ["#7a7a7a", "#b02b2b"]
    bars = axes.bar(labels, values, color=colors[: len(labels)])
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    for bar, value in zip(bars, values):
        axes.annotate(
            f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> int:
    config = load_config(ROOT / "configs")

    print("Gate 1: IMM vs single-KF track error (maneuvering threat, matched seeds)")
    gate1 = imm_vs_kf_tracking_error(config)
    print(
        f"  kf error={gate1['kf_error_m']:.3f} m   imm error={gate1['imm_error_m']:.3f} m   "
        f"margin={gate1['margin_m']:.3f} m"
    )
    print("  PASS" if gate1["margin_m"] > 0.0 else "  FAIL: IMM did not beat single-KF")
    _bar(
        FIGURE_DIR / "gate1_imm_vs_kf.png",
        "Gate 1: mean held-track position error (weave, iskander_m)",
        ["single-KF", "IMM"], [gate1["kf_error_m"], gate1["imm_error_m"]], "position error (m)",
    )

    print("\nGate 2: Hungarian vs greedy value-weighted damage (multi-battery, matched seeds)")
    gate2 = hungarian_vs_greedy_damage(ROOT / "configs")
    print(
        f"  greedy damage={gate2['greedy_damage']:.4f}   hungarian damage={gate2['hungarian_damage']:.4f}   "
        f"margin={gate2['margin']:.4f}"
    )
    if gate2["margin"] > 1e-9:
        print("  PASS (strict win)")
    elif gate2["margin"] >= -1e-9:
        print(
            "  TIE, not the expected direction under multi-battery geometry -- diagnose "
            "before touching any parameter. Check whether this raid's threats are all "
            "reachable by every battery (no real coupling for Hungarian to exploit) or "
            "whether greedy's slot-index order happens to already respect battery "
            "boundaries for this particular seed."
        )
    else:
        print("  FAIL: hungarian did worse than greedy -- this should not be possible; diagnose, do not tune")
    _bar(
        FIGURE_DIR / "gate2_hungarian_vs_greedy.png",
        "Gate 2: value destroyed (lower is better), constructed 2-battery contested case",
        ["greedy", "hungarian"], [gate2["greedy_damage"], gate2["hungarian_damage"]], "value destroyed",
    )

    print("\nGate 3: salvo(2) vs single-shot leakage (matched inventory-per-threat, aim dispersion on)")
    gate3 = salvo_vs_single_shot_leakage(config)
    print(
        f"  single-shot leakage={gate3['single_shot_leakage']:.4f}   "
        f"salvo leakage={gate3['salvo_leakage']:.4f}   delta={gate3['delta']:.4f}"
    )
    if gate3["delta"] > 1e-9:
        print("  PASS")
    elif gate3["delta"] >= -1e-9:
        print(
            "  TIE, not the expected direction -- diagnose before touching any parameter. "
            "aim_dispersion_rad should have decorrelated the two rounds' trajectories; "
            "check per-round closest-approach points actually differ now (see "
            "tests/test_dispersion.py) rather than raising the dispersion magnitude to "
            "force a result."
        )
    else:
        print("  FAIL: salvo did worse than single-shot -- diagnose, do not tune")
    _bar(
        FIGURE_DIR / "gate3_salvo_vs_single_shot.png",
        "Gate 3: leakage, single-shot vs salvo(2) (2 threats, 6 rounds, dispersed aim)",
        ["single-shot", "salvo(2)"], [gate3["single_shot_leakage"], gate3["salvo_leakage"]], "leakage fraction",
    )

    print(f"\nwrote figures to {FIGURE_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
