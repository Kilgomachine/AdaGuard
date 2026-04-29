"""Emit matplotlib figures for the AdaGuard TDSC paper.

Reads JSONs from data/paper_data/ and writes PDF+PNG figures to
paper_artifacts/figures/. Each figure is a standalone PDF meant
for \\includegraphics{} from Overleaf.

Design choices:
  * One figure per file (no subplots that force a shared caption).
  * Greyscale-friendly hatches in addition to colours (TDSC prints
    some pages in B/W).
  * PDF is the deliverable; PNG is written as a review-friendly
    twin so you can eyeball diffs without a viewer.

Usage:
  python scripts/paper/build_figures.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paper._data import (  # noqa: E402
    REPO_ROOT, DEFENCE_LABELS, DEFENCE_ORDER,
    ATTACK_LABELS,
    load_defence_sweep, load_client_diversity,
)

OUT_DIR = REPO_ROOT / "paper_artifacts" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colour-blind safe palette (Wong 2011) + hatches for B/W print.
_DEF_STYLE = {
    "none":      {"color": "#999999", "hatch": "",   "label": DEFENCE_LABELS["none"]},
    "fhe":       {"color": "#0072B2", "hatch": "//", "label": DEFENCE_LABELS["fhe"]},
    "maskcrypt": {"color": "#E69F00", "hatch": "xx", "label": DEFENCE_LABELS["maskcrypt"]},
    "fisher":    {"color": "#009E73", "hatch": "..", "label": DEFENCE_LABELS["fisher"]},
}

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,  # editable text for Overleaf compile
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def _savefig(fig, name: str):
    fig.savefig(OUT_DIR / f"{name}.pdf")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {OUT_DIR.relative_to(REPO_ROOT) / name}.{{pdf,png}}")


# --------------------------------------------------------------------------
# Helpers to pull a metric grid out of the sweep
# --------------------------------------------------------------------------

def _attack_key_for_gradinv(sweep):
    """Prefer breaching-backed key if present; else the homegrown one."""
    for d in DEFENCE_ORDER:
        if (d, "gradinversion_breaching", 1) in sweep:
            return "gradinversion_breaching"
    return "gradinversion"


def _attacks_ordered(sweep):
    return [_attack_key_for_gradinv(sweep), "ggcdm", "gi_nas"]


def _metric_grid(sweep, attacks, metric):
    """Return np.array of shape (n_attack, n_defence) with NaN for missing."""
    g = np.full((len(attacks), len(DEFENCE_ORDER)), np.nan)
    for i, a in enumerate(attacks):
        for j, d in enumerate(DEFENCE_ORDER):
            m = sweep.get((d, a, 1))
            if not m:
                continue
            if metric == "asr":
                v = (m.get("label_recovery") or {}).get("asr")
            else:
                v = m.get(metric)
            if v is not None:
                g[i, j] = float(v)
    return g


# --------------------------------------------------------------------------
# Figure 1: PSNR bars (grouped by attack)
# --------------------------------------------------------------------------

def fig_psnr_bars(sweep):
    attacks = _attacks_ordered(sweep)
    g = _metric_grid(sweep, attacks, "psnr")

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    n_d = len(DEFENCE_ORDER)
    width = 0.2
    x = np.arange(len(attacks))
    for j, d in enumerate(DEFENCE_ORDER):
        vals = g[:, j]
        ax.bar(
            x + (j - (n_d - 1) / 2) * width,
            np.where(np.isnan(vals), 0.0, vals),
            width=width,
            color=_DEF_STYLE[d]["color"],
            hatch=_DEF_STYLE[d]["hatch"],
            edgecolor="black",
            linewidth=0.6,
            label=_DEF_STYLE[d]["label"],
        )
        for xi, v in zip(x + (j - (n_d - 1) / 2) * width, vals):
            if np.isnan(v):
                continue
            ax.text(xi, v + 0.15, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=7)

    ax.axhline(10.0, color="red", linestyle=":", linewidth=1.0,
               label="recognizable threshold ($\\approx$10 dB)")
    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_LABELS.get(a, a) for a in attacks])
    ax.set_ylabel("Reconstruction PSNR (dB) — lower is better")
    ax.set_ylim(0, max(np.nanmax(g) + 6, 24))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=8, frameon=False, ncol=5)
    _savefig(fig, "fig_psnr_bars")


# --------------------------------------------------------------------------
# Figure 2: LPIPS bars (grouped by attack)
# --------------------------------------------------------------------------

def fig_lpips_bars(sweep):
    attacks = _attacks_ordered(sweep)
    g = _metric_grid(sweep, attacks, "lpips")

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    n_d = len(DEFENCE_ORDER)
    width = 0.2
    x = np.arange(len(attacks))
    for j, d in enumerate(DEFENCE_ORDER):
        vals = g[:, j]
        ax.bar(
            x + (j - (n_d - 1) / 2) * width,
            np.where(np.isnan(vals), 0.0, vals),
            width=width,
            color=_DEF_STYLE[d]["color"],
            hatch=_DEF_STYLE[d]["hatch"],
            edgecolor="black",
            linewidth=0.6,
            label=_DEF_STYLE[d]["label"],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_LABELS.get(a, a) for a in attacks])
    ax.set_ylabel("LPIPS (AlexNet) — higher is further from original")
    ax.set_ylim(0, max(np.nanmax(g) + 0.1, 0.85))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=8, frameon=False, ncol=4)
    _savefig(fig, "fig_lpips_bars")


# --------------------------------------------------------------------------
# Figure 3: Label-recovery ASR collapse
# --------------------------------------------------------------------------

def fig_asr_collapse(sweep):
    attacks = _attacks_ordered(sweep)
    g = _metric_grid(sweep, attacks, "asr")

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    n_d = len(DEFENCE_ORDER)
    width = 0.2
    x = np.arange(len(attacks))
    for j, d in enumerate(DEFENCE_ORDER):
        vals = g[:, j]
        ax.bar(
            x + (j - (n_d - 1) / 2) * width,
            np.where(np.isnan(vals), 0.0, vals),
            width=width,
            color=_DEF_STYLE[d]["color"],
            hatch=_DEF_STYLE[d]["hatch"],
            edgecolor="black",
            linewidth=0.6,
            label=_DEF_STYLE[d]["label"],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_LABELS.get(a, a) for a in attacks])
    ax.set_ylabel("Label-recovery ASR (B=1)")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=8, frameon=False, ncol=4)
    _savefig(fig, "fig_asr_collapse")


# --------------------------------------------------------------------------
# Figure 4: client-diversity histogram
# --------------------------------------------------------------------------

def fig_client_diversity(rows):
    if not rows:
        print("skip fig_client_diversity (no data)")
        return
    counter = Counter(r["n_unique"] for r in rows)
    xs = sorted(counter)
    ys = [counter[k] for k in xs]

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.bar(xs, ys, color="#56B4E9", edgecolor="black", linewidth=0.6)
    for xi, yi in zip(xs, ys):
        ax.text(xi, yi + 0.3, str(yi), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)
    ax.set_xlabel("# unique classes per client (of 10 on CIFAR-10)")
    ax.set_ylabel("# clients (top-diversity sample)")
    ax.set_title(
        "Client label diversity — round 249 "
        f"(n={len(rows)} most-diverse clients)",
        fontsize=10,
    )
    _savefig(fig, "fig_client_diversity")


# --------------------------------------------------------------------------
# Figure 5: B=1 vs B=4 diverse on undefended (GradInv, GI-NAS)
# --------------------------------------------------------------------------

def fig_b1_vs_b4_diverse(sweep):
    import json
    diverse_dir = REPO_ROOT / "data" / "paper_data" / "diverse"
    b4 = {}
    for name, attack in (
        ("gradinversion_b4_diverse.json", "gradinversion"),
        ("gi_nas_b4_diverse.json", "gi_nas"),
    ):
        p = diverse_dir / name
        if p.exists():
            with p.open() as f:
                b4[attack] = json.load(f)
    if not b4:
        print("skip fig_b1_vs_b4_diverse (no diverse data)")
        return

    attacks = list(b4)
    psnr_b1 = [sweep.get(("none", a, 1), {}).get("psnr", np.nan)
               for a in attacks]
    psnr_b4 = [b4[a].get("psnr", np.nan) for a in attacks]
    asr_b1 = [(sweep.get(("none", a, 1), {}).get("label_recovery") or {})
              .get("asr", np.nan) for a in attacks]
    asr_b4 = [(b4[a].get("label_recovery") or {}).get("asr", np.nan)
              for a in attacks]

    fig, (axp, axa) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x = np.arange(len(attacks))
    axp.bar(x - 0.2, psnr_b1, 0.4, color="#999999", label="B=1",
            edgecolor="black", linewidth=0.6)
    axp.bar(x + 0.2, psnr_b4, 0.4, color="#D55E00", label="B=4 (diverse)",
            edgecolor="black", linewidth=0.6, hatch="//")
    for xi, v in zip(x - 0.2, psnr_b1):
        if not np.isnan(v):
            axp.text(xi, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + 0.2, psnr_b4):
        if not np.isnan(v):
            axp.text(xi, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    axp.set_xticks(x)
    axp.set_xticklabels([ATTACK_LABELS.get(a, a) for a in attacks])
    axp.set_ylabel("Undefended PSNR (dB)")
    axp.set_ylim(0, 26)
    axp.legend(frameon=False, fontsize=8, loc="upper right")

    axa.bar(x - 0.2, asr_b1, 0.4, color="#999999", label="B=1",
            edgecolor="black", linewidth=0.6)
    axa.bar(x + 0.2, asr_b4, 0.4, color="#D55E00", label="B=4 (diverse)",
            edgecolor="black", linewidth=0.6, hatch="//")
    axa.set_xticks(x)
    axa.set_xticklabels([ATTACK_LABELS.get(a, a) for a in attacks])
    axa.set_ylabel("Label-recovery ASR")
    axa.set_ylim(0, 1.25)
    axa.legend(frameon=False, fontsize=8, loc="upper right")
    for xi, v in zip(x - 0.2, asr_b1):
        if not np.isnan(v):
            axa.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + 0.2, asr_b4):
        if not np.isnan(v):
            axa.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    _savefig(fig, "fig_b1_vs_b4_diverse")


# --------------------------------------------------------------------------
# Figure 6: LeakScore trajectory across training rounds (added 2026-04-29
# to address the reviewer ask "show that T1=0.3 and T2=0.7 are sensible
# choices across all 250 rounds, not just round 249"). Uses per-round
# combined_leakscore values that the simulator already saves into the
# Phase-1 training JSONs at data/paper_data/training/.
# --------------------------------------------------------------------------

def fig_leakscore_trajectory():
    """Mean ± std across seeds of the V4 (Fisher) per-round LeakScore.

    Pulls combined_leakscore from
    data/paper_data/training/fisher_seed{42,123,456}_300clients.json,
    averages across seeds, and overlays T1 and T2 thresholds plus the
    round-249 replay-snapshot marker.
    """
    import json
    train_dir = REPO_ROOT / "data" / "paper_data" / "training"
    seeds = [42, 123, 456]
    trajectories = []
    for seed in seeds:
        p = train_dir / f"fisher_seed{seed}_300clients.json"
        if not p.exists():
            print(f"skip fig_leakscore_trajectory: missing {p.name}")
            return
        with p.open() as f:
            obj = json.load(f)
        rounds = obj.get("rounds", [])
        ls = [r.get("combined_leakscore") for r in rounds]
        if not ls or any(v is None for v in ls):
            print(f"skip fig_leakscore_trajectory: incomplete LeakScore in {p.name}")
            return
        trajectories.append(np.asarray(ls, dtype=np.float64))

    # Align lengths defensively in case a seed crashed early.
    min_len = min(len(t) for t in trajectories)
    arr = np.stack([t[:min_len] for t in trajectories], axis=0)
    rounds_x = np.arange(1, min_len + 1)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=0)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    # Per-seed light traces in the background (visual honesty signal).
    for i, seed in enumerate(seeds):
        ax.plot(rounds_x, arr[i], color="#009E73", alpha=0.25, linewidth=0.8,
                label=f"seed {seed}" if i == 0 else None)

    # Mean trajectory + std band.
    ax.plot(rounds_x, mean, color="#009E73", linewidth=1.6,
            label="V4 LeakScore mean (n=3 seeds)")
    ax.fill_between(rounds_x, mean - std, mean + std,
                    color="#009E73", alpha=0.20, linewidth=0,
                    label="$\\pm$1 sample-std band")

    # Threshold horizontal lines.
    ax.axhline(0.3, color="#0072B2", linestyle="--", linewidth=1.0,
               label="$T_1 = 0.3$ (encryption fires)")
    ax.axhline(0.7, color="#D55E00", linestyle="--", linewidth=1.0,
               label="$T_2 = 0.7$ (gradient accum fires)")

    # Replay-snapshot marker at round 249.
    if min_len >= 249:
        ax.axvline(249, color="black", linestyle=":", linewidth=0.9, alpha=0.5)
        ls_249 = mean[248]
        ax.annotate(
            f"round 249\n(replay snapshot)\nLeakScore $\\approx$ {ls_249:.3f}",
            xy=(249, ls_249), xytext=(180, ls_249 + 0.18),
            fontsize=7, ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-", color="black",
                            linewidth=0.6, alpha=0.5),
        )

    ax.set_xlabel("Communication round")
    ax.set_ylabel("LeakScore (combined)")
    ax.set_xlim(0, min_len + 5)
    ax.set_ylim(0, max(1.0, float(arr.max()) + 0.05))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              fontsize=7.5, frameon=False, ncol=3)

    _savefig(fig, "fig_leakscore_trajectory")


def main():
    sweep = load_defence_sweep()
    diversity = load_client_diversity()

    fig_psnr_bars(sweep)
    fig_lpips_bars(sweep)
    fig_asr_collapse(sweep)
    fig_client_diversity(diversity)
    fig_b1_vs_b4_diverse(sweep)
    fig_leakscore_trajectory()


if __name__ == "__main__":
    main()
