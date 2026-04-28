"""Shared JSON loaders for paper-artifact generation.

Reads from data/paper_data/ (committed at repo root) and returns
structured dicts/lists for tables and figures to consume.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "paper_data"


DEFENCE_LABELS = {
    "none": "V1 (None)",
    "fhe": "V2 (FHE)",
    "maskcrypt": "V4 (MaskCrypt)",
    "fisher": "V6 (AdaGuard-Fisher)",
}
DEFENCE_ORDER = ["none", "fhe", "maskcrypt", "fisher"]

ATTACK_LABELS = {
    "gradinversion": "GradInversion",
    "gradinversion_breaching": "GradInversion (Yin)",
    "ggcdm": "GGCDM",
    "gi_nas": "GI-NAS",
}
ATTACK_ORDER = ["gradinversion", "ggcdm", "gi_nas"]


_FILENAME_RE = re.compile(
    r"^(?P<defence>none|fhe|maskcrypt|fisher)"
    r"_(?P<attack>gradinversion_breaching|gradinversion|ggcdm|gi_nas)"
    r"_b(?P<batch>\d+)\.json$"
)


def load_defence_sweep(data_dir: Path | None = None):
    """Return {(defence, attack, batch): metrics_dict} parsed from filenames.

    Backwards-compat single-seed loader. If per-seed subdirectories exist
    under ``data_dir`` (``seed42/``, ``seed123/`` ...), prefers ``seed42/``
    so the flat-file callers keep their original behaviour. If no per-seed
    subdir is present, falls back to the flat ``data_dir`` itself.
    """
    data_dir = data_dir or (DATA_DIR / "defence")
    seed_dirs = sorted(p for p in data_dir.glob("seed*") if p.is_dir())
    seed42_dir = next((p for p in seed_dirs if p.name == "seed42"), None)
    if seed42_dir is not None:
        # Explicit seed42/ subdir present — use it.
        scan_dir = seed42_dir
    else:
        # No seed42/ subdir; flat root is treated AS seed42 (the original
        # single-seed layout). Do NOT fall back to a different seed's subdir
        # — that would silently swap in seed123 or seed456 numbers and
        # mis-label them as the canonical single-seed reference.
        scan_dir = data_dir
    out = {}
    for p in sorted(scan_dir.glob("*.json")):
        m = _FILENAME_RE.match(p.name)
        if not m:
            continue
        key = (m.group("defence"), m.group("attack"), int(m.group("batch")))
        with p.open() as f:
            out[key] = json.load(f)
    return out


def load_defence_sweep_multiseed(data_dir: Path | None = None):
    """Return {(defence, attack, batch): {seed: metrics_dict, ...}}.

    Scans both per-seed subdirectories (``seed42/``, ``seed123/`` ...) and
    the flat ``data_dir`` root (treated as ``seed42`` if no ``seed42/``
    subdir is present, for backwards compat with the single-seed layout).
    """
    data_dir = data_dir or (DATA_DIR / "defence")
    out: dict = {}

    seed_dirs = sorted(p for p in data_dir.glob("seed*") if p.is_dir())
    has_seed42_subdir = any(p.name == "seed42" for p in seed_dirs)

    # Per-seed subdirectories (if any).
    for sd in seed_dirs:
        seed = sd.name.replace("seed", "")
        for p in sorted(sd.glob("*.json")):
            m = _FILENAME_RE.match(p.name)
            if not m:
                continue
            key = (m.group("defence"), m.group("attack"), int(m.group("batch")))
            with p.open() as f:
                out.setdefault(key, {})[seed] = json.load(f)

    # Flat root files (treated as seed42 only if there isn't already a seed42/).
    if not has_seed42_subdir:
        for p in sorted(data_dir.glob("*.json")):
            m = _FILENAME_RE.match(p.name)
            if not m:
                continue
            key = (m.group("defence"), m.group("attack"), int(m.group("batch")))
            with p.open() as f:
                out.setdefault(key, {})["42"] = json.load(f)

    return out


def aggregate_multiseed(multiseed_sweep, metrics=("psnr", "lpips", "ssim", "mse")):
    """Reduce a multi-seed sweep to per-cell summary statistics.

    Returns ``{(defence, attack, batch): {metric: {mean, std, n, values}}}``
    for the listed scalar metrics, plus a special ``"asr"`` metric pulled
    from ``label_recovery.asr`` (averaged across seeds).
    """
    import statistics as st

    summary: dict = {}
    for key, by_seed in multiseed_sweep.items():
        cell: dict = {}
        for metric in metrics:
            vals = []
            for seed, m in by_seed.items():
                v = m.get(metric)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            if not vals:
                cell[metric] = {"mean": None, "std": None, "n": 0,
                                "values": {}}
                continue
            cell[metric] = {
                "mean": st.mean(vals),
                "std": st.stdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals),
                "values": {seed: float(by_seed[seed][metric])
                           for seed in by_seed
                           if by_seed[seed].get(metric) is not None},
            }
        # ASR is nested.
        asr_vals = []
        for seed, m in by_seed.items():
            lr = m.get("label_recovery") or {}
            v = lr.get("asr")
            if v is None:
                continue
            try:
                asr_vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if asr_vals:
            cell["asr"] = {
                "mean": st.mean(asr_vals),
                "std": st.stdev(asr_vals) if len(asr_vals) > 1 else 0.0,
                "n": len(asr_vals),
                "values": {
                    seed: float((by_seed[seed].get("label_recovery") or {}).get("asr"))
                    for seed in by_seed
                    if (by_seed[seed].get("label_recovery") or {}).get("asr") is not None
                },
            }
        else:
            cell["asr"] = {"mean": None, "std": None, "n": 0, "values": {}}
        summary[key] = cell
    return summary


def load_consistency(path: Path | None = None):
    path = path or (DATA_DIR / "consistency" / "consistency.json")
    if not path.exists():
        return None
    with path.open() as f:
        blob = json.load(f)
    # File is nested under a top-level "consistency" key.
    return blob.get("consistency", blob)


def load_client_diversity(path: Path | None = None):
    """Return list of dicts: [{client, labels, n_unique}, ...].

    Parses the text file emitted by attack_sanity_check.py --list-clients.
    Actual format (space-separated table):
      client_NNN.pt                 K  [l1, l2, ...]
    """
    path = path or (DATA_DIR / "diverse" / "client_diversity.txt")
    if not path.exists():
        return []
    rows = []
    pat = re.compile(
        r"^\s*(?P<fname>client_\d+\.pt)\s+(?P<uniq>\d+)\s+"
        r"(?P<labels>\[[\d,\s]+\])"
    )
    with path.open() as f:
        for line in f:
            m = pat.match(line)
            if not m:
                continue
            labels = json.loads(m.group("labels"))
            rows.append({
                "client": m.group("fname"),
                "labels": labels,
                "n_unique": int(m.group("uniq")),
            })
    return rows


def defence_matrix(sweep, batch: int = 1):
    """Return dict[attack][defence] = metrics for the given batch size."""
    grid = {a: {} for a in ATTACK_ORDER}
    for (d, a, b), v in sweep.items():
        if b != batch or a not in grid:
            continue
        grid[a][d] = v
    return grid


def best_defence_per_attack(grid, metric: str, lower_is_better: bool = True):
    """For each attack, return the defence (excluding 'none') whose metric is best."""
    out = {}
    for attack, row in grid.items():
        candidates = {d: row[d].get(metric)
                      for d in DEFENCE_ORDER if d in row and d != "none"}
        candidates = {d: v for d, v in candidates.items() if v is not None}
        if not candidates:
            continue
        if lower_is_better:
            out[attack] = min(candidates, key=candidates.get)
        else:
            out[attack] = max(candidates, key=candidates.get)
    return out
