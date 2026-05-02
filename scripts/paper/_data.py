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
    # Paper-side convention: V1-V5 contiguous (codebase scenario
    # registry uses V1, V2, V3=DP (omitted), V4=MaskCrypt-guided,
    # V6=AdaGuard, V14=SelectiveShield, but the *paper* renumbers
    # them V1-V5 because DP is omitted from the headline matrix).
    # Only the V-strings change; the file-name tokens (none/fhe/
    # maskcrypt/fisher/selectiveshield) are stable.
    "none": "V1 (None)",
    "fhe": "V2 (FHE)",
    "maskcrypt": "V3 (MaskCrypt)",
    "fisher": "V4 (AdaGuard-Fisher)",
    "selectiveshield": "V5 (SelectiveShield)",
}
DEFENCE_ORDER = ["none", "fhe", "maskcrypt", "fisher", "selectiveshield"]

ATTACK_LABELS = {
    "gradinversion": "GradInversion",
    "gradinversion_breaching": "GradInversion (Yin)",
    "ggcdm": "GGCDM",
    "gi_nas": "GI-NAS",
}
ATTACK_ORDER = ["gradinversion", "ggcdm", "gi_nas"]


_FILENAME_RE = re.compile(
    r"^(?P<defence>none|fhe|maskcrypt|fisher|selectiveshield)"
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


# --------------------------------------------------------------------------
# V13 (Fisher-vs-random ablation): loader and aggregator.
#
# The V13 viability scenario reuses the V4 (Fisher) defence path with
# ``mask_mode='random'`` flipped on, holding rho=10% fixed. JSON layout
# under data/paper_data/defence/v13_random/ is FLAT (no per-seed subdir);
# the seed is encoded in the filename:
#    fisher_random_<attack>_b<B>_seed<seed>.json
# --------------------------------------------------------------------------

_V13_FILENAME_RE = re.compile(
    r"^fisher_random"
    r"_(?P<attack>gradinversion|ggcdm|gi_nas)"
    r"_b(?P<batch>\d+)"
    r"_seed(?P<seed>\d+)\.json$"
)

_V5_SS_FILENAME_RE = re.compile(
    r"^selectiveshield"
    r"_(?P<attack>gradinversion|ggcdm|gi_nas)"
    r"_b(?P<batch>\d+)"
    r"_seed(?P<seed>\d+)\.json$"
)


def load_v13_random_sweep(data_dir: Path | None = None):
    """Return ``{(attack, batch): {seed: metrics_dict}}`` for the V13 ablation.

    Reads the Fisher-vs-random ablation JSONs from
    ``data/paper_data/defence/v13_random/``. Each JSON has the same metric
    schema as the regular defence sweep (psnr, lpips, ssim, mse,
    label_recovery.asr) plus a ``defence_meta.mask_mode='random'`` field
    we record for traceability.

    Returns an empty dict if the directory does not exist — callers should
    skip emitting the comparison table in that case rather than raising.
    """
    data_dir = data_dir or (DATA_DIR / "defence" / "v13_random")
    out: dict = {}
    if not data_dir.exists():
        return out
    for p in sorted(data_dir.glob("*.json")):
        m = _V13_FILENAME_RE.match(p.name)
        if not m:
            continue
        key = (m.group("attack"), int(m.group("batch")))
        seed = m.group("seed")
        with p.open() as f:
            out.setdefault(key, {})[seed] = json.load(f)
    return out


def merge_v5_selectiveshield(multiseed_sweep, v5_data_dir=None):
    """Merge V5 SelectiveShield JSONs into an existing multi-seed sweep dict.

    V5 results follow the V13 layout convention: flat directory at
    ``data/paper_data/defence/v5_selectiveshield/`` with files named
    ``selectiveshield_<attack>_b<B>_seed<seed>.json`` (the seed
    suffix is in the filename rather than a per-seed subdirectory).
    The merge folds these into the existing
    ``{(defence, attack, batch): {seed: metrics}}`` shape used by
    :func:`load_defence_sweep_multiseed` and consumed by
    :func:`aggregate_multiseed`, so the existing
    ``build_table_defence_matrix_multiseed`` builder picks up V5 as
    a fifth defence column without code changes.

    Returns the same dict (mutated in place) for chain-of-call
    convenience. If the V5 directory doesn't exist, the dict is
    returned unchanged.
    """
    v5_data_dir = v5_data_dir or (DATA_DIR / "defence" / "v5_selectiveshield")
    if not v5_data_dir.exists():
        return multiseed_sweep
    for p in sorted(v5_data_dir.glob("*.json")):
        m = _V5_SS_FILENAME_RE.match(p.name)
        if not m:
            continue
        attack = m.group("attack")
        batch = int(m.group("batch"))
        seed = m.group("seed")
        key = ("selectiveshield", attack, batch)
        with p.open() as f:
            multiseed_sweep.setdefault(key, {})[seed] = json.load(f)
    return multiseed_sweep


def merge_v5_selectiveshield_single_seed(sweep, v5_data_dir=None,
                                         seed_to_use="42"):
    """Merge V5 SelectiveShield JSONs into a single-seed sweep dict.

    Mirror of :func:`merge_v5_selectiveshield` for the single-seed
    sweep dict shape ``{(defence, attack, batch): metrics}``. By
    default merges only the seed-42 entries to match the
    ``load_defence_sweep`` single-seed-reference behaviour. Pass
    ``seed_to_use=None`` to merge whichever seed is found first
    (used when only one V5 seed has run).
    """
    v5_data_dir = v5_data_dir or (DATA_DIR / "defence" / "v5_selectiveshield")
    if not v5_data_dir.exists():
        return sweep
    for p in sorted(v5_data_dir.glob("*.json")):
        m = _V5_SS_FILENAME_RE.match(p.name)
        if not m:
            continue
        seed = m.group("seed")
        if seed_to_use is not None and seed != seed_to_use:
            continue
        attack = m.group("attack")
        batch = int(m.group("batch"))
        key = ("selectiveshield", attack, batch)
        # Only set if not already present (don't overwrite a seed-42
        # value with a seed-other if the loader's single-seed default
        # changes upstream).
        if key not in sweep:
            with p.open() as f:
                sweep[key] = json.load(f)
    return sweep


def aggregate_v13(v13_sweep, metrics=("psnr", "lpips", "ssim", "mse")):
    """Reduce a V13 sweep to ``{(attack, batch): {metric: {mean, std, n, values}}}``.

    Mirrors :func:`aggregate_multiseed` but keyed by ``(attack, batch)``
    rather than ``(defence, attack, batch)`` since V13 is a single-defence
    ablation. Pulls ``label_recovery.asr`` into a synthetic ``"asr"`` metric
    just like the multi-seed aggregator.
    """
    import statistics as st

    summary: dict = {}
    for key, by_seed in v13_sweep.items():
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
