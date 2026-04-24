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
    """Return {(defence, attack, batch): metrics_dict} parsed from filenames."""
    data_dir = data_dir or (DATA_DIR / "defence")
    out = {}
    for p in sorted(data_dir.glob("*.json")):
        m = _FILENAME_RE.match(p.name)
        if not m:
            continue
        key = (m.group("defence"), m.group("attack"), int(m.group("batch")))
        with p.open() as f:
            out[key] = json.load(f)
    return out


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
