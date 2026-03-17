"""Per-round metric logging for AdaGuard experiments."""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch


class RoundLogger:
    """Logs per-round metrics to CSV and JSON for experiment analysis."""

    def __init__(self, output_dir='./results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.per_round_fisher = []
        self.per_round_maskcrypt = []

    def log_round(self, round_summary):
        """Log a round summary dict.

        Extracts scalar metrics and stores tensor data separately.
        """
        record = {}
        for k, v in round_summary.items():
            if isinstance(v, (int, float, str, bool)):
                record[k] = v
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                record[k] = v.item()

        self.records.append(record)

        # Store per-weight tensors for visualization
        if 'client_details' in round_summary and round_summary['client_details']:
            first_client = round_summary['client_details'][0]['metrics']
            self.per_round_fisher.append(
                first_client.get('fisher_per_weight', torch.tensor([]))
            )
            self.per_round_maskcrypt.append(
                first_client.get('maskcrypt_per_weight_abs', torch.tensor([]))
            )

    def to_dataframe(self):
        """Convert logged records to pandas DataFrame."""
        return pd.DataFrame(self.records)

    def save(self, experiment_name='default'):
        """Save logs to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = f"{experiment_name}_{timestamp}"

        # CSV
        df = self.to_dataframe()
        csv_path = self.output_dir / f"{prefix}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        # JSON (scalar metrics only)
        json_path = self.output_dir / f"{prefix}.json"
        with open(json_path, 'w') as f:
            json.dump(self.records, f, indent=2, default=str)
        print(f"Saved: {json_path}")

        return csv_path, json_path

    def reset(self):
        """Clear all logged data."""
        self.records.clear()
        self.per_round_fisher.clear()
        self.per_round_maskcrypt.clear()
