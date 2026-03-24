"""FedAvg gradient aggregation."""

import torch


def fedavg_aggregate(gradient_dicts):
    """Average gradients from multiple clients (FedAvg).

    Args:
        gradient_dicts: list of gradient dicts from clients

    Returns:
        averaged gradient dict
    """
    if not gradient_dicts:
        return {}

    avg = {}
    for name in gradient_dicts[0]:
        stacked = torch.stack([gd[name] for gd in gradient_dicts])
        avg[name] = stacked.mean(0)

    return avg


def apply_gradient_update(model, averaged_gradients, lr):
    """Apply aggregated weight deltas to global model.

    weight_delta = global_weights - local_weights (computed in client.py).
    This IS the direction of change, so we subtract it directly:
        new_global = global - avg(global - local) = avg(local)
    The lr parameter is kept for API compatibility but set to 1.0 for
    standard FedAvg (weight delta already encodes the full update).
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in averaged_gradients:
                p -= averaged_gradients[name]
