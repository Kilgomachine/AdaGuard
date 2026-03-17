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
    """Apply aggregated gradients to global model.

    Args:
        model: global model to update
        averaged_gradients: averaged gradient dict
        lr: learning rate
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in averaged_gradients:
                p -= lr * averaged_gradients[name]
