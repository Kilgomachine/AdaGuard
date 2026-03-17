"""Gradient extraction and noise utilities."""

import torch


def extract_gradients(model, images, labels, criterion, device):
    """Extract per-layer gradients from a forward-backward pass.

    Returns:
        gradient_dict: dict mapping param name -> gradient tensor
        flat_gradient: single flattened gradient vector
        loss_value: scalar loss
        outputs: model output logits (detached)
    """
    model.train()
    model.zero_grad()
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    gradient_dict = {}
    flat_parts = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            gradient_dict[name] = p.grad.clone().detach()
            flat_parts.append(p.grad.clone().detach().flatten())

    flat_gradient = torch.cat(flat_parts)
    return gradient_dict, flat_gradient, loss.item(), outputs.detach()


def add_noise_to_gradients(gradient_dict, sigma):
    """Add Gaussian noise to gradients for privacy simulation.

    Returns:
        noisy_dict: dict mapping param name -> noisy gradient tensor
        noisy_flat: flattened noisy gradient vector
    """
    noisy_dict = {
        name: grad + torch.randn_like(grad) * sigma
        for name, grad in gradient_dict.items()
    }
    noisy_flat = torch.cat([g.flatten() for g in noisy_dict.values()])
    return noisy_dict, noisy_flat
