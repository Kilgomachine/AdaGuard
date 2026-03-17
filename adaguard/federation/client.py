"""Federated Learning client — local training and gradient extraction."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..models.cnn import SmallCNN


class FLClient:
    """Simulates a single FL client's local training step."""

    def __init__(self, client_id, dataset, data_indices, config, device):
        self.client_id = client_id
        self.dataset = dataset
        self.data_indices = data_indices
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, global_state_dict, batch_size=None):
        """Perform one local training step and return gradients.

        Args:
            global_state_dict: current global model state
            batch_size: override batch size (optional)

        Returns:
            gradient_dict: per-layer gradients
            flat_gradient: flattened gradient vector
            loss: scalar loss
            outputs: model logits (detached)
            None if no data available
        """
        bs = batch_size or self.config['client_batch_size']

        # Create local model copy
        local_model = SmallCNN(
            num_classes=self.config['num_classes'],
        ).to(self.device)
        local_model.load_state_dict(global_state_dict)
        local_model.train()

        # Load a batch
        if not self.data_indices:
            return None
        loader = DataLoader(
            Subset(self.dataset, self.data_indices),
            batch_size=bs, shuffle=True,
        )
        try:
            imgs, lbls = next(iter(loader))
        except StopIteration:
            return None

        imgs, lbls = imgs.to(self.device), lbls.to(self.device)

        # Forward-backward
        local_model.zero_grad()
        outputs = local_model(imgs)
        loss = self.criterion(outputs, lbls)
        loss.backward()

        # Collect gradients
        gradient_dict = {
            name: p.grad.clone().detach()
            for name, p in local_model.named_parameters()
            if p.grad is not None
        }
        flat_gradient = torch.cat([g.flatten() for g in gradient_dict.values()])

        # Collect locally trained weights (for MaskCrypt paper formula)
        local_weights = {
            name: p.clone().detach()
            for name, p in local_model.named_parameters()
        }

        return {
            'gradient_dict': gradient_dict,
            'flat_gradient': flat_gradient,
            'loss': loss.item(),
            'outputs': outputs.detach(),
            'labels': lbls.detach(),
            'local_weights': local_weights,
            'images': imgs.detach(),
        }
