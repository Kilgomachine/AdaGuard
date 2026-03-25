"""Federated Learning client — local training and gradient extraction."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from ..models import create_model


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
        """Perform local training and return the pseudo-gradient (weight delta).

        The client trains for `client_local_steps` SGD steps on its local data,
        then returns the difference (global_weights - local_weights) as the
        gradient to send to the server. This is standard FedAvg.

        IMPORTANT: Returns full state_dict delta (including BatchNorm running
        stats) so the server can properly aggregate BN buffers. Without this,
        BN running_mean/running_var stay at init values and eval accuracy = 10%.

        Also returns the raw single-batch gradient for LeakScore metrics
        (since privacy attacks target individual gradients, not weight deltas).

        Returns:
            dict with gradient_dict, flat_gradient, loss, outputs, labels,
            local_weights, local_state_dict, images — or None if no data.
        """
        bs = batch_size or self.config['client_batch_size']
        local_steps = self.config.get('client_local_steps', 1)
        local_lr = self.config.get('client_lr', self.config.get('fl_lr', 0.01))

        if not self.data_indices:
            return None

        # Create local model copy
        local_model = create_model(
            self.config.get('model', 'smallcnn'),
            num_classes=self.config['num_classes'],
        ).to(self.device)
        local_model.load_state_dict(global_state_dict)
        local_model.train()

        # Save initial weights (parameters only, for pseudo-gradient)
        global_weights = {
            name: p.clone().detach()
            for name, p in local_model.named_parameters()
        }

        loader = DataLoader(
            Subset(self.dataset, self.data_indices),
            batch_size=bs, shuffle=True,
        )

        # Local SGD training for multiple steps
        optimizer = optim.SGD(local_model.parameters(), lr=local_lr, momentum=0.9)
        last_loss = 0.0
        last_outputs = None
        last_labels = None
        last_images = None
        step = 0

        for epoch_pass in range(max(1, local_steps // max(len(loader), 1) + 1)):
            for imgs, lbls in loader:
                if step >= local_steps:
                    break
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)

                optimizer.zero_grad()
                outputs = local_model(imgs)
                loss = self.criterion(outputs, lbls)
                loss.backward()
                optimizer.step()

                last_loss = loss.item()
                last_outputs = outputs.detach()
                last_labels = lbls.detach()
                last_images = imgs.detach()
                step += 1

            if step >= local_steps:
                break

        if last_outputs is None:
            return None

        # Compute pseudo-gradient: (global_weights - local_weights)
        # This is what the server uses for FedAvg aggregation on parameters
        local_weights = {
            name: p.clone().detach()
            for name, p in local_model.named_parameters()
        }

        gradient_dict = {}
        for name in global_weights:
            # pseudo-gradient = old - new (so server does: global -= delta = local)
            gradient_dict[name] = (global_weights[name] - local_weights[name])

        flat_gradient = torch.cat([g.flatten() for g in gradient_dict.values()])

        # Full state dict for proper FedAvg (includes BN running_mean/var)
        local_state_dict = {
            k: v.clone().detach() for k, v in local_model.state_dict().items()
        }

        # Also compute single-batch raw gradient for LeakScore analysis
        # (privacy attacks target the actual gradient, not weight deltas)
        local_model.zero_grad()
        raw_outputs = local_model(last_images)
        raw_loss = self.criterion(raw_outputs, last_labels)
        raw_loss.backward()

        raw_gradient_dict = {
            name: p.grad.clone().detach()
            for name, p in local_model.named_parameters()
            if p.grad is not None
        }
        raw_flat = torch.cat([g.flatten() for g in raw_gradient_dict.values()])

        return {
            'gradient_dict': raw_gradient_dict,   # for LeakScore metrics
            'weight_delta': gradient_dict,          # param delta for encryption
            'flat_gradient': raw_flat,
            'loss': last_loss,
            'outputs': last_outputs,
            'labels': last_labels,
            'local_weights': local_weights,         # param-only local weights
            'local_state_dict': local_state_dict,   # full state including BN
            'images': last_images,
        }
