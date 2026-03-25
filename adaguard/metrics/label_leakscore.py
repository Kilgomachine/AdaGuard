"""Label LeakScore: GLMIP, Confidence Gap, Cosine Similarity.

Measures how much label information is encoded in gradients.
"""

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


class GLMIPMetric:
    """Gradient-Label Mutual Information Proxy.

    Computes between-class vs within-class gradient variance ratio
    to estimate how distinguishable class labels are from gradients.
    Score = S_B / (S_B + S_W), normalized to [0,1].
    """

    def compute(self, model, dataset, device, criterion,
                num_classes=10, samples_per_class=20, focus_layers=None):
        model.eval()

        # Fast label indexing — avoid iterating 50K dataset items
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
            if isinstance(targets, torch.Tensor):
                targets = targets.tolist()
            label_idx = defaultdict(list)
            for i, l in enumerate(targets):
                label_idx[int(l)].append(i)
        elif hasattr(dataset, 'labels'):
            targets = dataset.labels
            if isinstance(targets, torch.Tensor):
                targets = targets.tolist()
            label_idx = defaultdict(list)
            for i, l in enumerate(targets):
                label_idx[int(l)].append(i)
        else:
            # Fallback: iterate dataset (slow)
            label_idx = defaultdict(list)
            for i in range(len(dataset)):
                _, l = dataset[i]
                label_idx[int(l) if isinstance(l, torch.Tensor) else l].append(i)

        class_grads = defaultdict(list)

        for c in range(num_classes):
            ids = label_idx.get(c, [])
            if not ids:
                continue

            sampled = random.sample(ids, min(samples_per_class, len(ids)))

            # Batch processing: accumulate per-sample gradients efficiently
            # Process in mini-batches of 16 for GPU efficiency
            MINI_BS = 16
            for batch_start in range(0, len(sampled), MINI_BS):
                batch_ids = sampled[batch_start:batch_start + MINI_BS]
                imgs = []
                lbls = []
                for idx in batch_ids:
                    img, lbl = dataset[idx]
                    imgs.append(img)
                    lbls.append(lbl if isinstance(lbl, int) else int(lbl))

                img_batch = torch.stack(imgs).to(device)
                lbl_batch = torch.tensor(lbls, dtype=torch.long, device=device)

                # Per-sample gradients via loop (need individual gradients, not batch mean)
                for i in range(len(img_batch)):
                    model.zero_grad()
                    out = model(img_batch[i:i+1])
                    criterion(out, lbl_batch[i:i+1]).backward()

                    if focus_layers:
                        parts = []
                        for name, p in model.named_parameters():
                            if p.grad is not None and name in focus_layers:
                                parts.append(p.grad.clone().detach().flatten())
                        flat = torch.cat(parts) if parts else torch.tensor([])
                    else:
                        flat = torch.cat([
                            p.grad.clone().detach().flatten()
                            for p in model.parameters() if p.grad is not None
                        ])

                    if flat.numel() > 0:
                        class_grads[c].append(flat)

        class_means = {c: torch.stack(gs).mean(0) for c, gs in class_grads.items()}
        all_g = [g for gs in class_grads.values() for g in gs]
        if not all_g:
            return {'glmip_score': 0.0, 'class_means': {}}

        mu = torch.stack(all_g).mean(0)

        S_B = sum(
            len(class_grads[c]) * torch.dot(class_means[c] - mu, class_means[c] - mu).item()
            for c in class_grads
        )
        S_W = sum(
            torch.dot(g - class_means[c], g - class_means[c]).item()
            for c in class_grads for g in class_grads[c]
        )

        score = S_B / (S_B + S_W) if (S_B + S_W) > 1e-12 else 0.0
        return {
            'glmip_score': max(0.0, min(1.0, score)),
            'S_B': S_B,
            'S_W': S_W,
            'class_means': class_means,
        }


class ConfidenceGapMetric:
    """Confidence Gap: max(p) - second_max(p).

    High gap means model is confident -> label is strongly encoded.
    """

    def compute(self, logits):
        if isinstance(logits, torch.Tensor):
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        else:
            probs = logits

        if probs.ndim == 2:
            gaps = [np.sort(p)[::-1][0] - np.sort(p)[::-1][1] for p in probs]
            return {'confidence_gap': max(0.0, min(1.0, float(np.mean(gaps))))}

        s = np.sort(probs)[::-1]
        return {'confidence_gap': max(0.0, min(1.0, float(s[0] - s[1])))}


class CosineSimilarityMetric:
    """Gradient Cosine Similarity Across Classes.

    Low inter-class cosine similarity -> label strongly encoded -> high leak risk.
    Score = 1 - normalized_mean_cosine, in [0,1].
    """

    def compute(self, class_means):
        if len(class_means) < 2:
            return {'cosine_leak_score': 0.0, 'mean_cosine_similarity': 0.0}

        classes = sorted(class_means.keys())
        sims = [
            F.cosine_similarity(
                class_means[classes[i]].unsqueeze(0),
                class_means[classes[j]].unsqueeze(0),
            ).item()
            for i in range(len(classes))
            for j in range(i + 1, len(classes))
        ]
        m = float(np.mean(sims))
        return {
            'cosine_leak_score': max(0.0, min(1.0, 1.0 - (m + 1.0) / 2.0)),
            'mean_cosine_similarity': m,
        }
