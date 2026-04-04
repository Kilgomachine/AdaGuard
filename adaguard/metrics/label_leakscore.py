"""Label LeakScore: GLMIP, Confidence Gap, Cosine Similarity.

Measures how much label information is encoded in gradients.
"""

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


class GLMIPMetric:
    """Gradient-Label Mutual Information Proxy (Eq. 3 in paper).

    Score_GLMIP = 1 - (-Σ P(g|c) · log P(g|c)) / log(C)

    Computes the entropy of per-class gradient energy.  When gradient
    energy is uniformly distributed across classes the label is hard to
    infer (Score → 0).  When one class dominates, the label leaks
    (Score → 1).
    """

    def compute(self, model, dataset, device, criterion,
                num_classes=10, samples_per_class=20, focus_layers=None):
        model.eval()

        # Fast label indexing
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
            label_idx = defaultdict(list)
            for i in range(len(dataset)):
                _, l = dataset[i]
                label_idx[int(l) if isinstance(l, torch.Tensor) else l].append(i)

        class_grad_norms = {}  # class → list of per-sample gradient L2 norms
        class_mean_grads = {}  # class → mean gradient vector (for cosine metric)

        for c in range(num_classes):
            ids = label_idx.get(c, [])
            if not ids:
                continue

            sampled = random.sample(ids, min(samples_per_class, len(ids)))
            norms = []
            grads = []

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
                        norms.append(flat.norm(2).item())
                        grads.append(flat)

            if norms:
                class_grad_norms[c] = norms
                class_mean_grads[c] = torch.stack(grads).mean(0)

        if not class_grad_norms:
            return {'glmip_score': 0.0, 'class_means': {}}

        # Eq. 3: P(g|c) = total gradient energy for class c / total energy
        class_energy = {
            c: sum(n ** 2 for n in norms_list)
            for c, norms_list in class_grad_norms.items()
        }
        total_energy = sum(class_energy.values())
        if total_energy < 1e-12:
            return {'glmip_score': 0.0, 'class_means': class_mean_grads}

        # Probability distribution over classes
        p_gc = np.array([class_energy.get(c, 0.0) for c in range(num_classes)])
        p_gc = p_gc / total_energy
        p_gc = p_gc[p_gc > 0]  # remove zero entries for log

        C = num_classes
        H = -np.sum(p_gc * np.log(p_gc))
        score = 1.0 - H / np.log(C) if C > 1 else 0.0

        return {
            'glmip_score': max(0.0, min(1.0, score)),
            'class_means': class_mean_grads,
        }


class ConfidenceGapMetric:
    """Confidence Gap (Eq. 4 in paper).

    Score_Gap = max(|g_logits|) - second_max(|g_logits|)

    where g_logits = ∂L/∂z = softmax(z) - one_hot(y).

    A wide gap means the true-class gradient is clearly separable from
    other classes, giving the attacker an analytical label leak (iDLG).
    """

    def compute(self, logits, labels=None):
        """Compute confidence gap from logit gradients.

        Args:
            logits: model output logits, shape (B, C)
            labels: ground-truth labels, shape (B,). If None, falls back
                    to softmax probability gap (less accurate).
        """
        if isinstance(logits, torch.Tensor):
            logits_t = logits.detach()
        else:
            logits_t = torch.tensor(logits)

        if logits_t.ndim == 1:
            logits_t = logits_t.unsqueeze(0)

        probs = F.softmax(logits_t, dim=-1)

        if labels is not None:
            # Paper Eq. 4: gradient of cross-entropy w.r.t. logits
            if isinstance(labels, torch.Tensor):
                labels_t = labels.detach()
            else:
                labels_t = torch.tensor(labels, dtype=torch.long)
            one_hot = torch.zeros_like(probs)
            one_hot.scatter_(1, labels_t.unsqueeze(1), 1.0)
            g_logits = (probs - one_hot).abs().cpu().numpy()
        else:
            # Fallback: use softmax probabilities directly
            g_logits = probs.cpu().numpy()

        gaps = []
        for row in g_logits:
            s = np.sort(row)[::-1]
            gaps.append(s[0] - s[1])

        return {'confidence_gap': max(0.0, min(1.0, float(np.mean(gaps))))}


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
