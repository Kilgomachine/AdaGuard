"""Analytical label recovery from the final-layer gradient (iDLG / LLG).

No optimisation loop — labels are read off the sign pattern of the
fully-connected layer's bias (or row-sum) gradient. Runs in milliseconds.

Theory
------
Cross-entropy with softmax over C classes on a batch of B samples:

    dL/db[c] = (1/B) * sum_i (p_i[c] - 1_{c = y_i})

where p_i is the softmax posterior for sample i and y_i its true label.
Rearranging, the batch-level class count n_c satisfies

    n_c = (sum_i p_i[c]) - B * dL/db[c]

Classes that appear in the batch push their bias gradient strongly
negative; classes that do not stay near zero or positive. The LLG+
algorithm of Wainakh et al. (PETS 2022) ranks classes by -dL/db and
allocates a batch of predicted labels using magnitude-proportional
rounding. The B=1 case reduces to the iDLG theorem (Zhao et al. 2021):
argmin of the bias gradient is the true label.

If fc.bias is not present in the gradient dict (e.g. the final Linear
was constructed with bias=False, or the bias component was encrypted),
we fall back to the row-sum of fc.weight, which carries the same sign
pattern whenever the penultimate activations are non-negative (which
holds for ResNet after ReLU + avgpool).

References
----------
- Zhao, Mopuri, Bilen. "iDLG: Improved Deep Leakage from Gradients",
  ICLR 2021.
- Wainakh, Ventola, Mussig, Keim, Cordero, Zimmer, Grube, Fellendorf,
  Schiering, Strufe, Kersting. "User-Level Label Leakage from Gradients
  in Federated Learning", PETS 2022.
"""

from collections import Counter
from typing import Dict, Iterable, List, Optional

import torch


def _find_fc_keys(grad_dict: Dict[str, torch.Tensor],
                  num_classes: int) -> Dict[str, Optional[str]]:
    """Auto-detect the names of the final-layer bias and weight gradients.

    A tensor is accepted as fc.bias if it is 1-D with length == num_classes,
    and as fc.weight if it is 2-D with first dim == num_classes. If several
    candidates match, the one whose key ends with '.bias' / '.weight'
    wins; ties are broken lexicographically, which in practice picks the
    outer-most classifier (e.g. 'fc.bias' beats 'layer4.1.bn2.bias').
    """
    bias_cands: List[str] = []
    weight_cands: List[str] = []
    for name, g in grad_dict.items():
        if not isinstance(g, torch.Tensor):
            continue
        if g.ndim == 1 and g.shape[0] == num_classes and name.endswith('.bias'):
            bias_cands.append(name)
        elif g.ndim == 2 and g.shape[0] == num_classes and name.endswith('.weight'):
            weight_cands.append(name)
    # Prefer the shortest key (likely the final classifier, e.g. 'fc.bias'
    # over 'layer4.1.bn2.bias'). Tiebreak lexicographically.
    bias_key = min(bias_cands, key=lambda k: (len(k), k)) if bias_cands else None
    weight_key = min(weight_cands, key=lambda k: (len(k), k)) if weight_cands else None
    return {'bias': bias_key, 'weight': weight_key}


def recover_labels_llg(
    grad_dict: Dict[str, torch.Tensor],
    batch_size: int,
    num_classes: int = 10,
    fc_bias_key: Optional[str] = None,
    fc_weight_key: Optional[str] = None,
) -> Dict[str, object]:
    """Recover a length-`batch_size` multiset of class labels from grad_dict.

    Args:
        grad_dict: per-layer gradient dict as saved by the Phase-1 pipeline.
        batch_size: number of labels to allocate (== client batch size).
        num_classes: number of output classes (default 10 for CIFAR-10).
        fc_bias_key, fc_weight_key: explicit overrides; auto-detected if None.

    Returns:
        dict with keys:
            'predicted' : list[int] of length == batch_size
            'source'    : 'fc.bias' | 'fc.weight.rowsum' | 'none'
            'key_used'  : actual grad_dict key consulted, or None
            'signal'    : float, total absolute signal (||g||_1). Near-zero
                          indicates the relevant gradient component was
                          masked/encrypted and no information leaked.
    """
    keys = _find_fc_keys(grad_dict, num_classes)
    bias_key = fc_bias_key if fc_bias_key is not None else keys['bias']
    weight_key = fc_weight_key if fc_weight_key is not None else keys['weight']

    g: Optional[torch.Tensor] = None
    source = 'none'
    key_used: Optional[str] = None
    signal = 0.0

    # Prefer fc.bias; if it's present but the tensor is all zeros (i.e. the
    # bias component was encrypted/masked) fall through to fc.weight.rowsum
    # so we still catch partial-encryption defences that leave some weight
    # rows intact.
    if bias_key is not None and bias_key in grad_dict:
        cand = grad_dict[bias_key].detach().flatten().float().cpu()
        cand_signal = float(cand.abs().sum().item())
        if cand_signal >= 1e-12:
            g = cand
            source = 'fc.bias'
            key_used = bias_key
            signal = cand_signal

    if g is None and weight_key is not None and weight_key in grad_dict:
        cand = grad_dict[weight_key].detach().sum(dim=1).float().cpu()
        cand_signal = float(cand.abs().sum().item())
        g = cand
        source = 'fc.weight.rowsum'
        key_used = weight_key
        signal = cand_signal

    if g is None:
        return {
            'predicted': [0] * int(batch_size),
            'source': 'none',
            'key_used': None,
            'signal': 0.0,
        }

    # Encrypted / masked-to-zero case: no information, return a constant
    # guess so accuracy collapses against any non-trivial truth set.
    if signal < 1e-12:
        return {
            'predicted': [0] * int(batch_size),
            'source': source,
            'key_used': key_used,
            'signal': signal,
        }

    order = torch.argsort(g)  # ascending => most-negative first

    if batch_size == 1:
        return {
            'predicted': [int(order[0].item())],
            'source': source,
            'key_used': key_used,
            'signal': signal,
        }

    # LLG+ magnitude-proportional allocation on the negative mass.
    neg = torch.clamp(-g, min=0.0)
    total_neg = float(neg.sum().item())
    if total_neg > 0:
        weights = neg / total_neg
        counts = torch.round(weights * batch_size).long()
    else:
        counts = torch.zeros(num_classes, dtype=torch.long)

    diff = int(batch_size - counts.sum().item())
    if diff > 0:
        for c in order.tolist():
            if diff == 0:
                break
            counts[c] += 1
            diff -= 1
    elif diff < 0:
        for c in reversed(order.tolist()):
            if diff == 0:
                break
            if counts[c] > 0:
                counts[c] -= 1
                diff += 1

    predicted: List[int] = []
    for c in range(num_classes):
        predicted.extend([c] * int(counts[c].item()))

    return {
        'predicted': predicted,
        'source': source,
        'key_used': key_used,
        'signal': signal,
    }


def label_recovery_accuracy(predicted: Iterable[int],
                            truth: Iterable[int]) -> float:
    """Multiset-overlap accuracy in [0, 1].

    Intersects predicted and truth as multisets, counts the overlap, and
    divides by len(truth). Order-independent. For a perfect LLG+ match
    on a B=16 batch with [1,1,5,5,6,6,...] the metric returns 1.0; for a
    trivial constant-zero prediction against the same truth it returns
    whatever fraction of the truth happens to be class 0.
    """
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().tolist()
    if isinstance(truth, torch.Tensor):
        truth = truth.detach().cpu().tolist()

    p = Counter(int(x) for x in predicted)
    t = Counter(int(x) for x in truth)
    overlap = sum((p & t).values())
    truth_list = list(truth) if not isinstance(truth, list) else truth
    denom = max(1, len(truth_list))
    return overlap / denom


def label_recovery_summary(
    grad_dict: Dict[str, torch.Tensor],
    truth_labels: Iterable[int],
    num_classes: int = 10,
) -> Dict[str, object]:
    """One-shot helper: run LLG and score against ground truth.

    Returns a dict suitable for dumping straight into the sanity-check
    JSON report:
        {
            'asr'       : float in [0, 1],
            'predicted' : list[int],
            'truth'     : list[int],
            'source'    : 'fc.bias' | 'fc.weight.rowsum' | 'none',
            'key_used'  : actual gradient key consulted,
            'signal'    : ||g||_1 of the final-layer signal used,
        }
    """
    if isinstance(truth_labels, torch.Tensor):
        truth_list = truth_labels.detach().cpu().tolist()
    else:
        truth_list = [int(x) for x in truth_labels]

    batch_size = len(truth_list) if len(truth_list) > 0 else 1
    rec = recover_labels_llg(grad_dict, batch_size, num_classes=num_classes)
    acc = label_recovery_accuracy(rec['predicted'], truth_list)

    return {
        'asr': float(acc),
        'predicted': list(rec['predicted']),
        'truth': [int(x) for x in truth_list],
        'source': rec['source'],
        'key_used': rec['key_used'],
        'signal': float(rec['signal']),
    }
