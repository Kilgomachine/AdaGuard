"""Unit tests for the AdaGuard-Fisher classifier-head guarantee.

Captures the seed-456 round-249 failure mode in a synthetic setting:
when ``fc.bias`` is small relative to a single conv layer's
gradient mass, a pure global top-K-by-Fisher selector can drop it
from the encrypted set, exposing the LLG+ label-decoding signal.

These tests should fail on the pre-fix selector (broken behaviour
reproducible via ``mandatory_layer_substrings=()``) and pass on the
post-fix default.

Run:
    python -m pytest tests/test_fisher_classifier_head_guarantee.py -v
or:
    python tests/test_fisher_classifier_head_guarantee.py

Compatibility note: this file deliberately avoids ``from __future__
import annotations`` and PEP-604 ``X | None`` syntax so it can be
run on any Python >= 3.5, including HPC login nodes that default
to an old system Python.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaguard.metrics.fisher import (  # noqa: E402
    FisherInformationMetric, DEFAULT_MANDATORY_LAYER_SUBSTRINGS,
)
from adaguard.encryption.fisher_encrypt import FisherEncryptor  # noqa: E402


def _adversarial_grad_dict(seed: int = 456):
    """Build a gradient_dict that mimics the seed-456 failure shape.

    A fat conv-like layer with all-large gradients overwhelms a tiny
    10-element ``fc.bias`` whose per-element Fisher score is moderate
    but whose aggregate mass is tiny. This is the exact pattern that
    bypassed the pre-fix top-K Fisher selector on seed 456 round 249.
    """
    torch.manual_seed(seed)
    return {
        # Pretend this is conv1 of ResNet-18 (large): 5,000 params with
        # all-positive gradients of magnitude ~1.0. Their fi = g^2 ~= 1.0.
        "layer1.0.conv1.weight": torch.randn(5000) + 1.5,
        # The classifier head: only 10 params, gradients of order ~0.05.
        # fi = g^2 ~= 2.5e-3 — well below the conv layer's median.
        "fc.bias": 0.05 * torch.randn(10),
        # A second small classifier-head tensor for completeness.
        "fc.weight": 0.05 * torch.randn(50),
    }


def _is_layer_fully_encrypted(metric_result, gradient_dict, layer_name: str):
    """Returns True iff every parameter of ``layer_name`` is in the mask."""
    mask = metric_result["encrypt_mask"]
    cursor = 0
    for name, grad in gradient_dict.items():
        n = grad.numel()
        if name == layer_name:
            return bool(mask[cursor:cursor + n].all().item())
        cursor += n
    raise KeyError(f"layer {layer_name!r} not found in gradient_dict")


def test_pre_fix_drops_fc_bias_from_topk():
    """Pre-fix selector (mandatory=()) drops fc.bias on the adversarial dict.

    This is the bug. We assert it explicitly so a future refactor
    that silently changes default behaviour is caught.
    """
    gd = _adversarial_grad_dict()
    metric = FisherInformationMetric(enc_pct=0.10, mandatory_layer_substrings=())
    result = metric.compute(gd)
    assert not _is_layer_fully_encrypted(result, gd, "fc.bias"), (
        "Adversarial fixture did not reproduce the pre-fix bug — fc.bias "
        "was unexpectedly inside the top-10% set. Bump conv layer size "
        "or shrink fc.bias gradients to restore the failure pattern."
    )
    assert result["classifier_head_forced_count"] == 0


def test_post_fix_forces_fc_bias_into_mask():
    """Post-fix selector (default mandatory list) forces fc.bias in.

    Default ``mandatory_layer_substrings=None`` resolves to
    ``DEFAULT_MANDATORY_LAYER_SUBSTRINGS`` which includes ``fc.bias``.
    """
    gd = _adversarial_grad_dict()
    metric = FisherInformationMetric(enc_pct=0.10)  # default = post-fix
    result = metric.compute(gd)
    assert _is_layer_fully_encrypted(result, gd, "fc.bias")
    assert _is_layer_fully_encrypted(result, gd, "fc.weight")
    # Some of these may already have been in the top-K via Fisher score —
    # the count is "what we ADDED beyond top-K", so it's >= 0.
    assert result["classifier_head_forced_count"] >= 1
    assert "fc.bias" in result["classifier_head_forced_layers"]


def test_post_fix_zeros_fc_bias_gradient():
    """End-to-end: FisherEncryptor.encrypt with default selector must
    zero fc.bias in the protected gradient on the adversarial fixture."""
    gd = _adversarial_grad_dict()
    enc = FisherEncryptor()  # default metric => post-fix
    protected, meta = enc.encrypt(gd, k=int(0.10 * sum(g.numel() for g in gd.values())))
    assert torch.allclose(protected["fc.bias"], torch.zeros_like(gd["fc.bias"])), (
        "Post-fix Fisher must zero fc.bias even when it falls below the "
        "global top-K threshold. This is the entire purpose of the "
        "classifier-head guarantee. See fisher.py module docstring."
    )
    assert torch.allclose(protected["fc.weight"], torch.zeros_like(gd["fc.weight"]))
    assert meta["classifier_head_forced_count"] >= 1
    assert "fc.bias" in meta["classifier_head_forced_layers"]


def test_pre_fix_leaves_fc_bias_signal_intact():
    """Ablation symmetry: with mandatory=(), fc.bias gradient survives
    encryption — confirming the failure mode the fix targets."""
    gd = _adversarial_grad_dict()
    pre_fix_metric = FisherInformationMetric(
        enc_pct=0.10, mandatory_layer_substrings=()
    )
    enc = FisherEncryptor(fisher_metric=pre_fix_metric)
    protected, meta = enc.encrypt(gd, k=int(0.10 * sum(g.numel() for g in gd.values())))
    # fc.bias should NOT be all-zeros on the pre-fix path.
    assert not torch.allclose(protected["fc.bias"], torch.zeros_like(gd["fc.bias"])), (
        "Pre-fix path unexpectedly zeroed fc.bias — the adversarial "
        "fixture must be re-tuned so the bug is reproducible."
    )
    assert meta["classifier_head_forced_count"] == 0


def test_default_substrings_cover_common_naming():
    """The default substring list must cover torchvision/HF/timm head naming.

    Regression guard: if someone shortens DEFAULT_MANDATORY_LAYER_SUBSTRINGS
    we want a loud failure, not a silent regression on unfamiliar models.
    """
    expected = {"fc.bias", "fc.weight",
                "classifier.bias", "classifier.weight",
                "head.bias", "head.weight"}
    assert set(DEFAULT_MANDATORY_LAYER_SUBSTRINGS) == expected


def test_pct_encrypted_reflects_merged_mask():
    """weights_to_encrypt and pct_encrypted must reflect the post-merge
    mask, not the pre-merge top-K count, so the JSON sweep summaries
    are self-consistent."""
    gd = _adversarial_grad_dict()
    metric = FisherInformationMetric(enc_pct=0.10)  # default = post-fix
    result = metric.compute(gd)
    expected_count = int(result["encrypt_mask"].sum().item())
    assert result["weights_to_encrypt"] == expected_count
    n_total = sum(g.numel() for g in gd.values())
    # Tolerance accommodates float32 (mask.float().mean()) vs float64
    # (Python int division) precision drift; we only need to catch
    # off-by-one or off-by-fraction bugs, not last-bit jitter.
    assert abs(result["pct_encrypted"] - expected_count / n_total) < 1e-6


if __name__ == "__main__":
    # Run inline so this file works without pytest installed on HPC.
    import traceback
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL  {name}\n        {e}")
            except Exception:
                failures += 1
                print(f"  ERROR {name}")
                traceback.print_exc()
    if failures:
        print(f"\n{failures} test(s) failed")
        sys.exit(1)
    print(f"\nall tests passed")
