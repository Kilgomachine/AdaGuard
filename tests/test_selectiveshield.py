"""Unit tests for the SelectiveShield encryptor (paper V5 baseline).

Locks in the contract that SelectiveShield = Fisher-targeted SHE
(top-K zeroing + classifier-head guarantee, identical to V4) plus a
DP-noise layer on the non-encrypted slice. The tests verify each
piece independently so a future refactor that breaks either the
Fisher side or the DP side gets caught loudly.

Run:
    python -m pytest tests/test_selectiveshield.py -v
or:
    python tests/test_selectiveshield.py

Compatibility note: avoids ``from __future__ import annotations`` and
PEP-604 union syntax for HPC login-node Python compat (matches the
test_fisher_classifier_head_guarantee.py convention).
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaguard.metrics.fisher import FisherInformationMetric  # noqa: E402
from adaguard.encryption.selectiveshield_encrypt import (  # noqa: E402
    SelectiveShieldEncryptor,
)


def _adversarial_grad_dict(seed=456):
    """Same fixture shape as the Fisher classifier-head tests.

    Adversarial: a fat conv-like layer dominates the Fisher score
    distribution; small classifier-head tensors carry the iDLG signal
    but aggregate-Fisher-rank low. A correct SelectiveShield should
    still encrypt fc.bias / fc.weight via the head guarantee.
    """
    torch.manual_seed(seed)
    return {
        "layer1.0.conv1.weight": torch.randn(5000) + 1.5,
        "fc.bias": 0.05 * torch.randn(10),
        "fc.weight": 0.05 * torch.randn(50),
    }


def _is_layer_zeroed(protected, layer):
    """Returns True iff every value in ``layer`` is exactly zero."""
    return torch.allclose(protected[layer], torch.zeros_like(protected[layer]))


def _layer_was_perturbed(original, protected, layer):
    """Returns True iff the protected layer differs from the original
    (within the original's order of magnitude — i.e. not just
    floating-point identity)."""
    diff = (original[layer] - protected[layer]).abs().max().item()
    orig_max = original[layer].abs().max().item()
    return diff > 1e-8 and diff < 100 * max(orig_max, 1e-8)


def test_encrypted_set_is_zeroed_by_selectiveshield():
    """Top-K-by-Fisher parameters must end up exactly zero (the SHE
    side of the hybrid). Use the classifier-head guarantee so fc.* is
    in the encrypted set on the adversarial fixture."""
    gd = _adversarial_grad_dict()
    enc = SelectiveShieldEncryptor(enc_pct=0.10, dp_clip_norm=1.0)
    k = max(1, int(0.10 * sum(g.numel() for g in gd.values())))
    protected, meta = enc.encrypt(gd, k=k)
    # fc.bias and fc.weight are forced into the encrypted set by the
    # default classifier-head guarantee, so both should be all-zero
    # in the protected dict.
    assert _is_layer_zeroed(protected, "fc.bias"), (
        "fc.bias was not zeroed by SelectiveShield — the classifier-"
        "head guarantee that V4 inherits is missing."
    )
    assert _is_layer_zeroed(protected, "fc.weight")
    assert "fc.bias" in meta["classifier_head_forced_layers"]


def test_non_encrypted_slice_gets_dp_noise():
    """Non-encrypted (Fisher-low) parameters must be perturbed by DP
    noise, not left as-is. Otherwise SelectiveShield collapses to V4
    (Fisher-only) and the comparison is meaningless."""
    gd = _adversarial_grad_dict()
    enc = SelectiveShieldEncryptor(
        enc_pct=0.10, dp_epsilon=10.0, dp_clip_norm=1.0,  # smaller eps -> bigger noise
    )
    k = max(1, int(0.10 * sum(g.numel() for g in gd.values())))
    protected, meta = enc.encrypt(gd, k=k)

    # The big conv layer holds the bulk of the non-encrypted slice
    # (most of its 5000 params are not in the top-10% by Fisher
    # because there's a head-guarantee carve-out). Verify that at
    # least SOME entries have been perturbed.
    layer = "layer1.0.conv1.weight"
    diff = (gd[layer] - protected[layer]).abs()
    n_perturbed = int((diff > 1e-12).sum().item())
    assert n_perturbed >= 1, (
        f"Expected at least one perturbed entry in {layer}; got "
        f"{n_perturbed}. The DP-noise pass is not firing."
    )
    # And the metadata reports a non-zero noise norm.
    assert meta["dp_noise_norm"] > 0.0


def test_metadata_has_both_fisher_and_dp_fields():
    """JSON sweep parsing assumes the SelectiveShield metadata has the
    union of Fisher fields (mask, threshold, head guarantee) and DP
    fields (epsilon, delta, sigma, clip)."""
    gd = _adversarial_grad_dict()
    enc = SelectiveShieldEncryptor()
    _, meta = enc.encrypt(gd, k=int(0.10 * sum(g.numel() for g in gd.values())))
    # Fisher side
    for key in ("strategy", "weights_encrypted", "pct_encrypted",
                "encryption_threshold", "classifier_head_forced_count",
                "classifier_head_forced_layers"):
        assert key in meta, f"Fisher metadata key '{key}' missing"
    # DP side
    for key in ("dp_epsilon", "dp_delta", "dp_clip_norm", "dp_sigma",
                "dp_clip_factor", "dp_noise_norm"):
        assert key in meta, f"DP metadata key '{key}' missing"
    assert meta["strategy"] == "selectiveshield"


def test_dp_sigma_responds_to_epsilon():
    """Calibration sanity: smaller epsilon -> bigger sigma."""
    gd = _adversarial_grad_dict()
    e_small = SelectiveShieldEncryptor(dp_epsilon=1.0)
    e_large = SelectiveShieldEncryptor(dp_epsilon=100.0)
    _, m_small = e_small.encrypt(gd, k=int(0.10 * sum(g.numel() for g in gd.values())))
    _, m_large = e_large.encrypt(gd, k=int(0.10 * sum(g.numel() for g in gd.values())))
    assert m_small["dp_sigma"] > m_large["dp_sigma"], (
        f"Smaller eps should produce larger sigma; got "
        f"sigma(eps=1)={m_small['dp_sigma']:.4f} "
        f"<= sigma(eps=100)={m_large['dp_sigma']:.4f}"
    )


def test_disabled_head_guarantee_drops_fc_bias():
    """Symmetric to the FisherEncryptor pre-fix test: with head
    guarantee disabled (mandatory_layer_substrings=()), fc.bias
    should NOT be zeroed on the adversarial fixture (Fisher score
    aggregate is too small to make the global top-K cut)."""
    gd = _adversarial_grad_dict()
    pre_fix_metric = FisherInformationMetric(
        enc_pct=0.10, mandatory_layer_substrings=(),
    )
    enc = SelectiveShieldEncryptor(fisher_metric=pre_fix_metric)
    protected, meta = enc.encrypt(
        gd, k=int(0.10 * sum(g.numel() for g in gd.values())),
    )
    # fc.bias is NOT all-zero (it's in the non-encrypted slice and got
    # DP-perturbed instead of zeroed).
    assert not torch.allclose(
        protected["fc.bias"], torch.zeros_like(gd["fc.bias"])
    ), (
        "Pre-fix SelectiveShield unexpectedly zeroed fc.bias — the "
        "adversarial fixture must be re-tuned."
    )
    assert meta["classifier_head_forced_count"] == 0


def test_overhead_estimate_matches_he_factor():
    """Communication overhead model: overhead = 1 + pct * (50 - 1).
    DP noise does not change the wire format, so it doesn't enter
    the cost model."""
    gd = _adversarial_grad_dict()
    enc = SelectiveShieldEncryptor()
    _, meta = enc.encrypt(gd, k=int(0.10 * sum(g.numel() for g in gd.values())))
    overhead = enc.compute_overhead(meta)
    expected = 1.0 + meta["pct_encrypted"] * (50.0 - 1.0)
    assert abs(overhead - expected) < 1e-9


if __name__ == "__main__":
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
