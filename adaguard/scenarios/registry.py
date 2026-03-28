"""Scenario definitions for Phase 2 post-training attack evaluation.

Each scenario specifies:
  - encryption: which encryption strategy to apply
  - label_weight, entropy_weight, empirical_weight: LeakScore component weights
  - T1, T2: AdaptiveEncryptionController thresholds (for AdaGuard scenarios)

Weight convention (matches user specification):
  label_weight  * mean(GLMIP, ConfidenceGap, CosineSimilarity)
  entropy_weight * mean(Shannon, Renyi, MinEntropy)
  empirical_weight * mean(GradInversion, GI-NAS, GGCDM) lightweight scores
"""

SCENARIOS = {
    # ── Baselines ──
    'no_encryption': {
        'id': 1,
        'encryption': 'none',
        'description': 'No encryption — full gradient exposure (worst case for privacy)',
    },
    'full_he': {
        'id': 2,
        'encryption': 'full',
        'description': 'Full homomorphic encryption — all parameters encrypted (best privacy, worst cost)',
    },
    'maskcrypt': {
        'id': 3,
        'encryption': 'maskcrypt',
        'description': 'MaskCrypt per paper (Hu & Li 2025) — gradient-guided selective HE',
        'maskcrypt_enc_pct': 0.1,  # paper default: 10% of parameters
    },

    # ── AdaGuard threat levels ──
    'adaguard_high': {
        'id': 4,
        'encryption': 'fisher',
        'description': 'AdaGuard high threat sensitivity — encrypts aggressively',
        'T1': 0.1, 'T2': 0.4,
        'label_weight': 1.0, 'entropy_weight': 1.0, 'empirical_weight': 0.0,
    },
    'adaguard_mid': {
        'id': 5,
        'encryption': 'fisher',
        'description': 'AdaGuard medium threat sensitivity — balanced',
        'T1': 0.3, 'T2': 0.7,
        'label_weight': 1.0, 'entropy_weight': 1.0, 'empirical_weight': 0.0,
    },
    'adaguard_low': {
        'id': 6,
        'encryption': 'fisher',
        'description': 'AdaGuard low threat sensitivity — minimal encryption',
        'T1': 0.5, 'T2': 0.9,
        'label_weight': 1.0, 'entropy_weight': 1.0, 'empirical_weight': 0.0,
    },

    # ── AdaGuard component ablation ──
    'label_only': {
        'id': 7,
        'encryption': 'fisher',
        'description': 'AdaGuard with only label leakscore (GLMIP + ConfGap + Cosine)',
        'label_weight': 1.0, 'entropy_weight': 0.0, 'empirical_weight': 0.0,
    },
    'entropy_only': {
        'id': 8,
        'encryption': 'fisher',
        'description': 'AdaGuard with only entropy leakscore (Shannon + Renyi + MinEntropy)',
        'label_weight': 0.0, 'entropy_weight': 1.0, 'empirical_weight': 0.0,
    },
    'empirical_only': {
        'id': 9,
        'encryption': 'fisher',
        'description': 'AdaGuard with only empirical leakscore (light 20-iter GI attacks)',
        'label_weight': 0.0, 'entropy_weight': 0.0, 'empirical_weight': 1.0,
    },

    # ── AdaGuard weighted combinations ──
    '2label_1entropy': {
        'id': 10,
        'encryption': 'fisher',
        'description': 'AdaGuard: 2x label + 1x entropy (label-heavy)',
        'label_weight': 2.0, 'entropy_weight': 1.0, 'empirical_weight': 0.0,
    },
    '1label_2entropy': {
        'id': 11,
        'encryption': 'fisher',
        'description': 'AdaGuard: 1x label + 2x entropy (entropy-heavy)',
        'label_weight': 1.0, 'entropy_weight': 2.0, 'empirical_weight': 0.0,
    },
}


def get_scenario(name):
    """Get scenario config by name. Raises KeyError if not found."""
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]


def list_scenarios():
    """Print all available scenarios."""
    print(f"\n{'='*80}")
    print(f"  Available Scenarios ({len(SCENARIOS)} total)")
    print(f"{'='*80}")
    for name, cfg in sorted(SCENARIOS.items(), key=lambda x: x[1]['id']):
        enc = cfg.get('encryption', '?')
        lw = cfg.get('label_weight', '-')
        ew = cfg.get('entropy_weight', '-')
        empw = cfg.get('empirical_weight', '-')
        t1 = cfg.get('T1', '-')
        t2 = cfg.get('T2', '-')
        print(f"  {cfg['id']:2d}. {name:25s} | enc={enc:10s} | "
              f"L={lw} E={ew} Emp={empw} | T1={t1} T2={t2}")
        print(f"      {cfg['description']}")
    print(f"{'='*80}\n")
