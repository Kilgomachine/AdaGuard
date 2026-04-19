"""Scenario definitions for AdaGuard evaluation.

Case 1 — Sensitivity Study (60 scenarios):
  Sweep one configurable variable at a time, everything else at defaults.
  Groups: S1-S11 (entropy_bins, grad_accum_K, T1, T2, label/entropy/empirical
  weights, batch_size, samples_per_class, maskcrypt_enc_pct, focus_layers).

Case 2 — Viability Study (12 scenarios):
  Head-to-head comparisons proving AdaGuard works vs competitors.
  Baselines (no defense, full HE, DP), MaskCrypt (guided + random), AdaGuard
  variants, scalability tests.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CASE 1: SENSITIVITY STUDY
# ═══════════════════════════════════════════════════════════════════════════════

SENSITIVITY_SCENARIOS = {}


def _add_sensitivity_group(group_name, variable, values, defaults=None):
    """Helper to generate a sweep group."""
    for i, val in enumerate(values, 1):
        sid = f"{group_name}.{i}"
        overrides = {variable: val}
        if defaults:
            overrides.update(defaults)
        SENSITIVITY_SCENARIOS[sid] = {
            'id': sid,
            'case': 'sensitivity',
            'group': group_name,
            'variable': variable,
            'value': val,
            'description': f'{variable}={val}',
            'config_overrides': overrides,
        }


# S1. Entropy Bins
_add_sensitivity_group('S1', 'entropy_bins', [10, 25, 50, 100, 200])

# S2. Gradient Accumulation K
_add_sensitivity_group('S2', 'grad_accum_K', [1, 2, 4, 8])

# S3. Encryption Threshold T1 (T2 fixed at 0.7)
_add_sensitivity_group('S3', 'T1', [0.1, 0.2, 0.3, 0.4, 0.5],
                       defaults={'T2': 0.7})

# S4. Encryption Threshold T2 (T1 fixed at 0.3)
_add_sensitivity_group('S4', 'T2', [0.5, 0.6, 0.7, 0.8, 0.9],
                       defaults={'T1': 0.3})

# S5. Label Weight alpha
_add_sensitivity_group('S5', 'label_weight', [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                       defaults={'entropy_weight': 1.0, 'empirical_weight': 0.0})

# S6. Entropy Weight beta
_add_sensitivity_group('S6', 'entropy_weight', [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                       defaults={'label_weight': 1.0, 'empirical_weight': 0.0})

# S7. Empirical Weight gamma
_add_sensitivity_group('S7', 'empirical_weight', [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                       defaults={'label_weight': 1.0, 'entropy_weight': 1.0})

# S8. Client Batch Size
_add_sensitivity_group('S8', 'client_batch_size', [1, 4, 8, 16, 32])

# S9. GLMIP Samples Per Class
_add_sensitivity_group('S9', 'mi_samples_per_class', [5, 10, 20, 40])

# S10. MaskCrypt Encrypt Ratio rho (gradient-guided mode)
for i, rho in enumerate([0.01, 0.05, 0.10, 0.20, 0.25], 1):
    sid = f"S10.{i}"
    SENSITIVITY_SCENARIOS[sid] = {
        'id': sid,
        'case': 'sensitivity',
        'group': 'S10',
        'variable': 'maskcrypt_enc_pct',
        'value': rho,
        'description': f'MaskCrypt rho={rho}',
        'config_overrides': {
            'encryption_top_percent': rho,
            'encryption': 'maskcrypt',
            'maskcrypt_mask_mode': 'gradient_guided',
        },
    }

# S11. Focus Layers for GLMIP (ResNet-18 layer names)
# The old fc1/fc2 values were leftover from a smallcnn baseline and don't
# exist in ResNet-18 — would silently match 0 params and give garbage.
#   S11.1: None       — all layers, broad/noisy label signal
#   S11.2: fc.*       — final classifier (iDLG-style), strongest per-theory
#   S11.3: layer4.1.* — last residual block, tests mid-layer leak
_focus_layer_values = [
    ('all', None),
    ('final_fc', ['fc.weight', 'fc.bias']),
    ('penultimate_conv', ['layer4.1.conv2.weight']),
]
for i, (label, layers) in enumerate(_focus_layer_values, 1):
    sid = f"S11.{i}"
    SENSITIVITY_SCENARIOS[sid] = {
        'id': sid,
        'case': 'sensitivity',
        'group': 'S11',
        'variable': 'focus_layers',
        'value': label,
        'description': f'focus_layers={label}',
        'config_overrides': {
            'focus_layers': layers,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CASE 2: VIABILITY STUDY
# ═══════════════════════════════════════════════════════════════════════════════

VIABILITY_SCENARIOS = {
    # -- Baselines --
    'V1': {
        'id': 'V1',
        'case': 'viability',
        'group': 'baselines',
        'name': 'no_defense',
        'description': 'No encryption — full gradient exposure (worst case for privacy)',
        'config_overrides': {
            'encryption': 'none',
        },
    },
    'V2': {
        'id': 'V2',
        'case': 'viability',
        'group': 'baselines',
        'name': 'full_he',
        'description': 'Full homomorphic encryption — all parameters encrypted (best privacy, worst cost)',
        'config_overrides': {
            'encryption': 'full',
        },
    },
    'V3': {
        'id': 'V3',
        'case': 'viability',
        'group': 'baselines',
        'name': 'dp_baseline',
        'description': 'Differential privacy (epsilon=50, delta=1e-5) — industry standard alternative',
        'config_overrides': {
            'encryption': 'dp',
            'dp_epsilon': 50.0,
            'dp_delta': 1e-5,
            'dp_clip_norm': 1.0,
        },
    },

    # -- MaskCrypt Competitor --
    'V4': {
        'id': 'V4',
        'case': 'viability',
        'group': 'maskcrypt',
        'name': 'maskcrypt_guided_10',
        'description': 'MaskCrypt gradient-guided (rho=0.10) — Hu & Li 2025 paper default',
        'config_overrides': {
            'encryption': 'maskcrypt',
            'encryption_top_percent': 0.10,
            'maskcrypt_mask_mode': 'gradient_guided',
        },
    },
    'V5': {
        'id': 'V5',
        'case': 'viability',
        'group': 'maskcrypt',
        'name': 'maskcrypt_random_10',
        'description': 'MaskCrypt random mask (rho=0.10) — naive baseline for mask selection',
        'config_overrides': {
            'encryption': 'maskcrypt',
            'encryption_top_percent': 0.10,
            'maskcrypt_mask_mode': 'random',
        },
    },

    # -- AdaGuard variants --
    'V6': {
        'id': 'V6',
        'case': 'viability',
        'group': 'adaguard',
        'name': 'adaguard_default',
        'description': 'AdaGuard balanced (T1=0.3, T2=0.7)',
        'config_overrides': {
            'encryption': 'fisher',
            'T1': 0.3, 'T2': 0.7,
            'label_weight': 1.0, 'entropy_weight': 1.0, 'empirical_weight': 0.0,
            'grad_accum_enabled': False,
        },
    },
    'V7': {
        'id': 'V7',
        'case': 'viability',
        'group': 'adaguard',
        'name': 'adaguard_with_accum',
        'description': 'AdaGuard + gradient accumulation (K=4, B_eff=16)',
        'config_overrides': {
            'encryption': 'fisher',
            'T1': 0.3, 'T2': 0.7,
            'label_weight': 1.0, 'entropy_weight': 1.0, 'empirical_weight': 0.0,
            'grad_accum_enabled': True, 'grad_accum_K': 4,
        },
    },
    'V8': {
        'id': 'V8',
        'case': 'viability',
        'group': 'adaguard',
        'name': 'adaguard_aggressive',
        'description': 'AdaGuard aggressive (T1=0.1, T2=0.4) — encrypt early, encrypt often',
        'config_overrides': {
            'encryption': 'fisher',
            'T1': 0.1, 'T2': 0.4,
            'label_weight': 1.0, 'entropy_weight': 1.0, 'empirical_weight': 0.0,
        },
    },
    'V9': {
        'id': 'V9',
        'case': 'viability',
        'group': 'adaguard',
        'name': 'adaguard_conservative',
        'description': 'AdaGuard conservative (T1=0.5, T2=0.9) — minimal encryption',
        'config_overrides': {
            'encryption': 'fisher',
            'T1': 0.5, 'T2': 0.9,
            'label_weight': 1.0, 'entropy_weight': 1.0, 'empirical_weight': 0.0,
        },
    },

    # -- Scalability --
    'V10': {
        'id': 'V10',
        'case': 'viability',
        'group': 'scalability',
        'name': 'adaguard_100_clients',
        'description': 'AdaGuard at 100 clients (10% participation)',
        'config_overrides': {
            'encryption': 'fisher',
            'T1': 0.3, 'T2': 0.7,
            'num_clients': 100, 'clients_per_round': 10,
        },
    },
    'V11': {
        'id': 'V11',
        'case': 'viability',
        'group': 'scalability',
        'name': 'adaguard_1000_clients',
        'description': 'AdaGuard at 1000 clients (10% participation)',
        'config_overrides': {
            'encryption': 'fisher',
            'T1': 0.3, 'T2': 0.7,
            'num_clients': 1000, 'clients_per_round': 100,
        },
    },
    'V12': {
        'id': 'V12',
        'case': 'viability',
        'group': 'scalability',
        'name': 'adaguard_high_participation',
        'description': 'AdaGuard 50% participation rate (100 clients, 50 per round)',
        'config_overrides': {
            'encryption': 'fisher',
            'T1': 0.3, 'T2': 0.7,
            'num_clients': 100, 'clients_per_round': 50,
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED REGISTRY + HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

ALL_SCENARIOS = {}
ALL_SCENARIOS.update(SENSITIVITY_SCENARIOS)
ALL_SCENARIOS.update(VIABILITY_SCENARIOS)

# Group index for quick lookups
SENSITIVITY_GROUPS = {}
for sid, cfg in SENSITIVITY_SCENARIOS.items():
    grp = cfg['group']
    if grp not in SENSITIVITY_GROUPS:
        SENSITIVITY_GROUPS[grp] = []
    SENSITIVITY_GROUPS[grp].append(sid)

VIABILITY_GROUPS = {}
for vid, cfg in VIABILITY_SCENARIOS.items():
    grp = cfg['group']
    if grp not in VIABILITY_GROUPS:
        VIABILITY_GROUPS[grp] = []
    VIABILITY_GROUPS[grp].append(vid)


def get_scenario(scenario_id):
    """Get scenario config by ID (e.g. 'S1.1', 'V4'). Raises KeyError if not found."""
    if scenario_id not in ALL_SCENARIOS:
        raise KeyError(
            f"Unknown scenario '{scenario_id}'. "
            f"Available: {sorted(ALL_SCENARIOS.keys())}"
        )
    return ALL_SCENARIOS[scenario_id]


def get_scenarios_by_case(case):
    """Get all scenarios for a case ('sensitivity' or 'viability')."""
    if case == 'sensitivity':
        return dict(SENSITIVITY_SCENARIOS)
    elif case == 'viability':
        return dict(VIABILITY_SCENARIOS)
    else:
        raise ValueError(f"Unknown case '{case}'. Use 'sensitivity' or 'viability'.")


def get_sensitivity_group(group_id):
    """Get all scenario IDs in a sensitivity group (e.g. 'S1')."""
    if group_id not in SENSITIVITY_GROUPS:
        raise KeyError(
            f"Unknown sensitivity group '{group_id}'. "
            f"Available: {sorted(SENSITIVITY_GROUPS.keys())}"
        )
    return [ALL_SCENARIOS[sid] for sid in SENSITIVITY_GROUPS[group_id]]


def list_scenarios():
    """Print all available scenarios organized by case and group."""
    print(f"\n{'=' * 90}")
    print(f"  AdaGuard Scenarios ({len(ALL_SCENARIOS)} total)")
    print(f"{'=' * 90}")

    # Sensitivity
    print(f"\n  CASE 1: SENSITIVITY STUDY ({len(SENSITIVITY_SCENARIOS)} scenarios)")
    print(f"  {'-' * 86}")
    for grp_id in sorted(SENSITIVITY_GROUPS.keys()):
        sids = SENSITIVITY_GROUPS[grp_id]
        first = ALL_SCENARIOS[sids[0]]
        var = first['variable']
        vals = [str(ALL_SCENARIOS[s]['value']) for s in sids]
        print(f"  {grp_id:5s} | {var:25s} | {len(sids)} runs: [{', '.join(vals)}]")

    # Viability
    print(f"\n  CASE 2: VIABILITY STUDY ({len(VIABILITY_SCENARIOS)} scenarios)")
    print(f"  {'-' * 86}")
    for grp_id in sorted(VIABILITY_GROUPS.keys()):
        vids = VIABILITY_GROUPS[grp_id]
        print(f"  {grp_id}:")
        for vid in vids:
            cfg = ALL_SCENARIOS[vid]
            name = cfg.get('name', vid)
            print(f"    {vid:5s} | {name:30s} | {cfg['description']}")

    print(f"\n{'=' * 90}\n")
