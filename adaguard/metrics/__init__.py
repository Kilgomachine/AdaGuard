from .entropy_leakscore import EntropyLeakScoreMetric
from .label_leakscore import GLMIPMetric, ConfidenceGapMetric, CosineSimilarityMetric
from .fisher import FisherInformationMetric
from .maskcrypt import MaskCryptMetric
from .empirical_leakscore import EmpiricalLeakScoreMetric
from .combined import CombinedLeakScore

LABEL_METRICS = ['glmip_score', 'confidence_gap', 'cosine_leak_score']
ENTROPY_METRICS = ['shannon_leak_score', 'renyi_leak_score', 'min_entropy_leak_score']
EMPIRICAL_METRICS = ['empirical_gradinversion', 'empirical_ginas', 'empirical_ggcdm']

# Training-time metrics (fast, computed every round)
TRAINING_METRICS = LABEL_METRICS + ENTROPY_METRICS

# Attack metrics (expensive, Phase 2 only)
ATTACK_METRICS = EMPIRICAL_METRICS

ALL_LEAK_METRICS = LABEL_METRICS + ENTROPY_METRICS + EMPIRICAL_METRICS
EXTRA_METRICS = [
    'fisher_concentration', 'fisher_round',
    'maskcrypt_vuln_score', 'maskcrypt_l2_score', 'pct_encrypted',
    'shannon_full', 'renyi_full', 'min_entropy_full',
]
