"""Combined LeakScore: weighted combination of entropy, label, and empirical scores.

LeakScore = (label_weight * LabelAvg + entropy_weight * EntropyAvg + empirical_weight * EmpiricalAvg)
            / (label_weight + entropy_weight + empirical_weight)

All inputs and output are in [0, 1].

Weight naming convention (matches user's specification):
  - label_weight: controls GLMIP + ConfidenceGap + CosineSimilarity influence
  - entropy_weight: controls Shannon + Renyi + MinEntropy influence
  - empirical_weight: controls lightweight GI attack scores influence
"""


class CombinedLeakScore:
    """Weighted combination of three LeakScore components.

    Hyperparameters label_weight, entropy_weight, empirical_weight control
    the influence of each component and can be tuned per scenario.
    """

    def __init__(self, label_weight=1.0, entropy_weight=1.0, empirical_weight=0.0):
        self.label_weight = label_weight
        self.entropy_weight = entropy_weight
        self.empirical_weight = empirical_weight

    def compute(self, entropy_avg, label_avg, empirical_avg=0.0):
        """Compute weighted combined LeakScore.

        Args:
            entropy_avg: entropy-based LeakScore (float in [0,1])
            label_avg: label-based LeakScore (float in [0,1])
            empirical_avg: empirical attack-based LeakScore (float in [0,1])

        Returns:
            float in [0,1] — combined LeakScore
        """
        denominator = self.label_weight + self.entropy_weight + self.empirical_weight
        if denominator < 1e-12:
            return 0.0

        total = (
            self.label_weight * label_avg
            + self.entropy_weight * entropy_avg
            + self.empirical_weight * empirical_avg
        )
        return max(0.0, min(1.0, total / denominator))

    def update_weights(self, label_weight=None, entropy_weight=None, empirical_weight=None):
        """Update combination weights for experiment sweeps."""
        if label_weight is not None:
            self.label_weight = label_weight
        if entropy_weight is not None:
            self.entropy_weight = entropy_weight
        if empirical_weight is not None:
            self.empirical_weight = empirical_weight

    # Backward compat aliases. Convention: alpha=label, beta=entropy,
    # gamma=empirical, matching the LeakScore equation in the paper
    # (sec. IV.B), the sensitivity table (S4-S6), and the scenario
    # registry comments.
    @property
    def alpha(self):
        return self.label_weight

    @property
    def beta(self):
        return self.entropy_weight

    @property
    def gamma(self):
        return self.empirical_weight
