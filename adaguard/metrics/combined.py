"""Combined LeakScore: weighted combination of entropy, label, and empirical scores.

LeakScore_final = (alpha * LeakScore + beta * LabelLeakScore + gamma * EmpiricalLeakScore)
                  / (alpha + beta + gamma)

All inputs and output are in [0, 1].
"""


class CombinedLeakScore:
    """Weighted combination of three LeakScore components.

    Hyperparameters alpha, beta, gamma control the influence of each
    component and can be tuned for experiments.
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute(self, leak_score, label_leak_score, empirical_leak_score):
        """Compute weighted combined LeakScore.

        Args:
            leak_score: entropy-based LeakScore (float in [0,1])
            label_leak_score: label-based LeakScore (float in [0,1])
            empirical_leak_score: empirical attack-based LeakScore (float in [0,1])

        Returns:
            float in [0,1] — combined LeakScore
        """
        denominator = self.alpha + self.beta + self.gamma
        if denominator < 1e-12:
            return 0.0

        total = (
            self.alpha * leak_score
            + self.beta * label_leak_score
            + self.gamma * empirical_leak_score
        )
        return max(0.0, min(1.0, total / denominator))

    def update_weights(self, alpha=None, beta=None, gamma=None):
        """Update combination weights for experiment sweeps."""
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
