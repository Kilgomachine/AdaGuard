"""Adaptive Encryption Controller — decides encryption policy based on LeakScore.

Implements tiered encryption:
  LeakScore < T1       → no encryption
  T1 <= LeakScore < T2 → partial (encrypt top-k sensitive parameters)
  LeakScore >= T2      → strong (encrypt larger fraction + validate with full GI)
"""

from dataclasses import dataclass


@dataclass
class EncryptionPolicy:
    """Encryption decision output."""
    level: str        # 'none', 'partial', 'strong'
    encrypt_pct: float  # fraction of parameters to encrypt [0, 1]
    run_full_gi: bool = False  # whether to run full GI attack for validation


class AdaptiveEncryptionController:
    """Decides encryption policy based on combined LeakScore.

    The encryption percentage scales linearly within each tier:
    - Below T1: no encryption (0%)
    - T1 to T2: scales from 0% to base_encrypt_pct
    - Above T2: scales from base_encrypt_pct toward 1.0
    """

    def __init__(self, T1=0.3, T2=0.7, base_encrypt_pct=0.1):
        self.T1 = T1
        self.T2 = T2
        self.base_encrypt_pct = base_encrypt_pct

    def decide(self, leak_score_final):
        """Determine encryption policy from combined LeakScore.

        Args:
            leak_score_final: combined LeakScore in [0, 1]

        Returns:
            EncryptionPolicy with level, encrypt_pct, and run_full_gi flag
        """
        if leak_score_final < self.T1:
            return EncryptionPolicy(level='none', encrypt_pct=0.0)

        if leak_score_final < self.T2:
            # Linear interpolation from 0 to base_encrypt_pct
            pct = self.base_encrypt_pct * (
                (leak_score_final - self.T1) / (self.T2 - self.T1)
            )
            return EncryptionPolicy(level='partial', encrypt_pct=pct)

        # Strong protection — scale up beyond base
        pct = self.base_encrypt_pct + (1.0 - self.base_encrypt_pct) * (
            (leak_score_final - self.T2) / (1.0 - self.T2)
            if self.T2 < 1.0 else 1.0
        )
        return EncryptionPolicy(
            level='strong',
            encrypt_pct=min(pct, 1.0),
            run_full_gi=True,
        )

    def compute_k(self, leak_score_final, total_params):
        """Compute dynamic k = number of parameters to encrypt.

        k = f(LeakScore) where higher leakage → more parameters encrypted.

        Args:
            leak_score_final: combined LeakScore in [0, 1]
            total_params: total number of model parameters

        Returns:
            int — number of parameters to encrypt
        """
        policy = self.decide(leak_score_final)
        return int(policy.encrypt_pct * total_params)
