try:
    from opacus.accountants.rdp import RDPAccountant
    from opacus.accountants.utils import get_noise_multiplier
except Exception as e:
    RDPAccountant = None
    get_noise_multiplier = None

class FederatedPrivacyAuditor:
    def __init__(self, target_delta=1e-5):
        if RDPAccountant is None:
            raise RuntimeError('Opacus not available; install opacus for privacy accounting')
        self.accountant = RDPAccountant()
        self.target_delta = target_delta

    def step(self, noise_multiplier, sample_rate, steps=1):
        for _ in range(steps):
            self.accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

    def get_epsilon(self):
        return self.accountant.get_epsilon(delta=self.target_delta)
