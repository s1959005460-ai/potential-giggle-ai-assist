import logging
from collections import defaultdict

class EpsilonAudit:
    def __init__(self, max_epsilon=10.0, mode='conservative'):
        self.max_epsilon = max_epsilon
        self.mode = mode
        self.client_eps = defaultdict(float)
        self.client_reports = defaultdict(list)

    def add_report(self, client_id, epsilon, delta, round_idx, params=None):
        # params may contain noise_multiplier, sample_rate, steps for server-side verification
        if self.mode == 'conservative':
            self.client_eps[client_id] += float(epsilon)
        else:
            # placeholder for RDP/advanced accounting integration
            self.client_eps[client_id] += float(epsilon)
        self.client_reports[client_id].append((round_idx, float(epsilon), float(delta)))
        logging.info(f"[EpsilonAudit] {client_id} eps sum={self.client_eps[client_id]:.4f}")
        return self.client_eps[client_id] <= self.max_epsilon

    def get_eps(self, client_id):
        return self.client_eps.get(client_id, 0.0)
