import threading, logging

class RoundManager:
    def __init__(self, start_round=0):
        self._lock = threading.Lock()
        self.current_round = start_round

    def get_round(self):
        with self._lock:
            return self.current_round

    def increment(self):
        with self._lock:
            self.current_round += 1
            logging.info(f"Round incremented to {self.current_round}")
            return self.current_round

    def set_round(self, r):
        with self._lock:
            self.current_round = r
