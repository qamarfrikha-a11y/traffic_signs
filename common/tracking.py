from collections import deque, Counter


class TemporalFilter:
    """Vote majoritaire sur les N dernières frames (BNF4)."""
    def __init__(self, window=8, min_votes=4):
        self.hist = deque(maxlen=window)
        self.min_votes = min_votes
        self.current = None

    def update(self, label):
        self.hist.append(label)
        votes = Counter(l for l in self.hist if l)
        if votes:
            lab, n = votes.most_common(1)[0]
            if n >= self.min_votes:
                self.current = lab
        return self.current


class SpeedMonitor:
    """Mémorise la dernière limite et compare à la vitesse simulée (BF5)."""
    def __init__(self, vehicle_speed=50):
        self.speed_limit = None
        self.vehicle_speed = vehicle_speed

    def set_limit(self, limit):
        self.speed_limit = limit

    def change_speed(self, delta):
        self.vehicle_speed = max(0, min(200, self.vehicle_speed + delta))

    @property
    def alert(self):
        return self.speed_limit is not None and self.vehicle_speed > self.speed_limit