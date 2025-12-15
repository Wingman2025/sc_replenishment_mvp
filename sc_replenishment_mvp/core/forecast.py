from dataclasses import dataclass
from collections import deque

@dataclass
class MovingAverageForecast:
    window: int = 4

    def reset(self, initial_demand: int):
        self.hist = deque([initial_demand] * self.window, maxlen=self.window)

    def observe(self, demand: int):
        self.hist.append(demand)

    def predict(self) -> float:
        return sum(self.hist) / len(self.hist)

    def clone(self):
        f = MovingAverageForecast(window=self.window)
        # si no hay hist todavía, la inicializamos en reset()
        if hasattr(self, "hist"):
            f.hist = deque(self.hist, maxlen=self.window)
        return f

@dataclass
class ExponentialMovingAverageForecast:
    beta: float = 0.3  # 0.05 muy suave, 0.8 reacciona rápido

    def reset(self, initial_demand: int):
        self.level = float(initial_demand)

    def observe(self, demand: int):
        self.level = (1.0 - self.beta) * self.level + self.beta * float(demand)

    def predict(self) -> float:
        return float(self.level)

    def clone(self):
        f = ExponentialMovingAverageForecast(beta=self.beta)
        if hasattr(self, "level"):
            f.level = float(self.level)
        return f
