from dataclasses import dataclass
from typing import Any

@dataclass
class BaseStockPolicy:
    """
    Política base-stock / order-up-to:

    - Calcula un forecast d_hat (desde 'forecaster')
    - S = (L+1) * d_hat + safety
    - IP = on_hand + sum(pipe)
    - q_raw = max(0, S - IP)
    - q = suavizado con alpha para no cambiar tan bruscamente:
         q = (1-alpha)*q_prev + alpha*q_raw

    Nota: es una regla matemática “clásica”. No aprende por sí misma.
    """
    L: int
    safety: int
    alpha: float
    forecaster: Any

    def reset(self, initial_demand: int):
        self.forecaster.reset(initial_demand=initial_demand)

    def observe(self, demand: int):
        self.forecaster.observe(demand)

    def decide(self, on_hand: int, pipe: list[int], q_prev: int) -> int:
        d_hat = self.forecaster.predict()

        IP = on_hand + sum(pipe)
        S = (self.L + 1) * d_hat + self.safety
        q_raw = max(0.0, S - IP)

        q = (1.0 - self.alpha) * q_prev + self.alpha * q_raw
        q_int = int(round(q))
        if q_int < 0:
            q_int = 0
        return q_int

    def clone(self):
        # clonamos para que cada episodio tenga su propio estado interno de forecast
        return BaseStockPolicy(
            L=self.L,
            safety=self.safety,
            alpha=self.alpha,
            forecaster=self.forecaster.clone(),
        )
