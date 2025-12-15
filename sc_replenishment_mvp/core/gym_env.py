"""
Opcional (Nivel 3): envolver el simulador como un entorno Gymnasium.

Solo lo necesitas si vas a entrenar RL con Stable-Baselines3.
Instala:
  pip install gymnasium stable-baselines3

Luego mira rl_train.py
"""

from __future__ import annotations

import random
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    gym = None
    spaces = None


class DCReplenishmentEnv:
    """
    Entorno RL minimalista:
    - Estado: [on_hand, pipe_1, pipe_2, d_hat, q_prev]
    - Acción discreta: elige un multiplicador m sobre d_hat → q = round(m*d_hat)
    """

    def __init__(
        self,
        L: int = 2,
        T: int = 52,
        demand_min: int = 50,
        demand_max: int = 120,
        multipliers=(0.0, 0.5, 1.0, 1.5, 2.0),
        h: float = 1.0,
        p: float = 20.0,
        s: float = 0.1,
        window: int = 4,
        seed: int = 1,
        init_on_hand: int = 250,
    ):
        if gym is None:
            raise ImportError("gymnasium no está instalado. Instala con: pip install gymnasium")

        self.L = int(L)
        self.T = int(T)
        self.demand_min = int(demand_min)
        self.demand_max = int(demand_max)
        self.multipliers = list(multipliers)
        self.h = float(h)
        self.p = float(p)
        self.s = float(s)
        self.window = int(window)

        self._rng = random.Random(seed)
        self._base_seed = seed
        self._init_on_hand = int(init_on_hand)

        # Espacios Gym
        self.action_space = spaces.Discrete(len(self.multipliers))
        # Observación: 5 números (acotamos con rangos amplios)
        high = np.array([1e6, 1e6, 1e6, 1e6, 1e6], dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros(5, dtype=np.float32), high=high, dtype=np.float32)

        self.reset(seed=seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = random.Random(int(seed))
        self.t = 0
        self.on_hand = self._init_on_hand
        self.pipe = [0 for _ in range(self.L)]
        self.q_prev = 0
        d0 = int((self.demand_min + self.demand_max) / 2)
        self.d_hist = [d0] * self.window

        obs = self._obs()
        info = {}
        return obs, info

    def _d_hat(self):
        return sum(self.d_hist) / len(self.d_hist)

    def _obs(self):
        d_hat = self._d_hat()
        pipe1 = self.pipe[0] if self.L >= 1 else 0
        pipe2 = self.pipe[1] if self.L >= 2 else 0
        return np.array([self.on_hand, pipe1, pipe2, d_hat, self.q_prev], dtype=np.float32)

    def step(self, action):
        # Acción → pedido
        m = self.multipliers[int(action)]
        d_hat = self._d_hat()
        q_t = int(round(max(0.0, m * d_hat)))

        # Llegadas
        arrivals = self.pipe[0]
        self.pipe = self.pipe[1:] + [q_t]

        # Demanda
        d_t = self._rng.randint(self.demand_min, self.demand_max)

        available = self.on_hand + arrivals
        served = min(available, d_t)
        lost = d_t - served
        self.on_hand = available - served

        abs_change = abs(q_t - self.q_prev)
        self.q_prev = q_t

        # Reward (negativo del coste)
        cost = self.h * self.on_hand + self.p * lost + self.s * abs_change
        reward = -cost

        # Avanza tiempo
        self.t += 1
        self.d_hist = self.d_hist[1:] + [d_t]

        terminated = self.t >= self.T
        truncated = False
        info = {"demand": d_t, "served": served, "lost": lost, "order_q": q_t, "cost": cost}

        return self._obs(), reward, terminated, truncated, info
