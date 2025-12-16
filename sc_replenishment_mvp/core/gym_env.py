"""
================================================================================
ENTORNO RL PARA REINFORCEMENT LEARNING - gym_env.py
================================================================================

¿QUÉ HACE ESTE ARCHIVO?
-----------------------
Este archivo "envuelve" nuestro simulador de DC en un formato que entienden
los algoritmos de Reinforcement Learning (RL). Específicamente, usa el
estándar "Gymnasium" (antes conocido como OpenAI Gym).

NIVEL 3 - OPCIONAL
------------------
Este archivo es OPCIONAL. Solo lo necesitas si quieres experimentar con
Reinforcement Learning. Si solo usas la política Base-Stock (Nivel 1/2),
puedes ignorar este archivo completamente.

¿QUÉ ES REINFORCEMENT LEARNING (RL)?
------------------------------------
RL es una técnica de IA donde un "agente" aprende a tomar decisiones
probando diferentes acciones y recibiendo "recompensas" o "castigos".

Analogía simple:
- Imagina entrenar a un perro: le das un premio cuando hace algo bien
- El perro aprende qué acciones llevan a premios
- RL funciona igual: el algoritmo prueba acciones y aprende de los resultados

En nuestro caso:
- AGENTE: El algoritmo que decide cuánto pedir
- ESTADO: Información del almacén (inventario, pipeline, forecast, etc.)
- ACCIÓN: Cuánto pedir esta semana
- RECOMPENSA: Negativo del coste (queremos minimizar coste = maximizar recompensa)

¿POR QUÉ GYMNASIUM?
-------------------
Gymnasium es un estándar de la industria para entornos RL. Si usas este
estándar, puedes aprovechar algoritmos ya implementados (como PPO, DQN, etc.)
sin tener que programarlos tú mismo.

REQUISITOS:
-----------
Para usar este archivo necesitas instalar:
  pip install gymnasium stable-baselines3

================================================================================
"""

# ------------------------------------------------------------------------------
# IMPORTACIONES
# ------------------------------------------------------------------------------
# 'annotations' permite usar tipos de Python más modernos
from __future__ import annotations

# 'random' para generar demanda aleatoria
import random

# 'numpy' es una librería para cálculos numéricos
# RL usa arrays numpy para los estados y observaciones
import numpy as np

# Intentamos importar Gymnasium
# Si no está instalado, las variables serán None (y daremos error si se intenta usar)
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    gym = None
    spaces = None


# ==============================================================================
# CLASE PRINCIPAL: DCReplenishmentEnv
# ==============================================================================

class DCReplenishmentEnv:
    """
    ENTORNO RL PARA EL PROBLEMA DE REPLENISHMENT
    =============================================

    Este entorno permite entrenar un agente RL para decidir cuánto pedir
    cada semana en un Centro de Distribución.

    CONCEPTOS CLAVE (para entender RL):
    -----------------------------------

    1. ESTADO (Observation):
       Lo que el agente "ve" para tomar su decisión.
       En nuestro caso: [on_hand, pipe_1, pipe_2, d_hat, q_prev]
       - on_hand: Inventario actual
       - pipe_1: Lo que llega en 1 semana
       - pipe_2: Lo que llega en 2 semanas
       - d_hat: Forecast de demanda (promedio de últimas semanas)
       - q_prev: Pedido de la semana anterior

    2. ACCIÓN (Action):
       Lo que el agente decide hacer.
       En lugar de elegir un número exacto (infinitas opciones),
       el agente elige un "multiplicador" sobre el forecast.

       Ejemplo con multipliers = [0.0, 0.5, 1.0, 1.5, 2.0]:
       - Acción 0 → pedir 0.0 × d_hat = 0 (no pedir nada)
       - Acción 1 → pedir 0.5 × d_hat (pedir la mitad del forecast)
       - Acción 2 → pedir 1.0 × d_hat (pedir exactamente el forecast)
       - Acción 3 → pedir 1.5 × d_hat (pedir 50% más)
       - Acción 4 → pedir 2.0 × d_hat (pedir el doble)

    3. RECOMPENSA (Reward):
       El "feedback" que recibe el agente después de cada acción.
       Reward = -coste (negativo porque RL MAXIMIZA y nosotros queremos MINIMIZAR)

       Si el coste es 500, reward = -500
       El agente aprenderá a elegir acciones que den rewards menos negativos.

    4. EPISODIO:
       Una "partida" completa de T semanas.
       Al terminar (terminated=True), se reinicia el entorno.

    ¿CÓMO SE USA?
    -------------
    Ver el archivo rl_train.py para un ejemplo completo de entrenamiento.
    """

    def __init__(
        self,
        L: int = 2,                              # Lead time (semanas)
        T: int = 52,                             # Duración del episodio (semanas)
        demand_min: int = 50,                    # Demanda mínima
        demand_max: int = 120,                   # Demanda máxima
        multipliers=(0.0, 0.5, 1.0, 1.5, 2.0),  # Opciones de multiplicador
        h: float = 1.0,                          # Coste holding
        p: float = 20.0,                         # Coste penalización
        s: float = 0.1,                          # Coste smoothing
        window: int = 4,                         # Ventana para forecast
        seed: int = 1,                           # Semilla aleatoria
        init_on_hand: int = 250,                 # Inventario inicial
    ):
        """
        Inicializa el entorno RL.

        Los parámetros son los mismos que en el simulador normal,
        más 'multipliers' que define las acciones disponibles.
        """

        # Verificar que Gymnasium esté instalado
        if gym is None:
            raise ImportError("gymnasium no está instalado. Instala con: pip install gymnasium")

        # Guardar todos los parámetros
        self.L = int(L)
        self.T = int(T)
        self.demand_min = int(demand_min)
        self.demand_max = int(demand_max)
        self.multipliers = list(multipliers)
        self.h = float(h)
        self.p = float(p)
        self.s = float(s)
        self.window = int(window)

        # Inicializar el generador de números aleatorios
        self._rng = random.Random(seed)
        self._base_seed = seed
        self._init_on_hand = int(init_on_hand)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # DEFINIR LOS "ESPACIOS" (requerido por Gymnasium)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Espacio de ACCIONES: discreto, tantas opciones como multiplicadores
        # Por ejemplo, si hay 5 multiplicadores, las acciones son 0, 1, 2, 3, 4
        self.action_space = spaces.Discrete(len(self.multipliers))

        # Espacio de OBSERVACIONES: 5 números continuos
        # Definimos límites (low, high) para cada componente
        high = np.array([1e6, 1e6, 1e6, 1e6, 1e6], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=high,
            dtype=np.float32
        )

        # Reiniciar el entorno para inicializar el estado
        self.reset(seed=seed)

    # --------------------------------------------------------------------------
    # MÉTODO: reset
    # --------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno para comenzar un nuevo episodio.

        Este método se llama:
        - Al inicio del entrenamiento
        - Cada vez que termina un episodio (después de T semanas)

        RETORNA:
        - obs: El estado inicial (observación)
        - info: Diccionario con información adicional (vacío por ahora)
        """
        # Reiniciar el generador aleatorio si se proporciona nueva semilla
        if seed is not None:
            self._rng = random.Random(int(seed))

        # Reiniciar el tiempo
        self.t = 0

        # Reiniciar el inventario al valor inicial
        self.on_hand = self._init_on_hand

        # Reiniciar el pipeline (todo vacío)
        self.pipe = [0 for _ in range(self.L)]

        # Reiniciar el pedido anterior
        self.q_prev = 0

        # Inicializar el historial de demanda (para calcular d_hat)
        d0 = int((self.demand_min + self.demand_max) / 2)
        self.d_hist = [d0] * self.window

        # Construir y retornar la observación inicial
        obs = self._obs()
        info = {}
        return obs, info

    # --------------------------------------------------------------------------
    # MÉTODO AUXILIAR: _d_hat
    # --------------------------------------------------------------------------
    def _d_hat(self):
        """
        Calcula el forecast de demanda (media móvil simple).

        Es el promedio de las últimas 'window' demandas observadas.
        """
        return sum(self.d_hist) / len(self.d_hist)

    # --------------------------------------------------------------------------
    # MÉTODO AUXILIAR: _obs
    # --------------------------------------------------------------------------
    def _obs(self):
        """
        Construye el vector de observación (estado) actual.

        El agente RL usa esta información para decidir qué acción tomar.
        Es un array numpy con 5 números:
        [on_hand, pipe_1, pipe_2, d_hat, q_prev]
        """
        d_hat = self._d_hat()
        pipe1 = self.pipe[0] if self.L >= 1 else 0
        pipe2 = self.pipe[1] if self.L >= 2 else 0
        return np.array([self.on_hand, pipe1, pipe2, d_hat, self.q_prev], dtype=np.float32)

    # --------------------------------------------------------------------------
    # MÉTODO: step (EL CORAZÓN DEL ENTORNO)
    # --------------------------------------------------------------------------
    def step(self, action):
        """
        Ejecuta UNA semana de simulación dado la acción elegida.

        Este método es el núcleo del entorno RL. Cada vez que se llama:
        1. Convierte la acción en un pedido
        2. Simula la semana (llegadas, demanda, ventas)
        3. Calcula el coste y la recompensa
        4. Retorna el nuevo estado

        PARÁMETROS:
        - action: Un entero (0, 1, 2, ...) indicando qué multiplicador usar

        RETORNA (tupla de 5 elementos):
        - obs: El nuevo estado después de la acción
        - reward: La recompensa (negativo del coste)
        - terminated: True si el episodio terminó (llegamos a T semanas)
        - truncated: True si se cortó por otra razón (siempre False aquí)
        - info: Diccionario con información adicional para debugging
        """

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PASO 1: Convertir la acción en un pedido
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Obtenemos el multiplicador correspondiente a esta acción
        m = self.multipliers[int(action)]

        # Calculamos el pedido: multiplicador × forecast
        d_hat = self._d_hat()
        q_t = int(round(max(0.0, m * d_hat)))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PASO 2: Procesar llegadas del pipeline
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        arrivals = self.pipe[0]  # Lo que llega esta semana
        self.pipe = self.pipe[1:] + [q_t]  # Actualizar pipeline

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PASO 3: Generar y procesar la demanda
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        d_t = self._rng.randint(self.demand_min, self.demand_max)

        available = self.on_hand + arrivals
        served = min(available, d_t)
        lost = d_t - served
        self.on_hand = available - served

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PASO 4: Calcular el cambio en pedidos
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        abs_change = abs(q_t - self.q_prev)
        self.q_prev = q_t

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PASO 5: Calcular coste y recompensa
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        cost = self.h * self.on_hand + self.p * lost + self.s * abs_change

        # IMPORTANTE: La recompensa es el NEGATIVO del coste
        # RL maximiza recompensa, nosotros queremos minimizar coste
        reward = -cost

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PASO 6: Avanzar el tiempo y actualizar historial
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t += 1
        self.d_hist = self.d_hist[1:] + [d_t]  # Ventana deslizante

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PASO 7: Verificar si el episodio terminó
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        terminated = self.t >= self.T  # Terminamos si llegamos a T semanas
        truncated = False  # No usamos truncado en este entorno

        # Información adicional para debugging/análisis
        info = {
            "demand": d_t,
            "served": served,
            "lost": lost,
            "order_q": q_t,
            "cost": cost
        }

        return self._obs(), reward, terminated, truncated, info
