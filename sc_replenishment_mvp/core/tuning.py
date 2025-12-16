"""
================================================================================
OPTIMIZACIÓN DE PARÁMETROS (TUNING) - tuning.py
================================================================================

¿QUÉ HACE ESTE ARCHIVO?
-----------------------
Este archivo contiene herramientas para encontrar los MEJORES PARÁMETROS
de la política de pedidos. En lugar de adivinar qué valores de 'safety' y
'alpha' usar, probamos muchas combinaciones y vemos cuál funciona mejor.

ANALOGÍA SIMPLE:
----------------
Imagina que quieres encontrar la mejor receta de galletas. Podrías:
- Opción A: Adivinar las cantidades de harina, azúcar, etc.
- Opción B: Probar MUCHAS combinaciones y ver cuál queda mejor

La "Opción B" es lo que hace este archivo, pero para la política de inventario.
Probamos diferentes valores de 'safety' y 'alpha', simulamos muchas veces
cada combinación, y reportamos cuál tuvo el menor coste.

¿POR QUÉ ES IMPORTANTE?
-----------------------
- Los parámetros "óptimos" dependen del contexto (lead time, demanda, costes)
- Un safety=60 puede ser perfecto para un producto pero terrible para otro
- Este método te permite encontrar buenos parámetros DE FORMA AUTOMÁTICA

================================================================================
"""

# ------------------------------------------------------------------------------
# IMPORTACIONES
# ------------------------------------------------------------------------------
# 'pandas' para crear tablas con los resultados
import pandas as pd

# Importamos la política que vamos a optimizar
from core.policies import BaseStockPolicy

# Importamos la función que corre múltiples episodios
from core.simulator import run_many


# ==============================================================================
# FUNCIÓN PRINCIPAL: grid_search
# ==============================================================================

def grid_search(
    # --- Parámetros del mundo (simulador) ---
    L: int,                    # Lead time
    T: int,                    # Semanas por episodio
    demand_min: int,           # Demanda mínima
    demand_max: int,           # Demanda máxima
    h: float,                  # Coste de holding
    p: float,                  # Coste de penalización
    s: float,                  # Coste de smoothing

    # --- El forecaster a usar (MA o EMA) ---
    forecaster,

    # --- Listas de valores a probar ---
    safeties: list[int],       # Lista de valores de safety a probar
    alphas: list[float],       # Lista de valores de alpha a probar

    # --- Configuración de la evaluación ---
    episodes: int,             # Episodios por combinación (más = más preciso)
    seed: int,                 # Semilla base para reproducibilidad
    init_on_hand: int,         # Inventario inicial
):
    """
    BÚSQUEDA EN GRILLA (GRID SEARCH)
    =================================

    ¿Qué hace?
    ----------
    Prueba TODAS las combinaciones posibles de (safety, alpha) y evalúa
    cada una corriendo múltiples episodios de simulación.

    Ejemplo:
    --------
    Si safeties = [0, 50, 100] y alphas = [0.5, 1.0], probará:
    - safety=0, alpha=0.5
    - safety=0, alpha=1.0
    - safety=50, alpha=0.5
    - safety=50, alpha=1.0
    - safety=100, alpha=0.5
    - safety=100, alpha=1.0
    = 6 combinaciones en total (3 × 2)

    Para cada combinación, corre 'episodes' episodios y calcula el
    coste PROMEDIO. La combinación con menor coste promedio es la "ganadora".

    ¿Qué retorna?
    -------------
    Una tabla (DataFrame) con todas las combinaciones probadas,
    ORDENADA de mejor a peor (menor coste primero).

    Columnas de la tabla:
    - safety: valor de safety usado
    - alpha: valor de alpha usado
    - fill_rate_mean: tasa de servicio promedio
    - lost_units_mean: unidades perdidas promedio
    - avg_inventory_mean: inventario promedio
    - avg_abs_order_change_mean: cambio en pedidos promedio
    - total_cost_mean: COSTE TOTAL PROMEDIO (la métrica principal)
    - episodes: cuántos episodios se corrieron

    NOTA SOBRE TIEMPO DE EJECUCIÓN:
    -------------------------------
    Si tienes 7 valores de safety × 5 valores de alpha × 200 episodios
    = 7,000 simulaciones. Cada simulación es rápida, pero muchas suman.
    Para "tuning rápido", usa menos valores o menos episodios.
    """

    # Lista donde acumularemos los resultados de cada combinación
    rows = []

    # --------------------------------------------------------------------------
    # BUCLE PRINCIPAL: Probar cada combinación de (safety, alpha)
    # --------------------------------------------------------------------------
    # Este es un "bucle anidado": para cada safety, probamos todos los alphas
    for safety in safeties:
        for alpha in alphas:

            # Crear una política con esta combinación de parámetros
            # Usamos forecaster.clone() para tener una copia fresca
            policy = BaseStockPolicy(
                L=L,
                safety=int(safety),
                alpha=float(alpha),
                forecaster=forecaster.clone()
            )

            # Evaluar esta política corriendo muchos episodios
            # run_many devuelve los KPIs promediados
            summary = run_many(
                L=L, T=T,
                demand_min=demand_min, demand_max=demand_max,
                h=h, p=p, s=s,
                policy=policy,
                seed=seed,           # Misma semilla base para comparación justa
                episodes=episodes,
                init_on_hand=init_on_hand
            )

            # Guardar los resultados de esta combinación
            rows.append({
                "safety": int(safety),
                "alpha": float(alpha),
                **summary  # Esto "desempaca" todos los KPIs del summary
            })

    # --------------------------------------------------------------------------
    # CREAR Y ORDENAR LA TABLA DE RESULTADOS
    # --------------------------------------------------------------------------
    # Convertimos la lista de resultados a un DataFrame (tabla)
    df = pd.DataFrame(rows)

    # Ordenamos por coste total promedio (menor es mejor)
    # reset_index(drop=True) renumera las filas 0, 1, 2, ...
    df = df.sort_values("total_cost_mean", ascending=True).reset_index(drop=True)

    # La primera fila (índice 0) es la MEJOR combinación encontrada
    return df
