import pandas as pd
from core.policies import BaseStockPolicy
from core.simulator import run_many

def grid_search(
    L: int,
    T: int,
    demand_min: int,
    demand_max: int,
    h: float,
    p: float,
    s: float,
    forecaster,
    safeties: list[int],
    alphas: list[float],
    episodes: int,
    seed: int,
    init_on_hand: int,
):
    rows = []
    for safety in safeties:
        for alpha in alphas:
            policy = BaseStockPolicy(L=L, safety=int(safety), alpha=float(alpha), forecaster=forecaster.clone())
            summary = run_many(
                L=L, T=T,
                demand_min=demand_min, demand_max=demand_max,
                h=h, p=p, s=s,
                policy=policy,
                seed=seed,
                episodes=episodes,
                init_on_hand=init_on_hand
            )
            rows.append({
                "safety": int(safety),
                "alpha": float(alpha),
                **summary
            })

    df = pd.DataFrame(rows).sort_values("total_cost_mean", ascending=True).reset_index(drop=True)
    return df
