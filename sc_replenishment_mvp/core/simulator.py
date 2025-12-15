import random
import pandas as pd

def simulate_episode(
    L: int,
    T: int,
    demand_min: int,
    demand_max: int,
    h: float,
    p: float,
    s: float,
    policy,
    seed: int = 1,
    init_on_hand: int = 250,
):
    """
    Simula 1 episodio (T semanas) del DC.

    Mundo:
    - Lead time fijo L (pipeline de tamaño L)
    - Demanda aleatoria uniforme [demand_min, demand_max]
    - Lost sales (sin backlog)

    Retorna:
    - KPIs del episodio
    - Traza semanal (DataFrame)
    """
    rng = random.Random(seed)

    on_hand = int(init_on_hand)
    pipe = [0 for _ in range(L)]  # pipe[0] llega esta semana, pipe[L-1] llega en L semanas
    q_prev = 0

    # Inicializamos el forecast con una demanda media “neutral”
    d0 = int((demand_min + demand_max) / 2)
    policy.reset(initial_demand=d0)

    rows = []
    total_demand = 0
    total_served = 0
    total_lost = 0
    sum_inventory = 0
    sum_abs_change = 0
    total_cost = 0.0

    for t in range(T):
        # Acción (pedido) según política
        q_t = policy.decide(on_hand=on_hand, pipe=pipe, q_prev=q_prev)

        # Llegadas: lo que estaba en pipe[0]
        arrivals = pipe[0]
        # desplazar pipeline e insertar pedido al final
        pipe = pipe[1:] + [q_t]

        # Demanda real
        d_t = rng.randint(demand_min, demand_max)

        available = on_hand + arrivals
        served = min(available, d_t)
        lost = d_t - served
        on_hand = available - served

        abs_change = abs(q_t - q_prev)
        q_prev = q_t

        # Coste (menor es mejor)
        cost = h * on_hand + p * lost + s * abs_change

        total_cost += cost
        total_demand += d_t
        total_served += served
        total_lost += lost
        sum_inventory += on_hand
        sum_abs_change += abs_change

        policy.observe(demand=d_t)

        rows.append({
            "week": t + 1,
            "on_hand": on_hand,
            "arrivals": arrivals,
            "order_q": q_t,
            "demand": d_t,
            "served": served,
            "lost": lost,
            "abs_order_change": abs_change,
            "cost": cost,
            "pipe_1w": pipe[0] if L >= 1 else 0,
            "pipe_2w": pipe[1] if L >= 2 else 0,
        })

    fill_rate = total_served / total_demand if total_demand > 0 else 0.0

    out = {
        "fill_rate": fill_rate,
        "lost_units": total_lost,
        "avg_inventory": sum_inventory / T,
        "avg_abs_order_change": sum_abs_change / T,
        "total_cost": total_cost,
        "trace": pd.DataFrame(rows),
    }
    return out

def run_many(
    L: int,
    T: int,
    demand_min: int,
    demand_max: int,
    h: float,
    p: float,
    s: float,
    policy,
    seed: int,
    episodes: int,
    init_on_hand: int,
):
    fills = []
    losts = []
    invs = []
    changes = []
    costs = []

    for i in range(episodes):
        ep = simulate_episode(
            L=L, T=T,
            demand_min=demand_min, demand_max=demand_max,
            h=h, p=p, s=s,
            policy=policy.clone(),
            seed=seed + i,
            init_on_hand=init_on_hand
        )
        fills.append(ep["fill_rate"])
        losts.append(ep["lost_units"])
        invs.append(ep["avg_inventory"])
        changes.append(ep["avg_abs_order_change"])
        costs.append(ep["total_cost"])

    return {
        "fill_rate_mean": float(sum(fills) / len(fills)),
        "lost_units_mean": float(sum(losts) / len(losts)),
        "avg_inventory_mean": float(sum(invs) / len(invs)),
        "avg_abs_order_change_mean": float(sum(changes) / len(changes)),
        "total_cost_mean": float(sum(costs) / len(costs)),
        "episodes": int(episodes),
    }
