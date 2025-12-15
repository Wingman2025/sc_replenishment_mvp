import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from core.simulator import simulate_episode, run_many
from core.tuning import grid_search
from core.policies import BaseStockPolicy
from core.forecast import MovingAverageForecast, ExponentialMovingAverageForecast

st.set_page_config(page_title="MVP Replenishment DC", layout="wide")

st.title("MVP: Replenishment en un DC (Simulador + Política + Tuning)")


def _init_state():
    if "last_episode_out" not in st.session_state:
        st.session_state.last_episode_out = None
    if "last_summary" not in st.session_state:
        st.session_state.last_summary = None
    if "last_tuning_df" not in st.session_state:
        st.session_state.last_tuning_df = None
    if "experiments" not in st.session_state:
        st.session_state.experiments = []


def _world_config(L, T, demand_min, demand_max, h, p, s, episodes, seed, forecast_type, k, ema_beta, safety, alpha):
    return {
        "L": int(L),
        "T": int(T),
        "demand_min": int(demand_min),
        "demand_max": int(demand_max),
        "h": float(h),
        "p": float(p),
        "s": float(s),
        "episodes": int(episodes),
        "seed": int(seed),
        "forecast_type": str(forecast_type),
        "k": int(k),
        "ema_beta": float(ema_beta),
        "safety": int(safety),
        "alpha": float(alpha),
    }


def _make_forecaster(forecast_type: str, k: int, ema_beta: float):
    if forecast_type.startswith("MA"):
        return MovingAverageForecast(window=int(k))
    return ExponentialMovingAverageForecast(beta=float(ema_beta))


def _add_experiment(name: str, kind: str, config: dict, payload: dict):
    st.session_state.experiments.append(
        {
            "id": f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{len(st.session_state.experiments)+1}",
            "ts": datetime.now().isoformat(timespec="seconds"),
            "name": name.strip() if name and name.strip() else f"{kind}-{len(st.session_state.experiments)+1}",
            "kind": kind,
            "config": config,
            **payload,
        }
    )

st.markdown(
    """
Este MVP muestra el flujo completo:
- **Simulador** (un mini-mundo reproducible)
- **Política** (regla matemática base-stock)
- **Evaluación** (KPIs por episodios)
- **Tuning** (buscar parámetros por simulación)
- **Forecast simple** (mejorar inputs sin RL)

Lead time fijo por defecto: **L=2**.
"""
)

_init_state()

with st.sidebar:
    st.header("Parámetros del mundo")
    L = st.number_input("Lead time L (semanas)", min_value=1, max_value=12, value=2, step=1)
    T = st.number_input("Semanas por episodio", min_value=10, max_value=208, value=52, step=1)

    demand_min = st.number_input("Demanda mínima", min_value=0, max_value=100000, value=50, step=1)
    demand_max = st.number_input("Demanda máxima", min_value=1, max_value=100000, value=120, step=1)

    st.header("Costes (para el objetivo)")
    h = st.number_input("h: coste inventario (por unidad)", min_value=0.0, value=1.0, step=0.1)
    p = st.number_input("p: penalización lost sales (por unidad)", min_value=0.0, value=20.0, step=1.0)
    s = st.number_input("s: penalización cambio pedido |Δq|", min_value=0.0, value=0.1, step=0.1)

    st.header("Forecast (Nivel 2)")
    forecast_type = st.selectbox("Método", ["MA (media móvil)", "EMA (media móvil exponencial)"])
    k = st.number_input("Ventana (K) para MA", min_value=1, max_value=52, value=4, step=1)
    ema_beta = st.slider("Beta EMA (0.05 = muy suave, 0.8 = reacciona rápido)", min_value=0.05, max_value=0.95, value=0.3, step=0.05)

    st.header("Política base-stock (Nivel 1)")
    safety = st.number_input("safety (stock de seguridad)", min_value=0, max_value=100000, value=60, step=5)
    alpha = st.slider("alpha (suavizado de pedidos)", min_value=0.05, max_value=1.0, value=0.7, step=0.05)

    st.header("Simulación")
    episodes = st.number_input("Episodios (corridas)", min_value=1, max_value=5000, value=200, step=10)
    seed = st.number_input("Seed base", min_value=0, max_value=10_000_000, value=1, step=1)

if int(demand_max) <= int(demand_min):
    st.error("`Demanda máxima` debe ser mayor que `Demanda mínima`.")
    st.stop()

forecaster = _make_forecaster(forecast_type=str(forecast_type), k=int(k), ema_beta=float(ema_beta))
policy = BaseStockPolicy(L=int(L), safety=int(safety), alpha=float(alpha), forecaster=forecaster)
config = _world_config(L, T, demand_min, demand_max, h, p, s, episodes, seed, forecast_type, k, ema_beta, safety, alpha)

tab_sim, tab_eval, tab_tune, tab_compare, tab_export = st.tabs(
    ["Simular", "Evaluar", "Tuning", "Comparar", "Exportar"]
)

with tab_sim:
    st.subheader("Simular (ver una corrida)")
    bcol1, bcol2 = st.columns([1, 2])
    with bcol1:
        simulate_clicked = st.button("Simular 1 episodio", type="primary")
    with bcol2:
        exp_name_episode = st.text_input("Nombre (opcional) para guardar", value="", key="exp_name_episode")

    if simulate_clicked:
        out = simulate_episode(
            L=int(L),
            T=int(T),
            demand_min=int(demand_min),
            demand_max=int(demand_max),
            h=float(h),
            p=float(p),
            s=float(s),
            policy=policy,
            seed=int(seed),
            init_on_hand=int((demand_min + demand_max) / 2 * 3),  # heurística: ~3 semanas de demanda media
        )
        st.session_state.last_episode_out = out

    if st.session_state.last_episode_out is not None:
        out = st.session_state.last_episode_out
        st.markdown("### KPIs (1 episodio)")
        kpis = {k: out[k] for k in ["fill_rate", "lost_units", "avg_inventory", "avg_abs_order_change", "total_cost"]}
        st.json(kpis)

        df = out["trace"]
        st.markdown("### Traza (semanal)")
        st.dataframe(df, use_container_width=True)

        st.markdown("### Gráficas")
        st.line_chart(df.set_index("week")[["on_hand", "demand", "order_q", "lost"]])

        dcol1, dcol2 = st.columns([1, 1])
        with dcol1:
            csv_trace = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar traza.csv",
                data=csv_trace,
                file_name="episode_trace.csv",
                mime="text/csv",
            )
        with dcol2:
            if st.button("Guardar episodio en historial"):
                _add_experiment(
                    name=exp_name_episode,
                    kind="episode",
                    config=config,
                    payload={"kpis": kpis, "trace": df},
                )

with tab_eval:
    st.subheader("Evaluar (promedio de muchos episodios)")
    ecol1, ecol2 = st.columns([1, 2])
    with ecol1:
        eval_clicked = st.button("Correr N episodios y resumir")
    with ecol2:
        exp_name_eval = st.text_input("Nombre (opcional) para guardar", value="", key="exp_name_eval")

    if eval_clicked:
        summary = run_many(
            L=int(L),
            T=int(T),
            demand_min=int(demand_min),
            demand_max=int(demand_max),
            h=float(h),
            p=float(p),
            s=float(s),
            policy=policy,
            seed=int(seed),
            episodes=int(episodes),
            init_on_hand=int((demand_min + demand_max) / 2 * 3),
        )
        st.session_state.last_summary = summary

    if st.session_state.last_summary is not None:
        summary = st.session_state.last_summary
        st.markdown("### Resultados promedio")
        st.json(summary)

        scol1, scol2 = st.columns([1, 1])
        with scol1:
            df_summary = pd.DataFrame([summary])
            csv_summary = df_summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar summary.csv",
                data=csv_summary,
                file_name="evaluation_summary.csv",
                mime="text/csv",
            )
        with scol2:
            if st.button("Guardar evaluación en historial"):
                _add_experiment(
                    name=exp_name_eval,
                    kind="evaluation",
                    config=config,
                    payload={"kpis": summary},
                )

with tab_tune:
    st.subheader("Tuning (grid search) — Nivel 1 automático")
    st.caption(
        "Busca las mejores combinaciones de (safety, alpha) corriendo simulaciones. Esto es 'sin IA' pero muy efectivo en supply chain."
    )

    tcol1, tcol2, tcol3 = st.columns([1, 1, 2])
    with tcol1:
        safety_list = st.text_input("Lista safety (coma)", value="0,20,40,60,80,100,120")
    with tcol2:
        alpha_list = st.text_input("Lista alpha (coma)", value="1.0,0.8,0.6,0.4,0.2")
    with tcol3:
        top_n = st.number_input("Mostrar top N", min_value=3, max_value=50, value=10, step=1)

    tune_col1, tune_col2 = st.columns([1, 2])
    with tune_col1:
        tune_clicked = st.button("Ejecutar tuning")
    with tune_col2:
        exp_name_tune = st.text_input("Nombre (opcional) para guardar", value="", key="exp_name_tune")

    if tune_clicked:
        safeties = [int(x.strip()) for x in safety_list.split(",") if x.strip()]
        alphas = [float(x.strip()) for x in alpha_list.split(",") if x.strip()]

        df = grid_search(
            L=int(L),
            T=int(T),
            demand_min=int(demand_min),
            demand_max=int(demand_max),
            h=float(h),
            p=float(p),
            s=float(s),
            forecaster=forecaster,
            safeties=safeties,
            alphas=alphas,
            episodes=int(episodes),
            seed=int(seed),
            init_on_hand=int((demand_min + demand_max) / 2 * 3),
        )
        st.session_state.last_tuning_df = df

    if st.session_state.last_tuning_df is not None:
        df = st.session_state.last_tuning_df
        st.markdown("### Top combinaciones (menor coste total promedio = mejor)")
        st.dataframe(df.head(int(top_n)), use_container_width=True)

        xcol1, xcol2 = st.columns([1, 1])
        with xcol1:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar resultados.csv", data=csv, file_name="tuning_results.csv", mime="text/csv")
        with xcol2:
            if st.button("Guardar tuning en historial"):
                _add_experiment(
                    name=exp_name_tune,
                    kind="tuning",
                    config=config,
                    payload={"df": df},
                )

    st.divider()

    with st.expander("Nivel 3 (opcional): RL — qué falta para activarlo"):
        st.markdown(
            """
Para RL necesitas:
1) envolver el simulador como un entorno **Gymnasium** (`reset()` y `step(action)`),
2) entrenar con **Stable-Baselines3** (PPO/DQN),
3) comparar contra el baseline (lo que ya hicimos).

Este MVP ya tiene el simulador y la métrica. El RL se añade como un plugin más de `policy`.
"""
        )

with tab_compare:
    st.subheader("Comparar experimentos")
    if len(st.session_state.experiments) == 0:
        st.info("Aún no hay experimentos guardados. Guarda un episodio, evaluación o tuning para compararlos aquí.")
    else:
        options = {
            f"{e['ts']} | {e['kind']} | {e['name']}": e["id"]
            for e in st.session_state.experiments
        }
        labels = list(options.keys())
        ccol1, ccol2 = st.columns([1, 1])
        with ccol1:
            left_label = st.selectbox("Experimento A", labels, index=0)
        with ccol2:
            right_label = st.selectbox("Experimento B", labels, index=min(1, len(labels) - 1))

        by_id = {e["id"]: e for e in st.session_state.experiments}
        left = by_id[options[left_label]]
        right = by_id[options[right_label]]

        lpanel, rpanel = st.columns([1, 1])
        with lpanel:
            st.markdown("### A")
            st.json({"name": left["name"], "kind": left["kind"], "ts": left["ts"]})
            if "kpis" in left and left["kpis"] is not None:
                st.json(left["kpis"])
        with rpanel:
            st.markdown("### B")
            st.json({"name": right["name"], "kind": right["kind"], "ts": right["ts"]})
            if "kpis" in right and right["kpis"] is not None:
                st.json(right["kpis"])

        if "trace" in left and "trace" in right:
            if left["trace"] is not None and right["trace"] is not None:
                st.markdown("### Traza (comparativa)")
                a = left["trace"].copy()
                b = right["trace"].copy()
                a["series"] = "A"
                b["series"] = "B"
                dfc = pd.concat([a, b], ignore_index=True)
                st.line_chart(dfc, x="week", y=["on_hand", "order_q", "lost"], color="series")

with tab_export:
    st.subheader("Exportar historial")
    if len(st.session_state.experiments) == 0:
        st.info("No hay nada para exportar todavía.")
    else:
        rows = []
        for e in st.session_state.experiments:
            rows.append(
                {
                    "id": e.get("id"),
                    "ts": e.get("ts"),
                    "name": e.get("name"),
                    "kind": e.get("kind"),
                }
            )
        df_hist = pd.DataFrame(rows)
        st.dataframe(df_hist, use_container_width=True)

        csv_hist = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar historial.csv", data=csv_hist, file_name="experiments.csv", mime="text/csv")

