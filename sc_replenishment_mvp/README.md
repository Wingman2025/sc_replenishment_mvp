# MVP: Simulador + Replenishment (DC) + Tuning + (opcional) RL

Este mini-proyecto es un **MVP educativo** para el caso:
- 1 DC, 1 producto/familia
- Decisión semanal: **cuánto pedir**
- Lead time fijo **L = 2 semanas**
- Demanda aleatoria uniforme (por defecto **50–120**)
- Sin backlog (si no hay stock → *lost sales*)

Incluye 3 niveles (en el sentido que hablamos):
1) **Nivel 1 (sin IA):** política base-stock (order-up-to) + suavizado + *tuning* por simulación (grid search).
2) **Nivel 2 (IA “ligera”):** “mejorar inputs” con forecast simple (MA/EMA) para que la regla decida mejor.
3) **Nivel 3 (opcional):** estructura para RL (Gymnasium/SB3). En este MVP es opcional y queda como “plugin”.

---

## Cómo ejecutar (local)

1) Crea y activa un entorno:
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/mac:
source .venv/bin/activate
```

2) Instala dependencias:
```bash
pip install -r requirements.txt
```

3) Lanza la app:
```bash
streamlit run app.py
```

---

## Qué verás en la app

- **Simular**: eliges `safety` y `alpha`, corres N episodios y ves KPIs + trazas.
- **Tuning**: buscas la mejor combinación (`safety`, `alpha`) según el coste total.
- **Forecast**: comparas MA vs EMA (input mejorado) usando la misma política.

---

## Notas importantes (verdad cruda)
- Esto no “entiende” supply chain: optimiza un objetivo (coste) dentro de este mini-mundo.
- Si tu mundo real tiene MOQ, pallets, capacidad, multi-SKU, etc., lo agregas **después**.
