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

La app está organizada para que puedas aprender iterando rápido. En general el flujo recomendado es:

1) Ajustas parámetros en el **sidebar** (panel izquierdo)
2) Corres un **episodio** para entender el comportamiento (tab **Simular**)
3) Corres **muchos episodios** para tener promedios estables (tab **Evaluar**)
4) Haces **tuning** para buscar buenos parámetros automáticamente (tab **Tuning**)
5) Guardas y comparas experimentos (tabs **Comparar** y **Exportar**)

### Tabs (qué hace cada una)

- **Simular**
  - Corre **1 episodio** (T semanas) y te muestra:
    - KPIs del episodio
    - La **traza semanal** (tabla) con inventario, demanda, pedidos, etc.
    - Gráficas
  - Útil para “ver la película” del sistema.

- **Evaluar**
  - Corre **N episodios** y devuelve KPIs **promedio**.
  - Útil para comparar políticas/parametrizaciones de forma más justa (menos ruido aleatorio).

- **Tuning**
  - Prueba combinaciones de (`safety`, `alpha`) con **grid search**.
  - Ordena resultados por **coste total promedio** (menor = mejor).

- **Comparar**
  - Compara 2 experimentos guardados “A vs B”.

- **Exportar**
  - Descarga el historial de experimentos guardados en CSV.

---

## Cómo “jugar” con los parámetros (y qué impacto tienen)

### 1) Parámetros del mundo

- **Lead time `L` (semanas)**
  - Qué es: cuánto tarda en llegar un pedido.
  - Impacto típico:
    - Si sube `L`, necesitas **más stock/pedidos** para evitar quiebres.
    - El sistema se vuelve más difícil de operar (más incertidumbre “en tránsito”).

- **Semanas por episodio `T`**
  - Qué es: duración de la simulación.
  - Impacto típico:
    - `T` más grande da un resultado más representativo (pero tarda más).

- **Demanda mínima / máxima (`demand_min`, `demand_max`)**
  - Qué es: rango de demanda aleatoria uniforme.
  - Impacto típico:
    - Rango más alto o más ancho = más presión de servicio y más variabilidad.

### 2) Costes del objetivo: `h`, `p`, `s`

El simulador calcula coste semanal como:

- holding: `h * on_hand`
- quiebres (lost sales): `p * lost`
- variabilidad de pedido: `s * |Δq|`

Interpretación práctica:

- **`h` (holding cost)**
  - “Me cuesta dinero tener inventario.”
  - Si subes `h`, el sistema tiende a:
    - bajar inventario promedio
    - aceptar más quiebres (si `p` no es muy alto)

- **`p` (penalización de lost sales / stockout)**
  - “Me duele no servir una unidad.”
  - Si subes `p`, el sistema tiende a:
    - subir inventario/pedidos para proteger fill rate
    - reducir `lost_units`

- **`s` (penalización por cambios bruscos en pedidos)**
  - “Me cuesta cambiar mucho el pedido semana a semana.”
  - Si subes `s`, el sistema tiende a:
    - suavizar pedidos (`order_q` cambia menos)
    - a veces subir inventario (para evitar ajustes bruscos)

Nota: lo importante no es solo el valor absoluto, sino la **relación** entre `h` vs `p` vs `s`.

### 3) Forecast (Nivel 2): MA vs EMA

- **MA (media móvil)**
  - Parámetro `K` (ventana).
  - K grande = forecast más estable pero más lento para reaccionar.

- **EMA (media móvil exponencial)**
  - Parámetro `beta`.
  - beta alto = reacciona rápido (más sensible a cambios), pero puede volverse “nervioso”.

### 4) Política base-stock (Nivel 1): `safety` y `alpha`

- **`safety` (stock de seguridad)**
  - Si subes `safety`, típicamente:
    - sube el inventario promedio
    - bajan los quiebres (`lost_units`)
    - sube el coste por holding si `h` es significativo

- **`alpha` (suavizado del pedido)**
  - `alpha` cerca de 1.0 = responde rápido (pedido más “agresivo”).
  - `alpha` bajo = responde lento (pedido más estable).
  - Si `s` es alto, a menudo conviene un `alpha` más bajo.

---

## Recetas de experimentos (rápidas y concretas)

Usa **Simular** para ver la dinámica y luego **Evaluar** con, por ejemplo, 200 episodios para confirmar.

- **Receta 1: Priorizar servicio (menos quiebres)**
  - Sube `p` (ej. 20 → 50)
  - Sube `safety` (ej. 60 → 120)
  - Espera ver:
    - `fill_rate` sube
    - `lost_units` baja
    - `avg_inventory` sube

- **Receta 2: Minimizar inventario (aceptando quiebres)**
  - Sube `h` (ej. 1.0 → 3.0)
  - Baja `safety` (ej. 60 → 0/20)
  - Espera ver:
    - `avg_inventory` baja
    - `lost_units` sube

- **Receta 3: Pedidos estables (operación “suave”)**
  - Sube `s` (ej. 0.1 → 1.0)
  - Baja `alpha` (ej. 0.7 → 0.3)
  - Espera ver:
    - `avg_abs_order_change` baja
    - `order_q` se vuelve menos errático

- **Receta 4: Más variabilidad de demanda**
  - Aumenta rango (ej. 50–120 → 20–200)
  - Luego prueba Tuning para re-optimizar (`safety`, `alpha`).

---

## Consejos prácticos (para que el “juego” sea consistente)

- Mantén fija la `seed` cuando compares dos configuraciones (comparación más justa).
- Cuando algo te guste en **Simular**, confirma en **Evaluar** (porque 1 episodio puede salir “con suerte”).
- Si cambias parámetros del mundo (`L`, demanda, costes), vuelve a correr **Tuning**.

---

## Notas importantes (verdad cruda)
- Esto no “entiende” supply chain: optimiza un objetivo (coste) dentro de este mini-mundo.
- Si tu mundo real tiene MOQ, pallets, capacidad, multi-SKU, etc., lo agregas **después**.
