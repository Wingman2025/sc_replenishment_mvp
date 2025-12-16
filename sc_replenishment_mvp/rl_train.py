"""
================================================================================
ENTRENAMIENTO CON REINFORCEMENT LEARNING - rl_train.py
================================================================================

¿QUÉ HACE ESTE ARCHIVO?
-----------------------
Este archivo entrena un agente de RL (Reinforcement Learning) para aprender
a decidir cuánto pedir cada semana en el Centro de Distribución.

NIVEL 3 - OPCIONAL
------------------
Este archivo es completamente OPCIONAL. Si la política Base-Stock (Nivel 1/2)
funciona bien para tu caso, NO necesitas esto.

¿CUÁNDO USAR RL?
----------------
- Cuando el problema es muy complejo (múltiples productos, restricciones raras)
- Cuando las reglas tradicionales no funcionan bien
- Cuando tienes curiosidad por experimentar con IA "de verdad"

¿QUÉ ALGORITMO USA?
-------------------
PPO (Proximal Policy Optimization) de la librería Stable-Baselines3.
Es uno de los algoritmos más populares y robustos para RL.

REQUISITOS:
-----------
Necesitas instalar las librerías de RL:
  pip install gymnasium stable-baselines3

CÓMO EJECUTAR:
--------------
Desde la terminal, en la carpeta del proyecto:
  python rl_train.py

El entrenamiento tomará unos minutos (dependiendo de tu computadora).
Al terminar, guardará el modelo entrenado en un archivo.

NOTA IMPORTANTE:
----------------
- Este es un ejemplo BÁSICO para que veas cómo funciona el ciclo de RL
- Para resultados "serios" necesitarías:
  * Más timesteps de entrenamiento (más tiempo)
  * Ajustar hiperparámetros del algoritmo
  * Probar diferentes escenarios y demandas
  * Comparar rigurosamente contra el baseline (Base-Stock)

================================================================================
"""

# ------------------------------------------------------------------------------
# IMPORTACIONES
# ------------------------------------------------------------------------------
# Importamos nuestro entorno RL personalizado (ver gym_env.py)
from core.gym_env import DCReplenishmentEnv

# Intentamos importar el algoritmo PPO de Stable-Baselines3
# Si no está instalado, mostramos un mensaje claro y terminamos
try:
    from stable_baselines3 import PPO
except Exception as e:
    raise SystemExit("Falta stable-baselines3. Instala con: pip install stable-baselines3")


# ==============================================================================
# FUNCIÓN PRINCIPAL: main
# ==============================================================================

def main():
    """
    Función principal que ejecuta el entrenamiento y prueba del agente RL.

    El proceso tiene 3 fases:
    1. CREAR el entorno (nuestro simulador de DC)
    2. ENTRENAR el agente (el algoritmo aprende probando acciones)
    3. PROBAR el agente entrenado (ver cómo lo hace)
    """

    # --------------------------------------------------------------------------
    # FASE 1: Crear el entorno
    # --------------------------------------------------------------------------
    # Creamos una instancia de nuestro entorno RL con los parámetros del mundo
    # Estos son los mismos parámetros que usamos en el simulador normal
    print("="*60)
    print("FASE 1: Creando el entorno de simulación...")
    print("="*60)

    env = DCReplenishmentEnv(
        L=2,                    # Lead time: 2 semanas
        T=52,                   # Episodio: 52 semanas (1 año)
        demand_min=50,          # Demanda mínima por semana
        demand_max=120          # Demanda máxima por semana
    )
    print("✓ Entorno creado exitosamente")

    # --------------------------------------------------------------------------
    # FASE 2: Entrenar el agente
    # --------------------------------------------------------------------------
    # Creamos el modelo PPO y lo entrenamos
    # "MlpPolicy" significa que usará una red neuronal simple (Multi-Layer Perceptron)
    # verbose=1 hace que muestre progreso durante el entrenamiento
    print("\n" + "="*60)
    print("FASE 2: Entrenando el agente RL (esto puede tomar unos minutos)...")
    print("="*60)

    model = PPO(
        "MlpPolicy",            # Tipo de política: red neuronal simple
        env,                    # El entorno donde aprenderá
        verbose=1               # Mostrar progreso
    )

    # total_timesteps = cuántos "pasos" de entrenamiento
    # 50,000 es poco (para que sea rápido). Para mejores resultados, usa 200,000+
    model.learn(total_timesteps=50_000)

    print("✓ Entrenamiento completado")

    # --------------------------------------------------------------------------
    # Guardar el modelo entrenado
    # --------------------------------------------------------------------------
    # Guardamos el modelo para poder usarlo después sin re-entrenar
    model.save("ppo_dc_replenishment")
    print("✓ Modelo guardado en 'ppo_dc_replenishment.zip'")

    # --------------------------------------------------------------------------
    # FASE 3: Probar el agente entrenado
    # --------------------------------------------------------------------------
    # Ahora vemos cómo lo hace el agente en un episodio completo
    print("\n" + "="*60)
    print("FASE 3: Probando el agente entrenado (1 episodio de 52 semanas)...")
    print("="*60)

    # Reiniciamos el entorno para empezar un episodio fresco
    obs, info = env.reset()

    # Acumulador para la recompensa total
    total_reward = 0.0

    # Corremos 52 semanas
    for week in range(52):
        # El modelo entrenado predice la mejor acción dado el estado actual
        # deterministic=True significa "no explores, usa tu mejor decisión"
        action, _ = model.predict(obs, deterministic=True)

        # Ejecutamos la acción en el entorno
        obs, reward, terminated, truncated, info = env.step(action)

        # Acumulamos la recompensa
        total_reward += reward

        # Si el episodio terminó antes (no debería pasar con T=52), salimos
        if terminated:
            break

    # --------------------------------------------------------------------------
    # Mostrar resultados
    # --------------------------------------------------------------------------
    print(f"\n✓ Reward total del episodio: {total_reward:.2f}")
    print(f"  (Recuerda: reward = -coste, así que más cercano a 0 = mejor)")
    print(f"  Coste total aproximado: {-total_reward:.2f}")

    print("\n" + "="*60)
    print("¡Listo! Ahora puedes:")
    print("- Cargar el modelo con: PPO.load('ppo_dc_replenishment')")
    print("- Comparar el coste RL vs el coste de Base-Stock")
    print("- Ajustar hiperparámetros para mejores resultados")
    print("="*60)


# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================
# Este bloque se ejecuta solo cuando corres "python rl_train.py" directamente
# No se ejecuta si importas este archivo desde otro lugar

if __name__ == "__main__":
    main()
