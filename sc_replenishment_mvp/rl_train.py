"""
Entrenamiento RL (Nivel 3) — opcional.

Requisitos:
  pip install gymnasium stable-baselines3

Ejecuta:
  python rl_train.py

Nota:
- Esto entrena rápido (pocos timesteps) para que veas el ciclo completo.
- Para resultados serios necesitarás más episodios/timesteps y más escenarios.
"""

from core.gym_env import DCReplenishmentEnv

try:
    from stable_baselines3 import PPO
except Exception as e:
    raise SystemExit("Falta stable-baselines3. Instala con: pip install stable-baselines3")

def main():
    env = DCReplenishmentEnv(L=2, T=52, demand_min=50, demand_max=120)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50_000)  # sube esto para aprender mejor

    model.save("ppo_dc_replenishment")

    # prueba rápida
    obs, info = env.reset()
    total_reward = 0.0
    for _ in range(52):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break

    print("Reward total (1 episodio):", total_reward)

if __name__ == "__main__":
    main()
