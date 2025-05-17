from stable_baselines3 import PPO
from humanoid_env import HumanoidEnv
import time

# Load the environment and model
env = HumanoidEnv(render=True)
model = PPO.load("ppo_humanoid_logs/models/ppo_humanoid_final")  # Adjust path if needed

obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    time.sleep(1./240.)
    if done:
        print("Episode finished. Resetting...")
        obs = env.reset()
