import gym
import numpy as np
import torch
import argparse
from humanoid_env import HumanoidEnv
from train import PPO

def check_model(model_path=None):
    """
    Test a trained model or run random actions if no model is provided.
    """
    # Create environment
    env = HumanoidEnv(render=True)
    observation_space = env.observation_space
    action_space = env.action_space
    
    # Initialize agent
    agent = PPO(
        observation_space=observation_space,
        action_space=action_space,
        hidden_size=512,
        lr=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.1,
        target_kl=0.01,
        update_epochs=10,
        batch_size=128,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        use_gae=True,
        normalize_advantage=True
    )
    
    # Load model if provided
    if model_path:
        agent.load(model_path)
        print(f"Testing trained model from: {model_path}")
    else:
        print("No model provided. Running with random actions.")
    
    try:
        # Run episodes
        for episode in range(5):  # Run 5 test episodes
            observation = env.reset()
            total_reward = 0
            done = False
            step = 0
            
            while not done and step < env.max_episode_steps:
                # Get action
                if model_path:
                    action = agent.select_action(observation)
                else:
                    action = env.action_space.sample()
                
                # Take action
                next_observation, reward, done, info = env.step(action)
                total_reward += reward
                
                # Update observation
                observation = next_observation
                
                # Render
                env.render()
                
                step += 1
            
            print(f"Episode {episode} finished after {step} steps with reward {total_reward:.2f}")
    
    except KeyboardInterrupt:
        print("Testing interrupted.")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained humanoid model")
    parser.add_argument("--model", type=str, help="Path to the trained model file")
    args = parser.parse_args()
    
    check_model(args.model)
