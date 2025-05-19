import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from train import PPO
from humanoid_env import HumanoidEnv

def run_training_with_monitoring(num_episodes=300, save_interval=25):
    """Run training with monitoring and visualization of progress"""
    # Create environment
    env = HumanoidEnv(render=True)
    observation_space = env.observation_space
    action_space = env.action_space
    
    # Initialize PPO agent with tuned hyperparameters
    agent = PPO(
        observation_space=observation_space,
        action_space=action_space,
        hidden_size=512,           # Larger network
        lr=1e-4,                   # Lower learning rate
        gamma=0.99,                # Discount factor
        gae_lambda=0.95,           # GAE lambda
        clip_ratio=0.1,            # Smaller clip ratio
        target_kl=0.01,            # KL divergence target
        update_epochs=10,          # Number of epochs
        batch_size=128,            # Larger batch size
        value_coef=0.5,            # Value loss coefficient
        entropy_coef=0.01,         # Entropy bonus coefficient
        max_grad_norm=0.5,         # Gradient clipping
        use_gae=True,              # Use GAE
        normalize_advantage=True   # Normalize advantages
    )
    
    # Initialize tracking variables
    episode_rewards = []
    episode_lengths = []
    avg_rewards = []
    
    try:
        for episode in range(num_episodes):
            observation = env.reset()
            total_reward = 0
            raw_rewards = []  # Track raw rewards before normalization
            
            for step in range(env.max_episode_steps):
                # Get action from the PPO agent
                action = agent.select_action(observation)
                
                # Take action in the environment
                next_observation, reward, done, info = env.step(action)
                total_reward += reward
                raw_rewards.append(reward)
                
                # Update the PPO agent
                agent.update(observation, action, reward, next_observation, done)
                
                # Update observation
                observation = next_observation
                
                # Render the environment
                env.render()
                
                if done:
                    break
            
            # Record episode statistics
            episode_rewards.append(total_reward)
            episode_lengths.append(step + 1)
            
            # Calculate running average (last 10 episodes)
            running_avg = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            avg_rewards.append(running_avg)
            
            # Print progress
            print(f"Episode {episode} finished after {step + 1} steps with total reward {total_reward:.2f}")
            print(f"Running average reward (last 10): {running_avg:.2f}")
            
            # Additional diagnostics
            if len(raw_rewards) > 0:
                print(f"Min reward: {min(raw_rewards):.2f}, Max reward: {max(raw_rewards):.2f}")
                print(f"Reward std: {np.std(raw_rewards):.2f}")
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                model_path = f"model_episode_{episode+1}.pth"
                agent.save(model_path)
                print(f"Model saved to {model_path}")
                
                # Plot and save learning curve
                plot_learning_curve(episode_rewards, avg_rewards, episode_lengths, episode+1)
    
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        env.close()
        
        # Final plot
        plot_learning_curve(episode_rewards, avg_rewards, episode_lengths, num_episodes, final=True)
        
        return episode_rewards, avg_rewards, episode_lengths

def plot_learning_curve(rewards, avg_rewards, lengths, episode, final=False):
    """Plot and save learning curves"""
    plt.figure(figsize=(12, 10))
    
    # Plot rewards
    plt.subplot(2, 1, 1)
    plt.plot(rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='Running Average (10 episodes)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    
    # Plot episode lengths
    plt.subplot(2, 1, 2)
    plt.plot(lengths)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Lengths')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    status = "final" if final else "progress"
    plt.savefig(f"learning_curve_{status}_episode_{episode}.png")
    plt.close()

def analyze_training_results(rewards, avg_rewards, lengths):
    """Analyze training results to determine if the model is learning"""
    # Check if rewards are improving
    if len(rewards) < 10:
        print("Not enough episodes to analyze results")
        return False
    
    # Calculate improvement over training
    first_10_avg = np.mean(rewards[:10])
    last_10_avg = np.mean(rewards[-10:])
    improvement = last_10_avg - first_10_avg
    
    print("\nTraining Analysis:")
    print(f"Average reward (first 10 episodes): {first_10_avg:.2f}")
    print(f"Average reward (last 10 episodes): {last_10_avg:.2f}")
    print(f"Improvement: {improvement:.2f}")
    
    # Check if episode lengths are increasing (agent stays alive longer)
    first_10_len_avg = np.mean(lengths[:10])
    last_10_len_avg = np.mean(lengths[-10:])
    length_improvement = last_10_len_avg - first_10_len_avg
    
    print(f"Average episode length (first 10 episodes): {first_10_len_avg:.2f}")
    print(f"Average episode length (last 10 episodes): {last_10_len_avg:.2f}")
    print(f"Length improvement: {length_improvement:.2f}")
    
    # Calculate reward volatility
    reward_std = np.std(rewards)
    recent_reward_std = np.std(rewards[-10:])
    
    print(f"Overall reward standard deviation: {reward_std:.2f}")
    print(f"Recent reward standard deviation: {recent_reward_std:.2f}")
    
    # Determine if the model is learning
    is_learning = improvement > 10 and length_improvement > 5
    
    if is_learning:
        print("\nThe model shows signs of learning to stand and walk!")
    else:
        print("\nThe model may need more training or parameter tuning to learn effectively.")
    
    return is_learning

if __name__ == "__main__":
    # Run training with monitoring
    print("Starting training with monitoring...")
    rewards, avg_rewards, lengths = run_training_with_monitoring(num_episodes=100)
    
    # Analyze results
    analyze_training_results(rewards, avg_rewards, lengths)
