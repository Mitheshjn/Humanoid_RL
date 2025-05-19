import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import time
import os
from humanoid_env import HumanoidEnv

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# PPO Algorithm Implementation
class PPO:
    def __init__(self, observation_space, action_space, 
                 hidden_size=512,               # Increased from 256
                 lr=1e-4,                       # Reduced from 3e-4
                 gamma=0.99, 
                 gae_lambda=0.95,               # Added GAE lambda
                 clip_ratio=0.1,                # Reduced from 0.2
                 target_kl=0.01, 
                 update_epochs=10, 
                 batch_size=128,                # Increased from 64
                 value_coef=0.5, 
                 entropy_coef=0.01,
                 max_grad_norm=0.5,             # Added gradient clipping
                 use_gae=True,                  # Use Generalized Advantage Estimation
                 normalize_advantage=True):     # Normalize advantages
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        
        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_gae = use_gae
        self.normalize_advantage = normalize_advantage
        
        # Initialize actor and critic networks
        self.actor = Actor(self.obs_dim, self.act_dim, hidden_size)
        self.critic = Critic(self.obs_dim, hidden_size)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Initialize memory buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # For logging
        self.training_step = 0
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        
    def select_action(self, observation):
        """Select action based on current policy"""
        with torch.no_grad():
            observation = torch.FloatTensor(observation).to(self.device)
            mean, std = self.actor(observation)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(observation)
            
        # Store in memory
        self.states.append(observation.cpu().numpy())
        self.actions.append(action.cpu().numpy())
        self.log_probs.append(log_prob.cpu().item())
        self.values.append(value.cpu().item())
        
        return action.cpu().numpy()
    
    def update(self, observation, action, reward, next_observation, done):
        """Store experience in memory"""
        self.rewards.append(reward)
        self.dones.append(done)
        
        # If episode is done, update policy
        if done:
            self.training_step += 1
            self._update_policy()
            self._clear_memory()
    
    def _compute_advantages(self, rewards, values, dones):
        """Compute advantages using GAE if enabled, otherwise use simple returns - values"""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        if self.use_gae:
            # GAE calculation
            last_gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
                advantages[t] = last_gae
                returns[t] = advantages[t] + values[t]
        else:
            # Simple advantage calculation
            returns = self._compute_returns()
            advantages = returns - np.array(values)
        
        return advantages, returns
    
    def _update_policy(self):
        """Update policy using PPO algorithm"""
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        values = torch.FloatTensor(np.array(self.values)).to(self.device)
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages(
            np.array(self.rewards), 
            np.array(self.values), 
            np.array(self.dones)
        )
        
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.update_epochs):
            # Generate random mini-batches
            batch_indices = np.random.permutation(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                # Get mini-batch
                idx = batch_indices[start_idx:start_idx + self.batch_size]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                
                # Get current log probs and values
                mean, std = self.actor(batch_states)
                dist = Normal(mean, std)
                curr_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().mean()
                curr_values = self.critic(batch_states).squeeze()
                
                # Compute ratio and clipped ratio
                ratio = torch.exp(curr_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                
                # Compute losses
                actor_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                critic_loss = ((curr_values - batch_returns) ** 2).mean()
                
                # Add entropy bonus
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # Update actor and critic
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Apply gradient clipping
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Calculate approximate KL divergence for early stopping
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                if approx_kl > self.target_kl:
                    break
    
    def _compute_returns(self):
        """Compute discounted returns"""
        returns = []
        running_return = 0
        
        # Compute returns in reverse order
        for t in reversed(range(len(self.rewards))):
            running_return = self.rewards[t] + self.gamma * running_return * (1 - self.dones[t])
            returns.insert(0, running_return)
            
        return returns
    
    def _clear_memory(self):
        """Clear memory buffer"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def save(self, filename):
        """Save model parameters"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load model parameters"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            print(f"Model loaded from {filename}")
        else:
            print(f"No model found at {filename}")


# Actor Network
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super(Actor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size // 2),  # Added extra layer
            nn.Tanh()
        )
        
        # Mean and log_std layers
        self.mean_layer = nn.Linear(hidden_size // 2, act_dim)
        self.log_std_layer = nn.Linear(hidden_size // 2, act_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
        
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0.0)
        
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        nn.init.constant_(self.log_std_layer.bias, -1.0)  # Start with smaller std for more stable initial actions
    
    def forward(self, x):
        features = self.network(x)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        
        # Constrain log_std to prevent numerical instability
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        return mean, std


# Critic Network
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        super(Critic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size // 2),  # Added extra layer
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x):
        return self.network(x)


# Training visualization
class TrainingVisualizer:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.start_time = time.time()
    
    def log_episode(self, episode, steps, total_reward):
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.episode_times.append(time.time() - self.start_time)
        
        # Calculate statistics
        avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
        avg_length = np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0
        
        # Print progress
        print(f"Episode {episode} finished after {steps} steps with total reward {total_reward:.2f}")
        print(f"Average reward (last 10): {avg_reward:.2f}, Average length (last 10): {avg_length:.2f}")
        print(f"Training time: {time.time() - self.start_time:.2f} seconds")
        print("-" * 50)


if __name__ == "__main__":
    # Create the environment with rendering
    env = HumanoidEnv(render=True)
    observation_space = env.observation_space
    action_space = env.action_space
    
    # Initialize the PPO agent with tuned hyperparameters
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
    
    # Initialize visualizer
    visualizer = TrainingVisualizer()
    
    # Training loop
    try:
        for episode in range(1000):  # Train for many episodes
            observation = env.reset()
            total_reward = 0
            
            for step in range(env.max_episode_steps):
                # Get action from the PPO agent
                action = agent.select_action(observation)
                
                # Take action in the environment
                next_observation, reward, done, info = env.step(action)
                total_reward += reward
                
                # Update the PPO agent
                agent.update(observation, action, reward, next_observation, done)
                
                # Update observation
                observation = next_observation
                
                # Render the environment
                env.render()
                
                if done:
                    break
            
            # Log episode statistics
            visualizer.log_episode(episode, step + 1, total_reward)
            
            # Save model periodically
            if (episode + 1) % 50 == 0:  # Save more frequently (every 50 episodes)
                agent.save(f"model_episode_{episode+1}.pth")
    
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        env.close()
