#humanoid_env.py
import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import os

class HumanoidEnv(gym.Env):
    def __init__(self, render=False, max_episode_steps=1000):
        super(HumanoidEnv, self).__init__()
        
        # Connect to the physics server
        self.render_mode = render
        if render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        # Set up the simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # Load the humanoid model
        self.humanoid_id = None
        
        # Define action and observation spaces
        # Action space: control signals for each joint
        self.num_joints = 17  # Based on the URDF file
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        
        # Observation space: joint positions, velocities, and robot state
        # We'll observe joint positions, velocities, and the robot's orientation and position
        obs_dim = self.num_joints * 2 + 6  # joint pos, vel + orientation(3) + position(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Joint indices mapping
        self.joint_indices = {}
        self.revolute_joints = []
        
        # Maximum episode steps
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Initial position and orientation
        self.initial_pos = [0, 0, 0.02]  # Starting precisely on ground surface
        self.initial_orn = p.getQuaternionFromEuler([0, 0, 0])
        
        # Load the plane
        p.loadURDF("plane.urdf")
        
        # Observation normalization
        self.obs_mean = None
        self.obs_var = None
        self.obs_count = 0
        self.enable_obs_normalization = True
        
        # Reward normalization
        self.reward_mean = 0
        self.reward_var = 1
        self.reward_count = 0
        self.enable_reward_normalization = True
        
        # Initial joint positions (slightly bent knees and arms for better stability)
        self.initial_joint_positions = {}
        
        # Initialize the environment
        self.reset()
    
    def reset(self):
        # Reset the simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        
        # Load the humanoid model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try different possible paths for the URDF file
        possible_paths = [
            os.path.join(current_dir, "urdf/humanoidV3.urdf"),
            os.path.join(current_dir, "urdf", "humanoidV3.urdf"),
            os.path.join(current_dir, "urdf", "humanoid.urdf"),
            os.path.join(current_dir, "humanoidV3.urdf")
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Found URDF at: {model_path}")
                break
        
        if model_path is None:
            raise FileNotFoundError("Could not find humanoid URDF file. Please ensure it exists in the correct location.")
        
        self.humanoid_id = p.loadURDF(
            model_path,
            basePosition=self.initial_pos,
            baseOrientation=self.initial_orn,
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION
        )
        
        # Get joint information
        self.num_joints = p.getNumJoints(self.humanoid_id)
        print(f"\nTotal joints: {self.num_joints}")
        
        # Identify revolute joints (the ones we can control)
        self.revolute_joints = []
        self.joint_indices = {}
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.humanoid_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            print(f"Joint {i}: {joint_name} (Type: {joint_type})")
            
            if joint_type == p.JOINT_REVOLUTE:
                self.revolute_joints.append(joint_name)
                self.joint_indices[joint_name] = i
        
        print(f"\nRevolute Joints Found: {len(self.revolute_joints)}")
        
        # Initialize observation normalization if not already done
        if self.obs_mean is None:
            obs_dim = len(self.revolute_joints) * 2 + 6  # joint pos, vel + orientation(3) + position(3)
            self.obs_mean = np.zeros(obs_dim)
            self.obs_var = np.ones(obs_dim)
        
        # Set initial joint positions for better stability
        self._set_initial_joint_positions()
        
        # Let the model settle briefly
        for _ in range(5):
            p.stepSimulation()
        
        # Reset step counter
        self.current_step = 0
        
        # Return initial observation
        observation = self._get_observation()
        
        # Update observation statistics for normalization
        if self.enable_obs_normalization:
            self._update_obs_stats(observation)
            
        return self._normalize_observation(observation) if self.enable_obs_normalization else observation
    
    def _set_initial_joint_positions(self):
        """Set initial joint positions for better stability"""
        # Initialize with slight bends in knees and arms
        for joint_name in self.revolute_joints:
            joint_index = self.joint_indices[joint_name]
            joint_info = p.getJointInfo(self.humanoid_id, joint_index)
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            
            # Set knee joints to slightly bent position
            if 'servo6_val' in joint_name or 'servo5_val' in joint_name:
                p.resetJointState(self.humanoid_id, joint_index, 0.2)
            # Set ankle joints to slightly bent position
            elif 'servo8_val' in joint_name or 'servo7_val' in joint_name:
                p.resetJointState(self.humanoid_id, joint_index, 0.1)
            # Set hip joints to slightly bent position
            elif 'servo4_val' in joint_name or 'servo3_val' in joint_name:
                p.resetJointState(self.humanoid_id, joint_index, 0.1)
            # Set arm joints to slightly bent position
            elif 'servo' in joint_name and ('13' in joint_name or '14' in joint_name or '15' in joint_name or '16' in joint_name):
                p.resetJointState(self.humanoid_id, joint_index, 0.1)
            else:
                # Set other joints to middle position
                mid_pos = (lower_limit + upper_limit) / 2
                p.resetJointState(self.humanoid_id, joint_index, mid_pos)
    
    def step(self, action):
        # Apply action to each joint
        for i, joint_name in enumerate(self.revolute_joints):
            if i >= len(action):
                break  # Prevent index out of bounds
                
            joint_index = self.joint_indices[joint_name]
            # Scale action from [-1, 1] to joint limits
            joint_info = p.getJointInfo(self.humanoid_id, joint_index)
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            
            # Scale the action to the joint limits
            scaled_action = lower_limit + (action[i] + 1.0) * 0.5 * (upper_limit - lower_limit)
            
            # Apply position control
            p.setJointMotorControl2(
                self.humanoid_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=scaled_action,
                force=100.0
            )
        
        # Step the simulation
        p.stepSimulation()
        
        # Get new observation
        observation = self._get_observation()
        
        # Update observation statistics for normalization
        if self.enable_obs_normalization:
            self._update_obs_stats(observation)
        
        # Calculate reward
        reward = self._compute_reward()
        
        # Update reward statistics for normalization
        if self.enable_reward_normalization:
            self._update_reward_stats(reward)
            reward = self._normalize_reward(reward)
        
        # Check if episode is done
        done = self._is_done()
        
        # Increment step counter
        self.current_step += 1
        
        # Check if max steps reached
        if self.current_step >= self.max_episode_steps:
            done = True
        
        # Return step information
        info = {}
        return (self._normalize_observation(observation) if self.enable_obs_normalization else observation), reward, done, info
    
    def _get_observation(self):
        # Get joint states
        joint_states = []
        for joint_name in self.revolute_joints:
            joint_index = self.joint_indices[joint_name]
            joint_state = p.getJointState(self.humanoid_id, joint_index)
            joint_position = joint_state[0]
            joint_velocity = joint_state[1]
            joint_states.extend([joint_position, joint_velocity])
        
        # Get base position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.humanoid_id)
        
        # Convert quaternion to Euler angles
        orn = p.getEulerFromQuaternion(orn)
        
        # Combine all observations
        observation = np.array(joint_states + list(orn) + list(pos), dtype=np.float32)
        
        return observation
    
    def _update_obs_stats(self, observation):
        """Update observation statistics for normalization"""
        self.obs_count += 1
        delta = observation - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = observation - self.obs_mean
        self.obs_var += delta * delta2
    
    def _normalize_observation(self, observation):
        """Normalize observation using running statistics"""
        if self.obs_count > 1:
            var = self.obs_var / (self.obs_count - 1) if self.obs_count > 1 else np.ones_like(self.obs_var)
            var = np.maximum(var, 1e-6)  # Avoid division by zero
            return (observation - self.obs_mean) / np.sqrt(var)
        return observation
    
    def _update_reward_stats(self, reward):
        """Update reward statistics for normalization"""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_var += delta * delta2
    
    def _normalize_reward(self, reward):
        """Normalize reward using running statistics"""
        if self.reward_count > 1:
            var = self.reward_var / (self.reward_count - 1) if self.reward_count > 1 else 1.0
            var = max(var, 1e-6)  # Avoid division by zero
            return (reward - self.reward_mean) / np.sqrt(var)
        return reward
    
    def _compute_reward(self):
        # Get base position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.humanoid_id)
        
        # Calculate height reward (encourage standing)
        height = pos[2]
        target_height = 0.5  # Target standing height
        height_reward = 2.0 * (height - 0.02)  # Reward for standing up from initial position
        
        # Calculate orientation reward (encourage upright orientation)
        orn_euler = p.getEulerFromQuaternion(orn)
        upright_reward = 1.0 - abs(orn_euler[0]) - abs(orn_euler[1])  # Penalize tilting
        
        # Calculate velocity reward (encourage forward movement)
        linear_vel, angular_vel = p.getBaseVelocity(self.humanoid_id)
        forward_vel = linear_vel[0]  # X-axis velocity
        velocity_reward = forward_vel  # Reward for moving forward
        
        # Calculate stability reward (encourage stable posture)
        stability_reward = 0.5 * (1.0 - min(0.3, abs(linear_vel[1])) / 0.3)  # Penalize sideways movement
        
        # Calculate energy penalty (discourage excessive movements)
        energy_penalty = 0
        for joint_name in self.revolute_joints:
            joint_index = self.joint_indices[joint_name]
            joint_state = p.getJointState(self.humanoid_id, joint_index)
            joint_velocity = joint_state[1]
            energy_penalty += joint_velocity ** 2
        energy_penalty *= 0.005  # Scale down the penalty
        
        # Calculate total reward
        reward = 3.0 * height_reward + 2.0 * upright_reward + 1.5 * velocity_reward + stability_reward - energy_penalty
        
        # Add large penalty for falling
        if height < 0.1:  # If the humanoid is too low (fallen)
            reward -= 50
        
        return reward
    
    def _is_done(self):
        # Get base position
        pos, _ = p.getBasePositionAndOrientation(self.humanoid_id)
        
        # Check if the humanoid has fallen
        if pos[2] < 0.1:  # If height is too low (adjusted for starting on ground)
            return True
        
        return False
    
    def render(self, mode='human'):
        if not self.render_mode:
            print("Rendering is not enabled. Initialize the environment with render=True to enable rendering.")
            return
        
        # Camera settings
        base_pos, _ = p.getBasePositionAndOrientation(self.humanoid_id)
        camera_distance = 3.0
        camera_yaw = 0
        camera_pitch = -30
        p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, base_pos)
        
        # Pause to allow visualization
        time.sleep(0.01)
    
    def close(self):
        p.disconnect(self.client)