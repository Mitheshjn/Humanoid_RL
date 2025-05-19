import numpy as np
import gym
from gym import spaces
import pybullet as p
import pybullet_data
import time
import xml.etree.ElementTree as ET

class HumanoidEnv(gym.Env):
    """
    A custom environment for a humanoid robot to learn to walk, based on PyBullet.
    """

    def __init__(self, render=False):
        """
        Initializes the HumanoidEnv.
        """
        self.render_mode = render  # Renamed to avoid conflict with render() method
        self.physics_client_id = -1  # Initialize to an invalid ID
        self.urdf_path = "./urdf/humanoidV3.urdf"  # Replace with the actual path to your URDF file!
        self.humanoid = None
        self.target = None  # Add a target for the robot to walk towards
        self.initial_joint_angles = {}
        self.observation_space = None
        self.action_space = None
        self.joints = []
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        self.joint_ranges = []
        self.episode_time_step = 0
        self.total_reward = 0  # Track cumulative reward
        self.episode_count = 0  # Track the number of episodes
        self.max_episode_steps = 1000  # Maximum steps per episode, adjust as needed.
        self.walk_target_speed = 1.0  # Target walking speed
        self.distance_weight = 0.1
        self.alive_bonus = 5.0
        self.control_cost_weight = 1e-4
        self.healthy_reward = 1.0
        self.termination_penalty = -100.0
        self.fall_penalty = -50  # Add a fall penalty
        self.stability_check_frequency = 10  # Check stability every 10 steps
        self.max_base_height = 3.0  # Maximum height of the robot's base.
        self.max_pitch_roll = 0.4  # Maximum pitch and roll angles in radians (about 23 degrees)
        self.reset_joint_damping = 0.1  # Damping for joint reset
        self.stabilize_duration = 200  # Increased stabilization duration
        self.joint_name_mapping = {}
        self.use_stabilization = True  # Add a flag to control stabilization
        self.joint_index_mapping = {}  # store the mapping
        self.mapped_joint_names = [] # Keep track of mapped joint names
        self.plane_id = None # Store the plane ID
        self.target_visual = None # Store the target visual ID

        # Connect to PyBullet physics server
        self._connect()

        # Set up the simulation
        self._setup_simulation()  
        self._build_action_space()
        self._build_observation_space()

    def _connect(self):
        """
        Connects to the PyBullet physics server.
        """
        if self.render_mode:
            self.physics_client_id = p.connect(
                p.GUI
            )  # Use GUI for visualization if needed.
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        if self.physics_client_id < 0:
            raise Exception("Failed to connect to PyBullet physics server.")
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )  # Needed for plane.urdf and other data files

    def _setup_simulation(self):
        self.plane_id = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.8) 
        self._load_robot()  

    def _load_robot(self):
        """
        Loads the humanoid robot from the URDF file into the simulation.
        """
        # Load the robot
        self.humanoid = p.loadURDF(
            self.urdf_path,
            useFixedBase=False, # The base of the robot is not fixed
            flags=p.URDF_USE_SELF_COLLISION, # Collision detection between different links of the same robot
        )

        if self.humanoid is None:
            raise Exception("Failed to load the humanoid robot from the URDF file.")

        # Initialize joint-related lists/dicts
        self.joints = []
        self.joint_index_mapping = {}
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        self.joint_ranges = []
        self.initial_joint_angles = {}

        num_joints = p.getNumJoints(self.humanoid) # Gets the total number of joints
        
        # Debug print all joints
        print("\nPyBullet Joint Indices and Names:")
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.humanoid, i)
            joint_name = joint_info[1].decode("UTF-8") # Index 1: A byte string representing the name of the joint, thats why decode is used
            print(f"Index {i}: {joint_name} (Type: {joint_info[2]})") # An Index 2: integer representing the type of the joint
            """
            Common types include:
                p.JOINT_REVOLUTE (rotational joint with one degree of freedom)
                p.JOINT_PRISMATIC (translational joint with one degree of freedom)
                p.JOINT_FIXED (no relative motion between the connected links)
                p.JOINT_SPHERICAL (rotational joint with three degrees of freedom)
                p.JOINT_PLANAR (translational joint in a plane with two degrees of freedom and one rotational)
            """

        # Collect revolute joints and their indices, cuz its important for RL 
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.humanoid, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]
            
            if joint_type == p.JOINT_REVOLUTE:
                # Store direct joint name without mapping
                self.joints.append(joint_name)
                self.joint_index_mapping[joint_name] = i
                
                # Get joint limits
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                joint_range = upper_limit - lower_limit
                
                # Store joint properties
                self.joint_limits_lower.append(lower_limit)
                self.joint_limits_upper.append(upper_limit)
                self.joint_ranges.append(joint_range)
                self.initial_joint_angles[joint_name] = (lower_limit + upper_limit) / 2.0

        if not self.joints:
            raise Exception("No revolute joints found in the URDF. The robot may not be controllable.")

        # Debug prints
        print("\nRevolute Joints Found:")
        print(self.joints)
        print("\nJoint Index Mapping:")
        print(self.joint_index_mapping)
        print("\nJoint Limits (Lower/Upper):")
        for name in self.joints:
            idx = self.joints.index(name)
            print(f"{name}: [{self.joint_limits_lower[idx]:.3f}, {self.joint_limits_upper[idx]:.3f}]")

        # Store initial base position
        self.initial_position = p.getBasePositionAndOrientation(self.humanoid)[0]

    def _build_action_space(self):
        """
        Builds the action space for the environment.  This is a Gym spaces.Box.
        """
        # Number of controllable joints
        num_actions = len(self.joints)
        self.action_space = spaces.Box(
            low=np.array(self.joint_limits_lower),
            high=np.array(self.joint_limits_upper),
            shape=(num_actions,),
            dtype=np.float32,
        )

    def _build_observation_space(self):
        """
        Builds the observation space for the environment. This is a Gym spaces.Box.
        """
        # More comprehensive observation space, including:
        # - Joint positions and velocities
        # - Robot's base position and orientation
        # - Target position
        # - Velocity of the robot.
        num_joints = len(self.joints)
        observation_dim = (
            num_joints * 2 + 6 + 3 + 3
        )  # Joint positions, velocities, base pos/ori, target pos, base velocity
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32
        )

    def reset(self):
        """
        Resets the environment to its initial state.
        """
        self.episode_time_step = 0
        self.total_reward = 0
        self.episode_count += 1

        # Validate physics client connection
        if not p.isConnected():
            self._connect()  # Attempt to reconnect if disconnected
            self._setup_simulation()

        # Reset base first
        p.resetBasePositionAndOrientation(
            self.humanoid,
            self.initial_position,
            p.getQuaternionFromEuler([0, 0, 0])
        )
        
        # Reset joint angles and velocities
        for joint_name in self.joints:
            if joint_name in self.initial_joint_angles:
                initial_angle = self.initial_joint_angles[joint_name]
                joint_index = self.joint_index_mapping.get(joint_name, -1)
                
                if joint_index == -1:
                    print(f"Warning: Joint {joint_name} not found in joint_index_mapping. Skipping reset.")
                    continue
                    
                p.resetJointState(self.humanoid, jointIndex=joint_index, targetValue=initial_angle, targetVelocity=0)

        # Generate new target
        self.target = [
            np.random.uniform(-5, 5),
            np.random.uniform(-5, 5),
            0,
        ]

        # Update visualization
        if self.render_mode:
            if self.target_visual is not None:
                p.removeBody(self.target_visual)
            
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.2,
                rgbaColor=[1, 0, 0, 1],
            )
            
            self.target_visual = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=self.target,
            )

        # Stabilization phase
        if self.use_stabilization:
            for _ in range(self.stabilize_duration):
                p.stepSimulation()
                if self.render_mode:
                    time.sleep(1/240)  # Add small delay if using GUI

        # After stabilization, apply joint control 
        for joint_name in self.joints:
            if joint_name in self.initial_joint_angles:
                initial_angle = self.initial_joint_angles[joint_name]
                joint_index = self.joint_index_mapping.get(joint_name, -1)
                
                if joint_index == -1:
                    print(f"Warning: Joint {joint_name} not found in joint_index_mapping. Skipping motor control.")
                    continue
                    
                # Apply position control with reasonable parameters
                p.setJointMotorControl2(
                    bodyIndex=self.humanoid,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=initial_angle,
                    positionGain=0.3,
                    velocityGain=0.05,
                    force=300
                )

        return self._get_observation()

    def _get_observation(self):
        """
        Gets the current observation of the environment.

        Returns:
            numpy.ndarray: The observation array.
        """
        # Get base position and orientation
        base_pos, base_ori = p.getBasePositionAndOrientation(self.humanoid)
        base_linear_velocity, base_angular_velocity = p.getBaseVelocity(self.humanoid)

        # Get joint positions and velocities
        joint_indices = []
        for joint_name in self.joints:
            idx = self.joint_index_mapping.get(joint_name, -1)
            if idx != -1:
                joint_indices.append(idx)
            else:
                print(f"Warning: Joint {joint_name} not found in mapping during observation.")
                # Append default values for missing joints to maintain array structure
                joint_indices.append(0)  # This will be handled in the try-except below
        
        joint_positions = []
        joint_velocities = []
        
        try:
            if joint_indices:
                joint_states = p.getJointStates(self.humanoid, joint_indices)
                for i, state in enumerate(joint_states):
                    if joint_indices[i] != 0:  # Skip placeholder indices
                        joint_positions.append(state[0])
                        joint_velocities.append(state[1])
                    else:
                        # Use default values for missing joints
                        joint_positions.append(0.0)
                        joint_velocities.append(0.0)
            else:
                # No valid joints found, use default values
                joint_positions = [0.0] * len(self.joints)
                joint_velocities = [0.0] * len(self.joints)
        except Exception as e:
            print(f"Error getting joint states: {e}")
            # Use default values on error
            joint_positions = [0.0] * len(self.joints)
            joint_velocities = [0.0] * len(self.joints)

        # Flatten the data
        observation = np.concatenate(
            [
                base_pos,
                base_ori[0:4],  # Quaternion orientation
                joint_positions,
                joint_velocities,
                self.target,  # target position
                base_linear_velocity,
            ]
        ).astype(np.float32)
        return observation

    def _apply_action(self, action):
        """
        Applies the given action to the humanoid robot.

        Args:
            action (numpy.ndarray): The action to apply.
        """
        # Clip the action to the joint limits
        clipped_action = np.clip(
            action, self.joint_limits_lower, self.joint_limits_upper
        )
        
        try:
            for i, joint_name in enumerate(self.joints):
                joint_index = self.joint_index_mapping.get(joint_name, -1)
                if joint_index == -1:
                    print(f"Warning: Joint {joint_name} not found in robot model. Skipping.")
                    continue
                    
                p.setJointMotorControl2(
                    bodyIndex=self.humanoid,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=clipped_action[i],
                )
        except Exception as e:
            print(f"Error applying action: {e}")

    def step(self, action):
        """
        Performs one step in the simulation.

        Args:
            action (numpy.ndarray): The action to take.

        Returns:
            tuple: (observation, reward, done, info)
        """
        self._apply_action(action)
        p.stepSimulation()
        self.episode_time_step += 1
        observation = self._get_observation()
        reward, done = self._calculate_reward_and_done(observation)
        self.total_reward += reward  # Accumulate reward.
        info = {
            "episode_reward": self.total_reward,
            "episode_step": self.episode_time_step,
        }
        return observation, reward, done, info

    def _calculate_reward_and_done(self, observation):
        """
        Calculates the reward and done signal.  This is where you define your reward function.

        Args:
            observation (numpy.ndarray): The current observation.

        Returns:
            tuple: (reward, done)
        """
        done = False
        reward = 0

        # Get base position and orientation
        base_pos = observation[0:3]
        base_ori = observation[3:7]
        base_velocity = observation[-3:]

        # Get joint states
        joint_positions = observation[7 : 7 + len(self.joints)]
        
        # Calculate forward progress reward
        forward_progress = base_velocity[0]  # Velocity in the x-direction
        reward += forward_progress

        # Distance to target
        dist_to_target = np.linalg.norm(np.array(base_pos) - np.array(self.target))
        reward += self.distance_weight * -dist_to_target

        # Penalize excessive joint movement (control cost)
        control_cost = np.sum(
            np.square(observation[7 + len(self.joints) : 7 + 2 * len(self.joints)])
        )
        reward -= self.control_cost_weight * control_cost

        # Encourage the robot to stay alive
        reward += self.alive_bonus

        # Check if robot is healthy
        if self.is_healthy(observation):
            reward += self.healthy_reward
        else:
            done = True
            reward += self.termination_penalty
            reward += self.fall_penalty  # Add fall penalty

        # Check for termination conditions (episode length)
        if self.episode_time_step >= self.max_episode_steps:
            done = True

        return reward, done

    def is_healthy(self, observation):
        """
        Check if the robot is in a healthy state (not fallen, within joint limits, etc.).

        Args:
            observation (numpy.ndarray): The current observation.

        Returns:
            bool: True if the robot is healthy, False otherwise.
        """
        base_pos = observation[0:3]
        base_ori = observation[3:7]

        # Example: Check if the robot is upright (using quaternion)
        quat = base_ori
        pitch = np.arcsin(2.0 * (quat[1] * quat[2] + quat[0] * quat[3]))
        roll = np.arctan2(
            2.0 * (quat[0] * quat[1] + quat[2] * quat[3]),
            1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]),
        )
        if np.abs(pitch) > self.max_pitch_roll or np.abs(roll) > self.max_pitch_roll:
            return False

        # Check for joint limits
        joint_positions = observation[7 : 7 + len(self.joints)]
        for i, joint_position in enumerate(joint_positions):
            if (
                joint_position < self.joint_limits_lower[i]
                or joint_position > self.joint_limits_upper[i]
            ):
                return False

        # Additional stability check: base height
        if base_pos[2] > self.max_base_height:
            return False

        return True

    def render(self, mode="human"):
        """
        Renders the environment. This method is called automatically if the
        render flag is set to True during initialization.
        """
        pass  # No need to do anything, the simulation runs with GUI if render is set to true.

    def close(self):
        """
        Closes the environment and disconnects from the physics server.
        """
        if self.physics_client_id >= 0:
            p.disconnect(self.physics_client_id)
            self.physics_client_id = -1