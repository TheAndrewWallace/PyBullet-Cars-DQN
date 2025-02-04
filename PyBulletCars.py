"""
DQN PyBullet Two-Car Environment

"""

import os
import random
from collections import deque
from typing import Tuple, List

import numpy as np
import pybullet as p
import pybullet_data
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class TwoCarEnvironment:
    """
    A two-car environment in PyBullet featuring two husky robots navigating towards their respective targets.
    """
    def __init__(self, gui: bool = True):
        """
        Initialise the environment.
        
        :param gui: Whether to display the PyBullet GUI.
        """
        connection_mode = p.GUI if gui else p.DIRECT
        self.physics_client = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.time_step = 0.05
        self.steps_per_choice = 50
        p.setTimeStep(self.time_step)
        p.setRealTimeSimulation(0)
        
        self.plane_id = p.loadURDF("plane.urdf")
        self.target_positions = [[2, 2, 0], [-2, -2, 0]]
        self.start_positions = [[-2, -2, 0.3], [2, 2, 0.3]]
        self.start_orientations = [
            p.getQuaternionFromEuler([0, 0, 0]),
            p.getQuaternionFromEuler([0, 0, 0])
        ]
        
        # Initial positions, orientations, speeds and rotations.
        self.agent_positions = self.start_positions.copy()
        self.agent_orientations = self.start_orientations.copy()
        self.agent_speeds = [0.0, 0.0]
        self.agent_rotations = [0.0, 0.0]
        
        self.huskies = [
            p.loadURDF("husky/husky.urdf", self.start_positions[0], self.start_orientations[0]),
            p.loadURDF("husky/husky.urdf", self.start_positions[1], self.start_orientations[1])
        ]
        self.reset()

    def reset(self) -> None:
        """
        Reset the environment to its initial state.
        """
        for i in range(2):
            p.resetBasePositionAndOrientation(self.huskies[i],
                                              self.agent_positions[i],
                                              self.agent_orientations[i])
            self.agent_speeds[i] = 0.0
            self.agent_rotations[i] = 0.0

    def get_state(self, agent_index: int) -> np.ndarray:
        """
        Retrieve the state vector for a given agent.
        
        The state comprises:
            - The agent's current position (3),
            - The agent's target position (3),
            - The agent's current yaw (1) obtained from its orientation,
            - The agent's current speed (1),
            - The other agent's current position (3),
            - The other agent's target position (3),
            - The other agent's current yaw (1),
            - The other agent's current speed (1).
            
        Total state dimension: 16
        
        :param agent_index: Index of the agent (0 or 1).
        :return: A numpy array representing the state.
        """
        # Get current position and orientation for the selected agent.
        pos, orn = p.getBasePositionAndOrientation(self.huskies[agent_index])
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]  # Only the yaw is relevant on a planar surface.
        speed = self.agent_speeds[agent_index]

        # For the other agent.
        other_agent_index = 1 - agent_index
        other_pos, other_orn = p.getBasePositionAndOrientation(self.huskies[other_agent_index])
        other_euler = p.getEulerFromQuaternion(other_orn)
        other_yaw = other_euler[2]
        other_speed = self.agent_speeds[other_agent_index]

        state = np.concatenate([
            np.array(pos),                                  # Agent's position (3)
            np.array(self.target_positions[agent_index]),   # Agent's target (3)
            np.array([yaw]),                                # Agent's yaw (1)
            np.array([speed]),                              # Agent's speed (1)
            np.array(other_pos),                            # Other agent's position (3)
            np.array(self.target_positions[other_agent_index]),  # Other agent's target (3)
            np.array([other_yaw]),                          # Other agent's yaw (1)
            np.array([other_speed])                         # Other agent's speed (1)
        ])
        return state

    def step(self, actions: List[int]) -> Tuple[np.ndarray, float, np.ndarray, float, bool]:
        """
        Execute a simulation step using the provided actions for both agents.
        
        Actions:
            0: Accelerate
            1: Decelerate
            2: Increase turn (rotate clockwise)
            3: Decrease turn (rotate anticlockwise)
            4: Do nothing
        
        
        :param actions: List of actions for each agent.
        :return: Tuple containing state_agent0, reward_agent0, state_agent1, reward_agent1, and a boolean done flag.
        """
        # Calculate previous distances to the targets for reward shaping.
        prev_distances = []
        for i in range(2):
            pos, _ = p.getBasePositionAndOrientation(self.huskies[i])
            target = self.target_positions[i]
            distance = np.linalg.norm(np.array(pos[:2]) - np.array(target[:2]))
            prev_distances.append(distance)

        # Process each agent's action.
        for i in range(2):
            if actions[i] == 0:
                self.agent_speeds[i] = min(self.agent_speeds[i] + 0.05, 10)
            elif actions[i] == 1:
                self.agent_speeds[i] = max(self.agent_speeds[i] - 0.05, -10)
            elif actions[i] == 2:
                self.agent_rotations[i] += 0.05
            elif actions[i] == 3:
                self.agent_rotations[i] -= 0.05
            # Action 4 corresponds to doing nothing.
            
            # Apply motor controls to the husky wheels.
            left_velocity = self.agent_speeds[i] + self.agent_rotations[i]
            right_velocity = self.agent_speeds[i] - self.agent_rotations[i]
            p.setJointMotorControl2(self.huskies[i], 2, p.VELOCITY_CONTROL, targetVelocity=left_velocity)
            p.setJointMotorControl2(self.huskies[i], 4, p.VELOCITY_CONTROL, targetVelocity=left_velocity)
            p.setJointMotorControl2(self.huskies[i], 3, p.VELOCITY_CONTROL, targetVelocity=right_velocity)
            p.setJointMotorControl2(self.huskies[i], 5, p.VELOCITY_CONTROL, targetVelocity=right_velocity)
        
        for _ in range(self.steps_per_choice):
            p.stepSimulation()

        # Calculate new distances to the targets.
        new_distances = []
        for i in range(2):
            pos, _ = p.getBasePositionAndOrientation(self.huskies[i])
            target = self.target_positions[i]
            new_distance = np.linalg.norm(np.array(pos[:2]) - np.array(target[:2]))
            new_distances.append(new_distance)
            
        # Initialise rewards and done flags.
        rewards = [0.0, 0.0]
        done = [False, False]

        # Compute rewards and termination conditions.
        for i in range(2):
            # Reward for reaching the target.
            if new_distances[i] <= 1:  # Using the distance threshold.
                rewards[i] = 1000.0
                done[i] = True

            # Penalty for moving too far from the centre.
            pos, _ = p.getBasePositionAndOrientation(self.huskies[i])
            distance_from_centre = np.linalg.norm([pos[0], pos[1]])
            if distance_from_centre > 10:
                rewards[i] -= 50.0

            # Reward shaping: reward for reducing the distance to the target.
            shaping = prev_distances[i] - new_distances[i]
            rewards[i] += shaping * 200

        # Penalty for collision.
        pos0, _ = p.getBasePositionAndOrientation(self.huskies[0])
        pos1, _ = p.getBasePositionAndOrientation(self.huskies[1])
        if (abs(pos0[0] - pos1[0]) <= 1 and abs(pos0[1] - pos1[1]) <= 1):
            rewards = [-1.0, -1.0]

        state_agent0 = self.get_state(0)
        state_agent1 = self.get_state(1)
        return state_agent0, rewards[0], state_agent1, rewards[1], all(done)

    def close(self) -> None:
        """
        Disconnect the PyBullet simulation.
        """
        p.disconnect()


class DQN(nn.Module):
    """
    Deep Q-Network model.
    """
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initialise the DQN model.
        
        :param state_dim: Dimension of the input state.
        :param action_dim: Number of possible actions.
        """
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        :param x: Input tensor.
        :return: Output Q-values.
        """
        return self.fc(x)


class ReplayBuffer:
    """
    Experience Replay Buffer for storing transitions.
    """
    def __init__(self, capacity: int):
        """
        Initialise the replay buffer.
        
        :param capacity: Maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, agent_index: int) -> None:
        """
        Add a transition to the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done, agent_index))

    def sample(self, batch_size: int, agent_index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions for a specific agent.
        
        :param batch_size: Number of transitions to sample.
        :param agent_index: The agent index to filter transitions.
        :return: Tuple of (states, actions, rewards, next_states, dones) as tensors.
        """
        agent_experiences = [exp for exp in self.buffer if exp[5] == agent_index]
        actual_batch_size = min(batch_size, len(agent_experiences))
        batch = random.sample(agent_experiences, actual_batch_size)
        states, actions, rewards, next_states, dones, _ = zip(*batch)
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))

    def __len__(self) -> int:
        """
        Return the current size of the buffer.
        """
        return len(self.buffer)


def save_models(dqn: DQN, target_dqn: DQN, episode: int, save_path: str = "models") -> None:
    """
    Save the DQN and target DQN models.
    
    :param dqn: The primary DQN model.
    :param target_dqn: The target DQN model.
    :param episode: Current episode number.
    :param save_path: Directory in which to save the models.
    """
    os.makedirs(save_path, exist_ok=True)
    torch.save(dqn.state_dict(), os.path.join(save_path, f"dqn_episode_{episode}.pth"))
    torch.save(target_dqn.state_dict(), os.path.join(save_path, f"target_dqn_episode_{episode}.pth"))
    logging.info(f"Models saved at episode {episode}.")


def load_models(dqn: DQN, target_dqn: DQN, load_path: str = "models", episode: int = None) -> None:
    """
    Load the DQN and target DQN models.
    
    :param dqn: The primary DQN model.
    :param target_dqn: The target DQN model.
    :param load_path: Directory from which to load the models.
    :param episode: Specific episode to load, or load the latest if None.
    """
    if episode is None:
        dqn_files = [f for f in os.listdir(load_path)
                     if f.startswith("dqn_episode_") and f.endswith(".pth")]
        if not dqn_files:
            raise FileNotFoundError("No saved models found.")
        try:
            latest_episode = max(int(f.split("_")[2].split(".")[0]) for f in dqn_files)
        except ValueError:
            raise ValueError("No valid episode numbers found in saved model filenames.")
        dqn_path = os.path.join(load_path, f"dqn_episode_{latest_episode}.pth")
        target_dqn_path = os.path.join(load_path, f"target_dqn_episode_{latest_episode}.pth")
    else:
        dqn_path = os.path.join(load_path, f"dqn_episode_{episode}.pth")
        target_dqn_path = os.path.join(load_path, f"target_dqn_episode_{episode}.pth")

    dqn.load_state_dict(torch.load(dqn_path))
    target_dqn.load_state_dict(torch.load(target_dqn_path))
    loaded_episode = episode if episode is not None else latest_episode
    logging.info(f"Models loaded from episode {loaded_episode}.")


def run_demo(dqn: DQN, device: torch.device, demo_steps: int = 80) -> None:
    """
    Run a demonstration simulation using the trained DQN policy.
    
    The function creates a new environment (with the GUI enabled) and then,
    in evaluation mode, runs a single episode using deterministic actions.
    
    :param dqn: Trained DQN model.
    :param device: The torch device.
    :param demo_steps: Maximum number of steps to run in the demonstration episode.
    """
    logging.info("Starting demonstration simulation...")
    env_demo = TwoCarEnvironment(gui=True)
    dqn.eval()  # Set the network to evaluation mode.
    states = [env_demo.get_state(0), env_demo.get_state(1)]
    done = [False, False]
    steps = 0

    while not all(done) and steps < demo_steps:
        actions = []
        for i in range(2):
            state_tensor = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0).to(device)
            action = torch.argmax(dqn(state_tensor)).item()
            actions.append(action)
        next_state_agent0, reward0, next_state_agent1, reward1, done_flag = env_demo.step(actions)
        states = [next_state_agent0, next_state_agent1]
        steps += 1
        time.sleep(0.05)  # Slow down the simulation for visualisation.

    logging.info("Demonstration complete. Closing the demo environment.")
    time.sleep(2)
    env_demo.close()


def main() -> None:
    # Hyperparameters
    state_dim = 16  # Updated state dimension: position (3), target (3), yaw (1), speed (1) for each agent.
    action_dim = 5  # 0: accelerate, 1: decelerate, 2: increase turn, 3: decrease turn, 4: do nothing
    learning_rate = 0.001
    buffer_capacity = 10000
    epsilon_start = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    gamma = 0.99
    batch_size = 64
    num_episodes = 401
    max_steps = 100

    # Initialise environment, models, optimiser and replay buffer.
    env = TwoCarEnvironment(gui=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn = DQN(state_dim, action_dim).to(device)
    target_dqn = DQN(state_dim, action_dim).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    optimiser = optim.Adam(dqn.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(buffer_capacity)

    # Attempt to load previously saved models.
    epsilon = epsilon_start
    try:
        load_models(dqn, target_dqn)
        epsilon = 0.1
    except (FileNotFoundError, ValueError) as e:
        logging.info(f"{e}. Starting training from scratch.")

    # Training loop.
    for episode in range(num_episodes):
        state_agent0 = env.get_state(0)
        state_agent1 = env.get_state(1)
        states = [state_agent0, state_agent1]
        done = [False, False]
        total_rewards = [0.0, 0.0]
        steps = 0

        while not all(done) and steps < max_steps:
            actions = []
            # Decide on actions for each agent using an epsilon-greedy strategy.
            for i in range(2):
                if random.random() < epsilon:
                    action = random.randrange(action_dim)
                else:
                    state_tensor = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0).to(device)
                    action = torch.argmax(dqn(state_tensor)).item()
                actions.append(action)

            next_state_agent0, reward0, next_state_agent1, reward1, done_flag = env.step(actions)
            replay_buffer.push(states[0], actions[0], reward0, next_state_agent0, done_flag, 0)
            replay_buffer.push(states[1], actions[1], reward1, next_state_agent1, done_flag, 1)
            states = [next_state_agent0, next_state_agent1]
            total_rewards[0] += reward0
            total_rewards[1] += reward1
            steps += 1

            # Train on a batch for each agent.
            for agent_idx in range(2):
                if len(replay_buffer) >= batch_size:
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size, agent_idx)
                    state_batch = state_batch.to(device)
                    action_batch = action_batch.to(device)
                    reward_batch = reward_batch.to(device)
                    next_state_batch = next_state_batch.to(device)
                    done_batch = done_batch.to(device)

                    next_q_values = target_dqn(next_state_batch).max(dim=1)[0]
                    target_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)
                    current_q_values = dqn(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

                    loss = loss_fn(current_q_values, target_q_values.detach())
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

        env.reset()
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Save models every 50 episodes.
        if episode % 50 == 0:
            save_models(dqn, target_dqn, episode)
        # Update the target network every 10 episodes.
        if episode % 10 == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        logging.info(f"Episode {episode} | Steps: {steps} | Rewards: {total_rewards} | Epsilon: {epsilon:.3f}")

    env.close()

    # Ask whether to run a demonstration simulation.
    run_demo_choice = input("Training complete. Run demonstration simulation? (y/n): ")
    if run_demo_choice.lower() == 'y':
        run_demo(dqn, device)


if __name__ == '__main__':
    main()
