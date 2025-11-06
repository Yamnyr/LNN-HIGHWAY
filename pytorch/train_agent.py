# train_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import highway_env
from collections import deque
import matplotlib.pyplot as plt
import os
import json
import argparse
import sys

print("✅ Script démarré avec succès !")
print("Arguments:", sys.argv)

# ================================
# PRIORITIZED REPLAY BUFFER
# ================================
class PrioritizedReplayBuffer:
    def __init__(self, max_size=20000, alpha=0.6):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)

    def sample(self, batch_size, beta=0.4):
        size = len(self.buffer)
        if size == 0:
            return None
        batch_size = min(batch_size, size)
        priorities = np.array(self.priorities, dtype=np.float64)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(size, batch_size, p=probs, replace=False)

        weights = (size * probs[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-8)

        samples = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*samples)

        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32), weights.astype(np.float32), indices)

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = float(abs(error)) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def size(self):
        return len(self.buffer)


# ================================
# REWARD SHAPING
# ================================
def compute_custom_reward(obs, action, base_reward, info, prev_obs=None):
    custom_reward = base_reward

    if 'speed' in info:
        speed = info['speed']
        target_speed = 30.0
        speed_ratio = min(speed / target_speed, 1.0)
        speed_reward = 0.5 * speed_ratio
        custom_reward += speed_reward
        if speed < 18.0:
            custom_reward -= 0.2

    if info.get('crashed', False):
        custom_reward -= 15.0
        return custom_reward

    custom_reward += 0.1

    if prev_obs is not None:
        try:
            ego_x_prev = prev_obs[0, 1]
            ego_x_curr = obs[0, 1]
            if ego_x_curr > ego_x_prev + 0.5:
                custom_reward += 0.2
            distance_traveled = ego_x_curr - ego_x_prev
            if distance_traveled > 0:
                custom_reward += 0.1 * distance_traveled
        except Exception:
            pass

    if prev_obs is not None and action in [0, 2]:
        try:
            vehicles = obs[1:, :2]
            vehicles_ahead = vehicles[:, 0] * vehicles[:, 1]
            valid = vehicles_ahead[vehicles_ahead > 0]
            min_distance = np.min(valid) if valid.size > 0 else 1.0
            if min_distance > 0.2:
                custom_reward += 0.8
            else:
                custom_reward -= 0.3
        except Exception:
            pass

    if action == 1 and prev_obs is not None:
        idle_count = getattr(compute_custom_reward, 'idle_count', 0) + 1
        compute_custom_reward.idle_count = idle_count
        if idle_count > 5:
            custom_reward -= 0.05
    else:
        compute_custom_reward.idle_count = 0

    return custom_reward


# ================================
# DUELING LSTM NETWORK
# ================================
class DuelingLSTMNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=64):
        super(DuelingLSTMNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.value_fc = nn.Linear(hidden_dim, 32)
        self.value = nn.Linear(32, 1)

        self.adv_fc = nn.Linear(hidden_dim, 32)
        self.adv = nn.Linear(32, n_actions)

    def forward(self, x):
        b, seq, sdim = x.shape
        x = x.view(b * seq, sdim)
        x = F.relu(self.fc1(x))
        x = x.view(b, seq, -1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]

        value = F.relu(self.value_fc(out))
        value = self.value(value)

        adv = F.relu(self.adv_fc(out))
        adv = self.adv(adv)

        q = value + adv - adv.mean(dim=1, keepdim=True)
        return q


# ================================
# AGENT
# ================================
class ImprovedLNNAgentTorch:
    def __init__(self, state_dim, n_actions, device=None, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.2, epsilon_decay=0.9995, lr=3e-4):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_network = DuelingLSTMNetwork(self.state_dim, self.n_actions).to(self.device)
        self.target_network = DuelingLSTMNetwork(self.state_dim, self.n_actions).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = PrioritizedReplayBuffer(max_size=20000)
        self.train_step_count = 0

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_action(self, state, training=True):
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        self.q_network.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).view(1, 1, -1).to(self.device)
            q = self.q_network(s)
            action = int(torch.argmax(q, dim=1).item())
        self.q_network.train()
        return action

    def train_step(self, batch_size=128):
        if self.replay_buffer.size() < batch_size:
            return 0.0

        beta = min(1.0, 0.4 + 0.6 * (self.train_step_count / 10000.0))
        self.train_step_count += 1

        sample = self.replay_buffer.sample(batch_size, beta=beta)
        if sample is None:
            return 0.0

        states, actions, rewards, next_states, dones, weights, indices = sample
        bs = states.shape[0]

        states_t = torch.FloatTensor(states).view(bs, 1, -1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).view(bs, 1, -1).to(self.device)
        actions_t = torch.LongTensor(actions).view(bs, 1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        q_values = self.q_network(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_network(next_states_t).argmax(dim=1, keepdim=True)
            next_q_target = self.target_network(next_states_t).gather(1, next_actions).squeeze(1)
            target = rewards_t + self.gamma * next_q_target * (1.0 - dones_t)

        td_errors = (target - q_values).detach().cpu().numpy()
        loss_per_sample = F.smooth_l1_loss(q_values, target, reduction='none')
        loss = (weights_t * loss_per_sample).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_errors)
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="lnn_agent_torch.pth"):
        weights_path = path if path.endswith('.pth') else path + '.pth'
        torch.save(self.q_network.state_dict(), weights_path)

        config_path = weights_path.replace('.pth', '_config.json')
        config = {
            'state_dim': int(self.state_dim),
            'n_actions': int(self.n_actions),
            'gamma': float(self.gamma),
            'epsilon': float(self.epsilon),
            'epsilon_min': float(self.epsilon_min),
            'epsilon_decay': float(self.epsilon_decay)
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)
        print(f"💾 Modèle sauvegardé: {weights_path}")

    def load(self, path="lnn_agent_torch.pth"):
        weights_path = path if path.endswith('.pth') else path + '.pth'
        config_path = weights_path.replace('.pth', '_config.json')
        if os.path.exists(weights_path) and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.state_dim = config['state_dim']
            self.n_actions = config['n_actions']
            self.gamma = config['gamma']
            self.epsilon = config['epsilon']
            self.epsilon_min = config['epsilon_min']
            self.epsilon_decay = config['epsilon_decay']

            self.q_network = DuelingLSTMNetwork(self.state_dim, self.n_actions).to(self.device)
            self.q_network.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.update_target_network()
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=3e-4)
            print(f"📂 Modèle chargé: {weights_path}")
        else:
            raise FileNotFoundError(f"Fichiers introuvables: {weights_path} ou {config_path}")


# ================================
# TRAIN LOOP
# ================================
def train_agent_improved(env_name="highway-fast-v0", episodes=300, max_steps=150,
                         save_path="lnn_agent_torch.pth", device=None):
    env_config = {
        "observation": {"type": "Kinematics", "vehicles_count": 10, "features": ["presence","x","y","vx","vy"], "normalize": True, "absolute": False},
        "lanes_count": 4, "vehicles_count": 50, "duration": 40, "collision_reward": -10, "reward_speed_range": [20,30],
        "normalize_reward": False, "offroad_terminal": True
    }

    env = gym.make(env_name, config=env_config)

    obs_space = env.observation_space
    state_dim = int(np.prod(obs_space.shape)) if hasattr(obs_space, 'shape') and len(obs_space.shape)>=2 else int(obs_space.shape[0])
    n_actions = env.action_space.n

    print(f"🚗 État: {state_dim} dimensions | 🎮 Actions: {n_actions} | 🎯 Objectif: apprendre à conduire vite et sûr\n")

    agent = ImprovedLNNAgentTorch(state_dim, n_actions, device=device)

    episode_rewards, raw_rewards, losses = [], [], []
    best_reward = -float('inf')

    for episode in range(episodes):
        state, _ = env.reset()
        state_flat = state.flatten()
        prev_state = state.copy()
        episode_reward = 0.0
        episode_loss = []

        for step in range(max_steps):
            action = agent.get_action(state_flat)
            next_state, base_reward, terminated, truncated, info = env.step(action)
            next_state_flat = next_state.flatten()
            done = bool(terminated or truncated)

            shaped_reward = compute_custom_reward(next_state, action, base_reward, info, prev_state)
            agent.replay_buffer.add(state_flat, action, shaped_reward, next_state_flat, done)

            if agent.replay_buffer.size() >= 128:
                loss = agent.train_step(batch_size=256)
                episode_loss.append(loss)

            episode_reward += shaped_reward
            state_flat = next_state_flat
            prev_state = next_state.copy()

            if done:
                break

        if episode % 5 == 0:
            agent.update_target_network()
        agent.decay_epsilon()

        episode_rewards.append(episode_reward)
        avg_loss = float(np.mean(episode_loss)) if episode_loss else 0.0
        losses.append(avg_loss)

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(save_path)

        if episode % 10 == 0:
            print(f"Épisode {episode}/{episodes} | Reward: {episode_reward:.2f} | Moy10: {np.mean(episode_rewards[-10:]):.2f} | Best: {best_reward:.2f} | Loss: {avg_loss:.4f} | ε: {agent.epsilon:.3f}")

    env.close()
    return agent, episode_rewards


# TEST AGENT
# ================================
def test_agent(agent, env_name="highway-fast-v0", episodes=10, render=False):
    env_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "normalize": True,
            "absolute": False
        },
        "lanes_count": 4,
        "vehicles_count": 50,
        "duration": 40,
        "collision_reward": -10,
        "reward_speed_range": [20, 30],
        "normalize_reward": False,
        "offroad_terminal": True
    }
    render_mode = "human" if render else None
    env = gym.make(env_name, config=env_config, render_mode=render_mode)

    for episode in range(episodes):
        state, _ = env.reset()
        state_flat = state.flatten()
        done = False
        total_reward = 0
        while not done:
            action = agent.get_action(state_flat, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            state_flat = next_state.flatten()
            total_reward += reward
            done = bool(terminated or truncated)
        print(f"Episode {episode+1}/{episodes} - Reward total: {total_reward:.2f}")
    env.close()


# ================================
# DEMO VISUELLE
# ================================
def demo_agent(model_path="lnn_agent_torch.pth", env_name="highway-fast-v0"):
    config_path = model_path.replace('.pth', '_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Le fichier de config est manquant: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    agent = ImprovedLNNAgentTorch(state_dim=config['state_dim'], n_actions=config['n_actions'])
    agent.load(model_path)
    test_agent(agent, env_name=env_name, episodes=1, render=True)


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train/test/demo')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--model', type=str, default='lnn_agent_torch.pth')
    args = parser.parse_args()

    if args.mode == 'train':
        train_agent_improved(episodes=args.episodes, save_path=args.model)
    elif args.mode == 'test':
        # Charger le config du modèle pour récupérer state_dim et n_actions
        config_path = args.model.replace('.pth', '_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Le fichier de config est manquant: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        agent = ImprovedLNNAgentTorch(state_dim=config['state_dim'], n_actions=config['n_actions'])
        agent.load(args.model)
        test_agent(agent, episodes=args.episodes)

    elif args.mode == 'demo':
        demo_agent(model_path=args.model)
    else:
        raise ValueError(f"Mode inconnu: {args.mode}")
