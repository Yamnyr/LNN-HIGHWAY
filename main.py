import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gymnasium as gym
import highway_env
from collections import deque
import matplotlib.pyplot as plt
import os


# Buffer de replay avec expérience prioritaire (optionnel)
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def size(self):
        return len(self.buffer)


def build_ltc_policy(state_dim, n_actions, units=32):
    """
    Construit un réseau LTC (Liquid Neural Network) avec keras-ncp.
    Fallback vers LSTM si keras-ncp est indisponible.
    """
    try:
        import kerasncp as kncp
        from kerasncp.tf import LTCCell

        # Architecture NCP optimisée
        wiring = kncp.wirings.NCP(
            inter_neurons=units,
            command_neurons=units // 2,
            motor_neurons=n_actions,
            sensory_fanout=4,
            inter_fanout=4,
            recurrent_command_synapses=4,
            motor_fanin=6,
        )

        # Création de la cellule LTC
        ltc_cell = LTCCell(wiring)

        # Modèle RNN avec LTC
        inputs = keras.Input(shape=(1, state_dim))
        x = layers.RNN(ltc_cell, return_sequences=False)(inputs)
        outputs = layers.Dense(n_actions, activation='linear')(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')

        print("✅ Modèle LNN construit avec keras-ncp/LTCCell")
        return model

    except (ImportError, Exception) as e:
        print(f"⚠️ keras-ncp indisponible: {e}")
        print("➡️ Fallback: LSTM classique")

        # Fallback LSTM
        inputs = keras.Input(shape=(1, state_dim))
        x = layers.LSTM(units, return_sequences=False)(inputs)
        outputs = layers.Dense(n_actions, activation='linear')(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')

        return model


class LNNAgent:
    def __init__(self, state_dim, n_actions, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-Network principal et cible
        self.q_network = build_ltc_policy(state_dim, n_actions)
        self.target_network = build_ltc_policy(state_dim, n_actions)
        self.update_target_network()

        self.replay_buffer = ReplayBuffer(max_size=10000)

    def update_target_network(self):
        """Copie les poids du Q-network vers le target network"""
        self.target_network.set_weights(self.q_network.get_weights())

    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Reshape pour format RNN: (1, 1, state_dim)
        state_input = state.reshape(1, 1, -1)
        q_values = self.q_network.predict(state_input, verbose=0)
        return np.argmax(q_values[0])

    def train_step(self, batch_size=64):
        """Entraînement sur un batch du replay buffer (Double DQN optionnel)"""
        if self.replay_buffer.size() < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Reshape pour format RNN
        states_input = states.reshape(batch_size, 1, -1)
        next_states_input = next_states.reshape(batch_size, 1, -1)

        # Double DQN: sélection d'action avec q_network, évaluation avec target_network
        current_q = self.q_network.predict(states_input, verbose=0)
        next_q_main = self.q_network.predict(next_states_input, verbose=0)
        next_q_target = self.target_network.predict(next_states_input, verbose=0)

        target_q = current_q.copy()

        for i in range(batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                # Double DQN: meilleure action selon q_network, valeur selon target_network
                best_action = np.argmax(next_q_main[i])
                target_q[i, actions[i]] = rewards[i] + self.gamma * next_q_target[i, best_action]

        # Entraînement
        loss = self.q_network.train_on_batch(states_input, target_q)

        return loss

    def decay_epsilon(self):
        """Diminue epsilon pour réduire l'exploration"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="lnn_agent.keras"):
        """Sauvegarde le modèle et les poids"""
        # Sauvegarde des poids uniquement (compatible avec LTC)
        weights_path = path.replace('.keras', '_weights.h5')
        self.q_network.save_weights(weights_path)

        # Sauvegarde des hyperparamètres
        import json
        config_path = path.replace('.keras', '_config.json')
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

        print(f"💾 Modèle sauvegardé: {weights_path} + {config_path}")

    def load(self, path="lnn_agent.keras"):
        """Charge un modèle pré-entraîné"""
        import json

        weights_path = path.replace('.keras', '_weights.h5')
        config_path = path.replace('.keras', '_config.json')

        # Vérifier si les nouveaux fichiers existent
        if os.path.exists(weights_path) and os.path.exists(config_path):
            # Charge la config
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Applique les hyperparamètres
            self.state_dim = config['state_dim']
            self.n_actions = config['n_actions']
            self.gamma = config['gamma']
            self.epsilon = config['epsilon']
            self.epsilon_min = config['epsilon_min']
            self.epsilon_decay = config['epsilon_decay']

            # Reconstruit le modèle et charge les poids
            self.q_network = build_ltc_policy(self.state_dim, self.n_actions)
            self.q_network.load_weights(weights_path)
            self.update_target_network()

            print(f"📂 Modèle chargé: {weights_path}")

        # Fallback: essayer de charger l'ancien format .keras (peut échouer avec LTC)
        elif os.path.exists(path):
            print(f"⚠️ Ancien format détecté ({path})")
            print("⚠️ Les modèles LTC ne peuvent pas être rechargés au format .keras")
            print("💡 Solution: Réentraîner avec --episodes 50 pour créer les nouveaux fichiers")
            raise FileNotFoundError(f"Utilisez le nouveau format de sauvegarde")

        else:
            print(f"❌ Fichiers introuvables: {weights_path} ou {config_path}")
            print(f"💡 Entraînez d'abord un modèle avec: python main.py --mode train")
            import numpy as np

def train_agent(env_name="highway-fast-v0", episodes=200, max_steps=100,
                save_path="lnn_agent.keras"):
    """
    Entraîne l'agent LNN sur Highway-Env
    """
    # Création de l'environnement
    env = gym.make(env_name, config={
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "normalize": True
        },
        "duration": 40,
    })

    # Dimensions
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    n_actions = env.action_space.n

    print(f"État: {state_dim} dimensions")
    print(f"Actions: {n_actions} possibles\n")

    # Création de l'agent
    agent = LNNAgent(state_dim, n_actions)

    # Métriques
    episode_rewards = []
    losses = []
    best_reward = -float('inf')

    # Nouvelles métriques détaillées
    collisions_history = []
    speeds_history = []
    lane_changes_history = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = state.flatten()
        episode_reward = 0
        episode_loss = []

        # Métriques de l'épisode
        episode_speeds = []
        previous_lane = None
        lane_change_count = 0
        crashed = False

        for step in range(max_steps):
            # Sélection et exécution de l'action
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = next_state.flatten()
            done = terminated or truncated

            # Tracking des métriques
            if 'speed' in info:
                episode_speeds.append(info['speed'])

            # Détection collision
            if 'crashed' in info and info['crashed']:
                crashed = True

            # Détection changement de voie (approximatif via position y)
            # Note: next_state[2] = y du véhicule ego (index 2 dans features)
            current_lane = int(next_state[2] * 3)  # Approximation 3 voies
            if previous_lane is not None and current_lane != previous_lane:
                lane_change_count += 1
            previous_lane = current_lane

            # Stockage dans le replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, done)

            # Entraînement (Double DQN)
            if agent.replay_buffer.size() > 64:
                loss = agent.train_step(batch_size=64)
                episode_loss.append(loss)

            episode_reward += reward
            state = next_state

            if done:
                break

        # Sauvegarde des métriques de l'épisode
        collisions_history.append(crashed)
        speeds_history.append(np.mean(episode_speeds) if episode_speeds else 0)
        lane_changes_history.append(lane_change_count)

        # Mise à jour du target network
        if episode % 10 == 0:
            agent.update_target_network()

        # Décroissance d'epsilon
        agent.decay_epsilon()

        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)

        # Sauvegarde du meilleur modèle
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(save_path)

        # Affichage périodique
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_speed = np.mean(speeds_history[-10:])
            collision_rate = np.sum(collisions_history[-10:]) / 10 * 100
            avg_lane_changes = np.mean(lane_changes_history[-10:])

            print(f"Épisode {episode}/{episodes} | Récompense: {episode_reward:.2f} | "
                  f"Moy(10): {avg_reward:.2f} | Best: {best_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | Loss: {avg_loss:.4f}")
            print(f"  └─ Vitesse moy: {avg_speed:.2f} | Collisions: {collision_rate:.0f}% | "
                  f"Changements voie: {avg_lane_changes:.1f}")

    env.close()

    # Visualisation
    plot_training_results(episode_rewards, losses, collisions_history,
                          speeds_history, lane_changes_history)

    return agent, episode_rewards


def plot_training_results(rewards, losses, collisions=None, speeds=None, lane_changes=None):
    """Affiche les courbes d'apprentissage avec métriques détaillées"""
    # Nombre de subplots selon les métriques disponibles
    n_plots = 2
    if collisions is not None:
        n_plots = 5

    fig = plt.figure(figsize=(16, 10))

    # 1. Récompenses
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(rewards, alpha=0.6, label='Récompense par épisode', color='steelblue')
    if len(rewards) > 10:
        smoothed = np.convolve(rewards, np.ones(10) / 10, mode='valid')
        ax1.plot(range(9, len(rewards)), smoothed, 'r-', linewidth=2,
                 label='Moyenne mobile (10)')
    ax1.axhline(y=np.mean(rewards), color='green', linestyle='--',
                alpha=0.5, label=f'Moyenne globale: {np.mean(rewards):.2f}')
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Récompense totale')
    ax1.set_title('Progression de l\'apprentissage')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Pertes
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(losses, alpha=0.6, color='orange')
    if len(losses) > 10:
        smoothed_loss = np.convolve(losses, np.ones(10) / 10, mode='valid')
        ax2.plot(range(9, len(losses)), smoothed_loss, 'darkred', linewidth=2)
    ax2.set_xlabel('Épisode')
    ax2.set_ylabel('Loss (MSE)')
    ax2.set_title('Évolution de la perte')
    ax2.grid(True, alpha=0.3)

    if collisions is not None:
        # 3. Taux de collision
        ax3 = plt.subplot(2, 3, 3)
        collision_rate = []
        window = 20
        for i in range(len(collisions)):
            start = max(0, i - window)
            rate = np.sum(collisions[start:i + 1]) / (i - start + 1) * 100
            collision_rate.append(rate)
        ax3.plot(collision_rate, color='red', linewidth=2)
        ax3.set_xlabel('Épisode')
        ax3.set_ylabel('Taux de collision (%)')
        ax3.set_title('Taux de collision (fenêtre 20 épisodes)')
        ax3.grid(True, alpha=0.3)

        # 4. Vitesse moyenne
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(speeds, alpha=0.6, color='blue')
        if len(speeds) > 10:
            smoothed_speeds = np.convolve(speeds, np.ones(10) / 10, mode='valid')
            ax4.plot(range(9, len(speeds)), smoothed_speeds, 'darkblue', linewidth=2)
        ax4.set_xlabel('Épisode')
        ax4.set_ylabel('Vitesse moyenne')
        ax4.set_title('Évolution de la vitesse')
        ax4.grid(True, alpha=0.3)

        # 5. Changements de voie
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(lane_changes, alpha=0.6, color='purple')
        if len(lane_changes) > 10:
            smoothed_lc = np.convolve(lane_changes, np.ones(10) / 10, mode='valid')
            ax5.plot(range(9, len(lane_changes)), smoothed_lc, 'darkviolet', linewidth=2)
        ax5.set_xlabel('Épisode')
        ax5.set_ylabel('Nombre de changements')
        ax5.set_title('Changements de voie par épisode')
        ax5.grid(True, alpha=0.3)

        # 6. Résumé statistique
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        stats_text = f"""
        📊 STATISTIQUES FINALES

        Récompense moyenne: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}
        Récompense max: {np.max(rewards):.2f}

        Taux de collision: {np.sum(collisions) / len(collisions) * 100:.1f}%
        Vitesse moyenne: {np.mean(speeds):.2f}

        Changements de voie moy: {np.mean(lane_changes):.1f}

        Total épisodes: {len(rewards)}
        """
        ax6.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                 verticalalignment='center')

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("\n📊 Graphiques sauvegardés: training_results.png")
    plt.show()


def test_agent(agent, env_name="highway-fast-v0", episodes=5, render=False):
    """
    Teste l'agent entraîné
    """
    env = gym.make(env_name,
                   render_mode="human" if render else None,
                   config={
                       "observation": {
                           "type": "Kinematics",
                           "vehicles_count": 5,
                           "features": ["presence", "x", "y", "vx", "vy"],
                           "normalize": True
                       },
                       "duration": 40,
                   })

    test_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = state.flatten()
        episode_reward = 0

        done = False
        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.flatten()
            done = terminated or truncated

            episode_reward += reward
            state = next_state

        test_rewards.append(episode_reward)
        print(f"Test épisode {episode + 1}: Récompense = {episode_reward:.2f}")

    env.close()
    avg_test = np.mean(test_rewards)
    std_test = np.std(test_rewards)
    print(f"\n📈 Récompense moyenne: {avg_test:.2f} ± {std_test:.2f}")

    return test_rewards


def demo_agent(model_path="lnn_agent.keras", env_name="highway-fast-v0"):
    """
    Démo visuelle de l'agent entraîné
    """
    # Chargement de l'agent
    env = gym.make(env_name, config={
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "normalize": True
        },
        "duration": 40,
    })

    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    n_actions = env.action_space.n

    agent = LNNAgent(state_dim, n_actions)
    agent.load(model_path)

    env.close()

    # Test avec rendu visuel
    print("\n🎮 Lancement de la démo visuelle...")
    test_agent(agent, env_name=env_name, episodes=3, render=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='LNN Agent pour Highway-Env')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'demo'],
                        help='Mode: train, test ou demo')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Nombre d\'épisodes d\'entraînement')
    parser.add_argument('--model', type=str, default='lnn_agent.keras',
                        help='Chemin du modèle')

    args = parser.parse_args()

    if args.mode == 'train':
        print("=== Entraînement de l'agent LNN sur Highway-Env ===\n")
        agent, rewards = train_agent(
            env_name="highway-fast-v0",
            episodes=args.episodes,
            max_steps=100,
            save_path=args.model
        )

        print("\n=== Test de l'agent entraîné ===\n")
        test_agent(agent, episodes=5, render=False)

    elif args.mode == 'test':
        print("=== Test de l'agent ===\n")
        env = gym.make("highway-fast-v0", config={
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy"],
                "normalize": True
            },
            "duration": 40,
        })
        state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
        n_actions = env.action_space.n
        env.close()

        agent = LNNAgent(state_dim, n_actions)
        agent.load(args.model)
        test_agent(agent, episodes=10, render=False)

    elif args.mode == 'demo':
        demo_agent(model_path=args.model)

    print("\n✅ Terminé!")