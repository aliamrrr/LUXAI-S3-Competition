import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
from luxai_s3.wrappers import LuxAIS3GymEnv
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper function: returns a movement direction (0: stay, 1: up, 2: right, 3: down, 4: left)
def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1

# ---------------------------------------
# QMIX Network Components (Unchanged)
# ---------------------------------------

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class MixingNetwork(nn.Module):
    def __init__(self, hidden_size, n_agents, state_dim):
        super(MixingNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.hyper_w_1 = nn.Linear(state_dim, hidden_size * n_agents)
        self.hyper_w_final = nn.Linear(state_dim, hidden_size)

    def forward(self, q_values, states):
        batch_size = q_values.size(0)
        w1 = torch.abs(self.hyper_w_1(states)).view(batch_size, self.hidden_size, self.n_agents)
        b1 = torch.zeros(batch_size, self.hidden_size).to(states.device)
        hidden = torch.bmm(w1, q_values.unsqueeze(2)).squeeze(2) + b1
        hidden = torch.relu(hidden)
        w_final = torch.abs(self.hyper_w_final(states)).view(batch_size, 1, self.hidden_size)
        b_final = torch.zeros(batch_size, 1).to(states.device)
        q_tot = torch.bmm(w_final, hidden.unsqueeze(2)).squeeze(2) + b_final
        return q_tot

# ---------------------------------------
# QMIX Agent Class (Updated)
# ---------------------------------------

class Agent:
    def __init__(self, player: str, env_cfg, training=False):
        self.player = player
        self.env_cfg = env_cfg
        self.training = training
        self.team_id = 0 if player == "player_0" else 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define sizes
        self.state_size = 27  # Matches the state vector size from get_state
        self.action_size = 6  # 0: stay, 1-4: move directions, 5: sap
        self.hidden_size = 256
        self.n_agents = self.env_cfg["max_units"]

        # Initialize networks
        self.q_network = QNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_q_network = QNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.mixing_network = MixingNetwork(self.hidden_size, self.n_agents, self.state_size * self.n_agents).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(list(self.q_network.parameters()) + list(self.mixing_network.parameters()), lr=0.0005)

        # Load model if not training
        if not training:
            self.load_model()

        # Training variables
        self.buffer = []
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.update_target_every = 1000
        self.step_counter = 0

        # Visitation map for exploration bonus
        self.visit_counts = np.zeros((env_cfg["map_width"], env_cfg["map_height"]))

    def get_state(self, unit_id, obs, step):
        unit_pos = obs["units"]["position"][self.team_id][unit_id]
        unit_energy = obs["units"]["energy"][self.team_id][unit_id]

        # Relic nodes (point tiles)
        relic_nodes = np.array(obs["relic_nodes"])
        relic_mask = np.array(obs["relic_nodes_mask"])
        if not relic_mask.any():
            closest_relic = np.array([-1, -1])
        else:
            visible_relics = relic_nodes[relic_mask]
            distances = np.linalg.norm(visible_relics - unit_pos, axis=1)
            closest_relic = visible_relics[np.argmin(distances)]

        # Friendly units (up to 5)
        friendly_positions = obs["units"]["position"][self.team_id]
        friendly_mask = obs["units_mask"][self.team_id]
        friendly_pos_list = [pos for i, pos in enumerate(friendly_positions) if friendly_mask[i] and i != unit_id]
        friendly_pos_list = friendly_pos_list[:5] + [np.array([-1, -1])] * (5 - len(friendly_pos_list))

        # Enemy units (up to 5)
        opp_team_id = 1 - self.team_id
        enemy_positions = obs["units"]["position"][opp_team_id]
        enemy_mask = obs["units_mask"][opp_team_id]
        enemy_pos_list = [pos for i, pos in enumerate(enemy_positions) if enemy_mask[i]]
        enemy_pos_list = enemy_pos_list[:5] + [np.array([-1, -1])] * (5 - len(enemy_pos_list))

        # Flatten positions
        friendly_flat = np.concatenate(friendly_pos_list)
        enemy_flat = np.concatenate(enemy_pos_list)

        # On point tile flag
        on_point_tile = int(any(np.array_equal(unit_pos, rn) for rn in relic_nodes[relic_mask]))

        # Construct state vector
        state = np.concatenate([
            unit_pos, closest_relic, [unit_energy], [step / 505.0],
            friendly_flat, enemy_flat, [on_point_tile]
        ])
        return torch.FloatTensor(state).to(self.device)

    def get_valid_actions(self, unit_pos, unit_energy, obs):
        valid_mask = [True] * 6  # 0: stay, 1-4: move, 5: sap
        directions = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]  # Corresponding to actions 0-4
        tile_type = obs["map_features"]["tile_type"]
        map_width, map_height = tile_type.shape

        # Validate movement actions
        for i in range(1, 5):
            next_pos = [unit_pos[0] + directions[i][0], unit_pos[1] + directions[i][1]]
            if not (0 <= next_pos[0] < map_width and 0 <= next_pos[1] < map_height):
                valid_mask[i] = False
            elif tile_type[next_pos[0], next_pos[1]] == 2:  # Obstacle
                valid_mask[i] = False
            elif unit_energy < self.env_cfg["unit_move_cost"]:
                valid_mask[i] = False

        # Validate sap action
        if unit_energy < self.env_cfg["unit_sap_cost"] * 2:  # Prevent sap if energy too low
            valid_mask[5] = False

        return valid_mask

    def act(self, step, obs, remainingOverageTime=60):
        unit_mask = np.array(obs["units_mask"][self.team_id])
        available_unit_ids = np.where(unit_mask)[0]
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        for unit_id in available_unit_ids:
            state = self.get_state(unit_id, obs, step)
            unit_pos = obs["units"]["position"][self.team_id][unit_id]
            with torch.no_grad():
                q_values = self.q_network(state).cpu().numpy()
            valid_mask = self.get_valid_actions(unit_pos, obs["units"]["energy"][self.team_id][unit_id], obs)

            # Check if unit is on a point tile
            on_point_tile = state[-1].item() == 1

            # Shape Q-values for desired behaviors
            if on_point_tile:
                q_values[0] += 10.0  # Large positive bias for staying on point tiles

            # Penalize moving to tiles occupied by friendly units
            friendly_positions = [pos for i, pos in enumerate(obs["units"]["position"][self.team_id]) if obs["units_mask"][self.team_id][i] and i != unit_id]
            directions = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]
            for i in range(1, 5):  # Movement actions
                next_pos = [unit_pos[0] + directions[i][0], unit_pos[1] + directions[i][1]]
                if any(np.array_equal(next_pos, pos) for pos in friendly_positions):
                    q_values[i] -= 5.0  # Penalty for overlapping with friendly units

            # Boltzmann exploration
            temperature = max(0.1, self.epsilon)
            q_values_valid = q_values.copy()
            q_values_valid[~np.array(valid_mask)] = -float('inf')
            probs = torch.softmax(torch.tensor(q_values_valid) / temperature, dim=0).numpy()
            action = np.random.choice(len(valid_mask), p=probs / probs.sum())  # Normalize probabilities

            if action == 5:  # Sap action
                opp_team_id = 1 - self.team_id
                opp_positions = np.array(obs["units"]["position"][opp_team_id])
                opp_energies = np.array(obs["units"]["energy"][opp_team_id])
                opp_mask = np.array(obs["units_mask"][opp_team_id])
                valid_targets = [(pos, energy) for i, (pos, energy) in enumerate(zip(opp_positions, opp_energies)) if opp_mask[i] and pos[0] != -1]
                if valid_targets:
                    # Score targets: prioritize enemies on point tiles (+10 bonus) or with high energy
                    relic_nodes = np.array(obs["relic_nodes"])
                    relic_mask = np.array(obs["relic_nodes_mask"])
                    point_tiles = [tuple(rn) for rn in relic_nodes[relic_mask]]
                    scores = [energy + 10 if tuple(pos) in point_tiles else energy for pos, energy in valid_targets]
                    target_pos = valid_targets[np.argmax(scores)][0]
                    actions[unit_id] = [5, target_pos[0], target_pos[1]]
                else:
                    actions[unit_id] = [0, 0, 0]  # Default to stay if no targets
            else:  # Movement or stay action
                actions[unit_id] = [action, 0, 0]

            # Update visitation counts during training for exploration tracking
            if self.training:
                pos = unit_pos.astype(int)
                self.visit_counts[pos[0], pos[1]] += 1

        if self.training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return actions

    def learn(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack([torch.stack(s) for s in states]).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(-1)
        next_states = torch.stack([torch.stack(s) for s in next_states]).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device).unsqueeze(-1)

        # Use the true team reward without modifications
        global_state = states.view(states.size(0), -1)
        next_global_state = next_states.view(next_states.size(0), -1)

        q_values = self.q_network(states)
        q_values_selected = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixing_network(q_values_selected, global_state)

        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(dim=2)[0]
            next_q_tot = self.mixing_network(next_q_values, next_global_state)
            targets = rewards + (1 - dones) * 0.99 * next_q_tot

        loss = nn.MSELoss()(q_tot, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_counter += 1
        if self.step_counter % self.update_target_every == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            logging.info(f"Target network updated at step {self.step_counter}")

    def save_model(self):
        torch.save({
            'q_network_state': self.q_network.state_dict(),
            'mixing_network_state': self.mixing_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, f'qmix_model_{self.player}.pth')

    def load_model(self):
        try:
            # model_path = f"workspace/qmix_model_{self.player}.pth"
            model_file_name = f"qmix_model_{self.player}.pth"
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_file_name)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.q_network.load_state_dict(checkpoint['q_network_state'])
            self.mixing_network.load_state_dict(checkpoint['mixing_network_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            logging.info(f"Loaded model for {self.player} from {model_path}")
        except FileNotFoundError:
            logging.warning(f"No trained model found for {self.player} at {model_path}. Starting from scratch.")

# ---------------------------------------
# Training and Evaluation Functions (Unchanged)
# ---------------------------------------

def train(player_0, player_1, num_games=100, save_interval=10):
    env = LuxAIS3GymEnv(numpy_output=True)
    obs, info = env.reset(seed=42)
    env_cfg = info["params"]

    agents = {"player_0": player_0, "player_1": player_1}

    logging.info("Starting training...")
    print("Training started...")

    for game in range(num_games):
        obs, info = env.reset()
        game_done = False
        step = 0
        total_reward_0 = 0
        total_reward_1 = 0

        while not game_done:
            actions = {}
            for player in ["player_0", "player_1"]:
                actions[player] = agents[player].act(step, obs[player])

            next_obs, rewards, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] or truncated[k] for k in terminated}

            total_reward_0 += rewards["player_0"]
            total_reward_1 += rewards["player_1"]

            for player in ["player_0", "player_1"]:
                if agents[player].training:
                    agent = agents[player]
                    states_all = [agent.get_state(unit_id, obs[player], step) for unit_id in range(agent.n_agents)]
                    next_states_all = [agent.get_state(unit_id, next_obs[player], step + 1) for unit_id in range(agent.n_agents)]
                    actions_all = [actions[player][unit_id][0] for unit_id in range(agent.n_agents)]
                    reward = rewards[player].item()
                    done = dones[player].item()
                    agent.buffer.append((states_all, actions_all, reward, next_states_all, done))

                    if len(agent.buffer) >= 128:
                        batch = random.sample(agent.buffer, 128)
                        agent.learn(batch)

            if any(dones.values()):
                game_done = True
            step += 1
            obs = next_obs

        print(f"Game {game + 1}/{num_games} finished. Total rewards: player_0={total_reward_0}, player_1={total_reward_1}")
        logging.info(f"Game {game + 1}/{num_games} finished with rewards: player_0={total_reward_0}, player_1={total_reward_1}")

        if (game + 1) % save_interval == 0:
            for player in ["player_0", "player_1"]:
                agents[player].save_model()
            logging.info(f"Models saved after {game + 1} games.")
            print(f"Models saved after {game + 1} games.")

    env.close()
    logging.info("Training finished.")
    print("Training finished.")

def evaluate(player_0, player_1, num_games=10):
    env = LuxAIS3GymEnv(numpy_output=True)
    obs, info = env.reset(seed=42)
    env_cfg = info["params"]

    agents = {"player_0": player_0, "player_1": player_1}
    total_rewards = {"player_0": 0, "player_1": 0}

    logging.info("Starting evaluation...")
    print("Evaluation started...")

    for game in range(num_games):
        obs, info = env.reset()
        game_done = False
        game_reward_0 = 0
        game_reward_1 = 0

        while not game_done:
            actions = {}
            for player in ["player_0", "player_1"]:
                actions[player] = agents[player].act(0, obs[player])
            obs, rewards, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] or truncated[k] for k in terminated}

            game_reward_0 += rewards["player_0"]
            game_reward_1 += rewards["player_1"]

            if any(dones.values()):
                game_done = True

        total_rewards["player_0"] += game_reward_0
        total_rewards["player_1"] += game_reward_1

        print(f"Evaluation Game {game + 1}/{num_games} finished. Rewards: player_0={game_reward_0}, player_1={game_reward_1}")
        logging.info(f"Game {game + 1}/{num_games} finished with rewards: player_0={game_reward_0}, player_1={game_reward_1}")

    print(f"Evaluation finished. Total rewards: player_0={total_rewards['player_0']}, player_1={total_rewards['player_1']}")
    logging.info(f"Total evaluation rewards: {total_rewards}")

    env.close()