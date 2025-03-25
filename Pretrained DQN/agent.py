import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
from luxai_s3.wrappers import LuxAIS3GymEnv
from tqdm import tqdm

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, player: str, env_cfg, training=False):
        self.player = player
        self.team_id = 0 if player == "player_0" else 1
        self.env_cfg = env_cfg
        self.training = training

        self.state_size = 27
        self.action_size = 6
        self.hidden_size = 256
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.update_target_every = 1000

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(10000)
        self.step_counter = 0

        self.visit_counts = np.zeros((env_cfg["map_width"], env_cfg["map_height"]))

        if not training:
            self.load_model()
            self.epsilon = 0.0

    def _state_representation(self, unit_id, obs, step):
        unit_pos = obs["units"]["position"][self.team_id][unit_id]
        unit_energy = obs["units"]["energy"][self.team_id][unit_id]

        relic_nodes = np.array(obs["relic_nodes"])
        relic_mask = np.array(obs["relic_nodes_mask"])
        if not relic_mask.any():
            closest_relic = np.array([-1, -1])
        else:
            visible_relics = relic_nodes[relic_mask]
            distances = np.linalg.norm(visible_relics - unit_pos, axis=1)
            closest_relic = visible_relics[np.argmin(distances)]

        friendly_positions = obs["units"]["position"][self.team_id]
        friendly_mask = obs["units_mask"][self.team_id]
        friendly_pos_list = [pos for i, pos in enumerate(friendly_positions) if friendly_mask[i] and i != unit_id]
        friendly_pos_list = friendly_pos_list[:5] + [np.array([-1, -1])] * (5 - len(friendly_pos_list))

        opp_team_id = 1 - self.team_id
        enemy_positions = obs["units"]["position"][opp_team_id]
        enemy_mask = obs["units_mask"][opp_team_id]
        enemy_pos_list = [pos for i, pos in enumerate(enemy_positions) if enemy_mask[i]]
        enemy_pos_list = enemy_pos_list[:5] + [np.array([-1, -1])] * (5 - len(enemy_pos_list))

        friendly_flat = np.concatenate(friendly_pos_list)
        enemy_flat = np.concatenate(enemy_pos_list)

        on_point_tile = int(any(np.array_equal(unit_pos, rn) for rn in relic_nodes[relic_mask]))

        state = np.concatenate([
            unit_pos, closest_relic, [unit_energy], [step / 505.0],
            friendly_flat, enemy_flat, [on_point_tile]
        ])
        return torch.FloatTensor(state).to(self.device)

    def _get_valid_actions(self, unit_pos, unit_energy, obs):
        valid_mask = [True] * 6
        directions = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]
        tile_type = obs["map_features"]["tile_type"]
        map_width, map_height = tile_type.shape

        for i in range(1, 5):
            next_pos = [unit_pos[0] + directions[i][0], unit_pos[1] + directions[i][1]]
            if not (0 <= next_pos[0] < map_width and 0 <= next_pos[1] < map_height):
                valid_mask[i] = False
            elif tile_type[next_pos[0], next_pos[1]] == 2:
                valid_mask[i] = False
            elif unit_energy < self.env_cfg["unit_move_cost"]:
                valid_mask[i] = False

        if unit_energy < self.env_cfg["unit_sap_cost"] * 2:
            valid_mask[5] = False

        return valid_mask

    def act(self, step: int, obs, remainingOverageTime=60):
        unit_mask = np.array(obs["units_mask"][self.team_id])
        available_unit_ids = np.where(unit_mask)[0]
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        for unit_id in available_unit_ids:
            state = self._state_representation(unit_id, obs, step)
            unit_pos = obs["units"]["position"][self.team_id][unit_id]
            valid_mask = self._get_valid_actions(unit_pos, obs["units"]["energy"][self.team_id][unit_id], obs)

            with torch.no_grad():
                q_values = self.policy_net(state).cpu().numpy()

            if state[-1].item() == 1:
                q_values[0] += 10.0

            if self.training and random.random() < self.epsilon:
                q_values_valid = q_values.copy()
                q_values_valid[~np.array(valid_mask)] = -float('inf')
                action = np.argmax(q_values_valid)
            else:
                q_values_valid = q_values.copy()
                q_values_valid[~np.array(valid_mask)] = -float('inf')
                temperature = max(0.1, self.epsilon)
                probs = torch.softmax(torch.tensor(q_values_valid) / temperature, dim=0).numpy()
                action = np.random.choice(len(valid_mask), p=probs / probs.sum())

            if action == 5:
                opp_team_id = 1 - self.team_id
                opp_positions = np.array(obs["units"]["position"][opp_team_id])
                opp_energies = np.array(obs["units"]["energy"][opp_team_id])
                opp_mask = np.array(obs["units_mask"][opp_team_id])
                valid_targets = [(pos, energy) for i, (pos, energy) in enumerate(zip(opp_positions, opp_energies)) if opp_mask[i] and pos[0] != -1]
                if valid_targets:
                    relic_nodes = np.array(obs["relic_nodes"])
                    relic_mask = np.array(obs["relic_nodes_mask"])
                    point_tiles = [tuple(rn) for rn in relic_nodes[relic_mask]]
                    scores = [energy + 10 if tuple(pos) in point_tiles else energy for pos, energy in valid_targets]
                    target_pos = valid_targets[np.argmax(scores)][0]
                    actions[unit_id] = [5, target_pos[0], target_pos[1]]
                else:
                    actions[unit_id] = [0, 0, 0]
            else:
                actions[unit_id] = [action, 0, 0]

            if self.training:
                pos = unit_pos.astype(int)
                self.visit_counts[pos[0], pos[1]] += 1

        if self.training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return actions

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        rewards = [float(r) if r is not None else 0.0 for r in rewards]
        dones = [float(d) if d is not None else 0.0 for d in dones]  # Ensure dones are scalars
        
        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        for i, state in enumerate(states):
            pos = state[:2].cpu().numpy().astype(int)
            exploration_bonus = 0.01 / (1 + self.visit_counts[pos[0], pos[1]])
            rewards[i] += exploration_bonus
            if state[-1].item() == 1:
                rewards[i] += 1.0

        q_values = self.policy_net(states)
        q_values_selected = q_values.gather(1, actions)
        next_q_values = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values_selected, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_counter += 1
        if self.step_counter % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save_model(self):
        torch.save(self.policy_net.state_dict(), f'dqn_model_{self.player}.pth')

    def load_model(self):
        model_file_name = f"dqn_model_{self.player}.pth"
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_file_name)
        if os.path.exists(model_path):
            self.policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        else:
            print(f"No trained model found for {self.player} at {model_path}")


def evaluate_agents(agent_cls, seed=42, training=True, games_to_play=10):
    env = LuxAIS3GymEnv(numpy_output=True)
    obs, info = env.reset(seed=seed)
    env_cfg = info["params"]

    player_0 = agent_cls("player_0", env_cfg, training=training)
    player_1 = agent_cls("player_1", env_cfg, training=training)

    rewards_0, rewards_1 = [], []
    losses_0, losses_1 = [], []

    for _ in tqdm(range(games_to_play), desc="Playing games", unit="game"):
        obs, info = env.reset(seed=random.randint(0, 1000))  # Random seed for variety
        game_done = False
        step = 0
        total_reward_0 = 0
        total_reward_1 = 0

        while not game_done:
            actions = {}
            states = {}
            next_states = {}
            for agent in [player_0, player_1]:
                actions[agent.player] = agent.act(step, obs[agent.player])
                if training:
                    # Use a list sized to max_units, fill only active units
                    states[agent.player] = [None] * env_cfg["max_units"]
                    for unit_id in range(env_cfg["max_units"]):
                        if obs[agent.player]["units_mask"][agent.team_id][unit_id]:
                            states[agent.player][unit_id] = agent._state_representation(unit_id, obs[agent.player], step)

            next_obs, rewards, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] or truncated[k] for k in terminated}

            total_reward_0 += rewards["player_0"]
            total_reward_1 += rewards["player_1"]

            if training:
                for agent in [player_0, player_1]:
                    # Same for next_states
                    next_states[agent.player] = [None] * env_cfg["max_units"]
                    for unit_id in range(env_cfg["max_units"]):
                        if next_obs[agent.player]["units_mask"][agent.team_id][unit_id]:
                            next_states[agent.player][unit_id] = agent._state_representation(unit_id, next_obs[agent.player], step + 1)

                    # Push experiences using correct indices
                    for unit_id in range(env_cfg["max_units"]):
                        if obs[agent.player]["units_mask"][agent.team_id][unit_id]:
                            action = actions[agent.player][unit_id][0]
                            state = states[agent.player][unit_id]  # Now matches unit_id
                            next_state = next_states[agent.player][unit_id] if next_states[agent.player][unit_id] is not None else state
                            reward = float(rewards[agent.player])
                            done = float(dones[agent.player])
                            agent.memory.push(state, action, reward, next_state, done)

                    loss = agent.learn()
                    if loss is not None:
                        if agent.player == "player_0":
                            losses_0.append(loss)
                        else:
                            losses_1.append(loss)

            obs = next_obs
            step += 1
            if any(dones.values()):
                game_done = True

        rewards_0.append(total_reward_0)
        rewards_1.append(total_reward_1)
        if training:
            player_0.save_model()
            player_1.save_model()
            if (_ + 1) % 10 == 0:
                print(f"Games {_+1}: Avg Reward P0={sum(rewards_0[-10:])/10}, P1={sum(rewards_1[-10:])/10}")
            if (_ + 1) % 50 == 0:
                print(f"Saved models at game {_+1}")

    env.close()
    print(f"Total rewards: player_0={sum(rewards_0)}, player_1={sum(rewards_1)}")
    return losses_0, losses_1, rewards_0, rewards_1

