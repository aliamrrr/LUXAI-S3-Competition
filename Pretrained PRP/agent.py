import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
from luxai_s3.wrappers import LuxAIS3GymEnv

# Dueling DQN with Layer Normalization
class DuelingDQN(nn.Module):
    """ The idea behind dueling deep q networks is instead of learning the Q value directly we seperate it into two different components 

        V(s) : the value of the state s ( how good is a state )
        A(s,a) : the advantage of taking an action a in the same state s 

        This trick is beneficial for training the neural network. For example, imagine our agent navigating in space :
        - The value function V(s) allows the network to check whether a particular area of the map is good or not (open, safe, or having rewards), independently of the actions.
        - The advantage function A(s,a) manages the relative benefits of the possible actions in that zone, such as going left, right, or forward.
        
        By decoupling the estimation of state value from the differences between actions, the network can learn more efficiently. Instead, of relearning
        both the value of a state and the advantage of taking actions each time in the case of traditional deep q network, the DDQN is better at generalization
        by learning a baseline V(s) for all actions and then computing the nuances of taking these actions. 
    
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(DuelingDQN, self).__init__()

        # We start by a linear layer to extract features from our input and then normalize it for stability
        self.feature = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        # The advantage network is applyed to figure out the advantage of taking each action in a state s
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

        # The state value network is applyed to seperately judge the value of being in this state ( Is this a good state to be at ? That last state i was at seems better )
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # Extract the features from the input 
        x = F.relu(self.ln1(self.feature(x)))
        # Compute the advantage of taking each action in the state 
        adv = self.advantage(x)
        # Compute the value of being at the state in the first place
        val = self.value(x)
        # Combine both measurements. We remove the mean of the advantages to solve the problem of "identifiability". Basically,
        # we guide the network to learn unique values of the advantages and the values. If we didn't remove the mean, the equation would
        # have multiple solutions and we will confuse the network during training.
        return val + adv - adv.mean()

# We use a Prioritised replay buffer to further improve our learning 
class PrioritizedReplayBuffer:
    """
    To train our agent, we need to use a buffer where we store past experiences and use them to train our agent in batches.  
    As the agent progresses in the map, we collect these experiences in the buffer and then sample a batch from them for training.  

    The main idea behind the prioritized replay buffer is to sample "important" experiences primarily instead of less significant ones.  
    Important experiences are those where the model made significant errors in estimating the Q-values, as these are the ones that  
    provide the most learning signal.  

    To do this, we assign each experience a sampling probability based on the model’s error (also called TD error).  
    The higher the error, the more likely we are to sample that experience. These probabilities are updated  
    after learning with the new data.  

    However, since our sampling distribution is no longer uniform, we need a way to balance it.  
    To make sure we still take into account other experiences, we use "importance sampling weights" to correct this bias during training.  
    The idea is that when we sample data and use it in backpropagation, we apply these weights to reduce the effect of frequently sampled  
    experiences and increase the effect of less important ones.  

    This technique helps mitigate the biased sampling so that, although we prioritize certain experiences,  
    we do not ignore other meaningful replays, preventing overfitting and ensuring more stable learning.
    """

    def __init__(self, capacity, alpha=0.6):
        """Initialize two buffers to stock the experiences and their priorities"""
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.epsilon = 1e-5  # We use epsilon to avoid experiences with zero priority ( all experiences have a chance to be sampled)

    def push(self, state, action, reward, next_state, done):
        """Push the experiences in the buffer and assign them default priorities of 1"""
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)


    def sample(self, batch_size, beta=0.4):
        """Sample experiences based on probabilities computed from the priorities, 
           along with their indices and importance sampling weights
        """
        scaled_priorities = np.array(self.priorities) ** self.alpha
        probs = scaled_priorities / scaled_priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples,indices, torch.FloatTensor(weights).to("cpu")

    def update_priorities(self, indices, errors):
        """Update the priorities with the new found estimation errors"""
        for i, err in zip(indices, errors.detach().cpu().numpy()):
            self.priorities[i] = abs(err) + self.epsilon

    def __len__(self):
        """Return the length of the experiences buffer"""
        return len(self.buffer)

# Now we define our agent that is going to use the above techniques
class Agent:
    def __init__(self, player, env_cfg, training=False):
        """Initialize the agent, particularly with these main parameters : 
         
         - policy network : The main network that defines the policy of the agent and dictate its movements. 
         - target network : A reference network used to train the network by providing fixed values for the q values. It 
                            is essentially a copy of the policy network but updated less frequently. We use it to have stable targets
                            during training and avoid changing these tragets when we update the model.
         - memory         : The prioritized replay buffer used to store and sample experiences.
         
         """
        self.player = player
        self.opp_team_id = 1 if player == "player_0" else 0
        self.team_id = 0 if player == "player_0" else 1

        self.env_cfg = env_cfg
        self.training = training

        self.state_size = 8  
        self.action_size = 6  # Actions: stay, up, right, down, left, sap
        self.hidden_size = 256
        self.gamma = 0.99
        self.lr = 0.0001
        self.batch_size = 64
        self.target_update_freq = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DuelingDQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net = DuelingDQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = PrioritizedReplayBuffer(20000)

        self.epsilon = 1.0 if training else 0.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        if not training:
            self.load_model()

    def _state_representation(self, unit_pos, unit_energy, relic_nodes, step, relic_mask, obs):
        """A tensor that we use to represent the state of the game, we feed it to the network to generate actions"""
        # Closest relic
        if not relic_mask.any():
            closest_relic = np.array([-1, -1])
        else:
            visible_relics = relic_nodes[relic_mask]
            distances = np.linalg.norm(visible_relics - unit_pos, axis=1)
            closest_relic = visible_relics[np.argmin(distances)]

        # Closest enemy unit (if visible)
        opp_positions = obs["units"]["position"][self.opp_team_id]
        opp_mask = obs["units_mask"][self.opp_team_id]
        enemy_dist = 999
        for opp_id, pos in enumerate(opp_positions):
            if opp_mask[opp_id] and pos[0] != -1:
                dist = np.linalg.norm(pos - unit_pos)
                enemy_dist = min(enemy_dist, dist)

        # State vector
        state = np.concatenate([
            unit_pos,              # Unit position (2)
            closest_relic,         # Closest relic position (2)
            [unit_energy],         # Unit energy (1)
            [enemy_dist],          # Distance to nearest enemy (1)
            [step / 505.0],        # Normalized step (1)
            [np.sum(relic_mask)]   # Number of visible relics (1)
        ])
        return torch.FloatTensor(state).to(self.device)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """The function that we call to get the action of the agent. It creates a state representation based on the passed observations 
           parameter.

            It uses an epsilon greedy approach to take actions which works as follows : 
            It draw a random number between 0 and 1, if it is less than epsilon, it returns a random action. If it is higher
            it uses the policy network to determine the action and returns it. 

        
        """
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energys = np.array(obs["units"]["energy"][self.team_id])
        relic_nodes = np.array(obs["relic_nodes"])
        relic_mask = np.array(obs["relic_nodes_mask"])
        self.score = np.array(obs["team_points"][self.team_id])

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        available_units = np.where(unit_mask)[0]

        for unit_id in available_units:
            state = self._state_representation(
                unit_positions[unit_id], unit_energys[unit_id], relic_nodes, step, relic_mask, obs
            ).unsqueeze(0).to(self.device)

            if np.random.rand() < self.epsilon:
                actions[unit_id] = [random.randint(0, self.action_size-1), 0, 0]  # Random action
            else:
                with torch.no_grad():
                    q_values = self.policy_net(state)
                    action_type = q_values.argmax().item()
                    actions[unit_id] = [action_type, 0, 0]

        return actions

    def learn(self):
        """This is the function that we call to make the model learn from a sampled batch from the saved experiences 
            It uses a prioritized replay buffer for back propagation, favoring therfore important experiences and using the 
            importance sampling weights to reduce the sampling bias as discussed earlier.
        """
        if len(self.memory) < self.batch_size:
            return None

        transitions, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze()
            targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = (weights * F.mse_loss(q_values, targets, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        errors = torch.abs(q_values - targets).detach()
        self.memory.update_priorities(indices, errors)

        if random.randint(0, self.target_update_freq) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

    def save_model(self):
        """A function we use to save our model to be used afterwards in matches"""
        torch.save(self.policy_net.state_dict(), f'dqn_{self.player}.pth')

    def load_model(self):
        """A function we use to load our saved policy model"""
        try:
            self.policy_net.load_state_dict(torch.load(f'dqn_{self.player}.pth'))
            self.target_net.load_state_dict(self.policy_net.state_dict())
        except FileNotFoundError:
            print("No saved model found.")

def evaluate_agents(agent_1_cls, agent_2_cls, seed=42, training=True, games_to_play=3):
    """
    The evaluate function is used to play matches between two opposing agents. It can either be used in a training context to 
    train the models or to simply visualize the performance of both models. 
    
    It defines also the reward function used to train the models, below is a complete description of how we made it. 
    """
    env = LuxAIS3GymEnv(numpy_output=True)
    obs, info = env.reset(seed=seed)
    env_cfg = info["params"]

    player_0 = agent_1_cls("player_0", env_cfg, training=training)
    player_1 = agent_2_cls("player_1", env_cfg, training=training)
    # Instead of having different buffers for different agents, we share it so that opposant agents 
    # can both benefit from all experiences. The shared memory is of course used only in training
    shared_memory = PrioritizedReplayBuffer(20000)
    player_0.memory = shared_memory
    player_1.memory = shared_memory
    player_0.team_id = 0
    player_1.team_id = 1
    player_0.opp_team_id = 1
    player_1.opp_team_id = 0

    loss_player_0 = []
    loss_player_1 = []

    for i in tqdm(range(games_to_play), desc="Playing games", unit="game"):
        losses_0 = []
        losses_1 = []

        obs, info = env.reset()
        game_done = False
        step = 0
        last_obs = None
        last_actions = None

        while not game_done:
            actions = {}
            if training and step > 0:
                last_obs = {
                    "player_0": obs["player_0"].copy(),
                    "player_1": obs["player_1"].copy()
                }

            actions["player_0"] = player_0.act(step=step, obs=obs["player_0"])
            actions["player_1"] = player_1.act(step=step, obs=obs["player_1"])

            prev_action = {
                "player_0": np.full(env_cfg["max_units"], -1),
                "player_1": np.full(env_cfg["max_units"], -1)
            }

            if training:
                last_actions = actions.copy()

            next_obs, _rewards, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}

            if training and last_obs is not None:
                for unit_id in range(env_cfg["max_units"]):
                    for agent in [player_0, player_1]:
                        if obs[agent.player]["units_mask"][agent.team_id][unit_id]:

                            # Now the most important part, computing the reward for the action
                            # We will apply rewards sequentially taking into account different aspects
                            reward = 0

                            # 1. New Relics Collected or Being Guarded
                            # Reward if a new relic tile has been discovered
                            new_relics = sum(next_obs[agent.player]["relic_nodes_mask"]) - sum(last_obs[agent.player]["relic_nodes_mask"])
                            reward += 2 * new_relics

                            # Also, if the unit is on an active relic node tile, give a bonus.
                            unit_pos = np.array(next_obs[agent.player]["units"]["position"][agent.team_id][unit_id])
                            relic_bonus = 0
                            # Get positions of active relics from the observation.
                            active_relics = np.array(obs[agent.player]["relic_nodes"])[np.array(obs[agent.player]["relic_nodes_mask"], dtype=bool)]
                            # Reward the ships based on the proximity from the tiles (the ship is guarding the relic)
                            if active_relics.size > 0:
                                # Use a small tolerance (e.g., 0.5) for more flexibility
                                if any(np.linalg.norm(unit_pos - relic_pos) < 0.5 for relic_pos in active_relics):
                                    relic_bonus = 1
                            reward += relic_bonus

                            # 2. Exploration Reward
                            # Encourage ships to move by rewarding the distance traveled from the last step.
                            prev_pos = np.array(last_obs[agent.player]["units"]["position"][agent.team_id][unit_id])
                            distance_moved = np.linalg.norm(unit_pos - prev_pos)
                            exploration_bonus = 0.05 * distance_moved  
                            reward += exploration_bonus

                            # 3. Cluster/Grouping Bonus and Penalty
                            # We want to favorize clustering for our ships so that they are stronger facing the opponent
                            # Get positions of all active allied units.
                            team_positions = np.array(obs[agent.player]["units"]["position"][agent.team_id])
                            team_mask = np.array(obs[agent.player]["units_mask"][agent.team_id], dtype=bool)
                            active_positions = team_positions[team_mask]
                            # Count how many allies are within a certain radius of this ship.
                            cluster_radius = 2.0 # In our case we picked a distance of 2 to have a compact cluster
                            if active_positions.size > 0:
                                distances = np.linalg.norm(active_positions - unit_pos, axis=1)
                                cluster_count = np.sum(distances < cluster_radius)
                            else:
                                cluster_count = 0

                            # Verify thresholds and apply rewards :
                            if cluster_count >= 5 and cluster_count <= 7  : 
                                cluster_reward = 0.5
                            else:
                                cluster_reward = -0.3  # deduct points if not enough allies are nearby
                            reward += cluster_reward

                            # 4. Reward if new ships have joined the cluster compared to last time:
                            prev_team_positions = np.array(last_obs[agent.player]["units"]["position"][agent.team_id])
                            prev_team_mask = np.array(last_obs[agent.player]["units_mask"][agent.team_id], dtype=bool)
                            prev_active_positions = prev_team_positions[prev_team_mask]
                            if prev_active_positions.size > 0:
                                prev_distances = np.linalg.norm(prev_active_positions - prev_pos, axis=1)
                                prev_cluster_count = np.sum(prev_distances < cluster_radius)
                            else:
                                prev_cluster_count = 0

                            if cluster_count > prev_cluster_count:
                                reward += 0.5  # bonus for attracting new allies

                            # 5. Other Domain-Specific Rewards ===
                            # For example, if a ship executes a successful sap action against an enemy, add a bonus.
                            if last_actions[agent.player][unit_id][0] == 5 and any(next_obs[agent.player]["units_mask"][agent.opp_team_id]):
                                reward += 0.5

                            
                            # Also, to avoid situations where the model learns that moving only in one dimension
                            # is the best strategy, add a penalty to encourage diverse movements
                            if last_actions[agent.player][unit_id][0] == prev_action[agent.player][unit_id] : 
                                reward -= 0.5 
                            prev_action[agent.player][unit_id] = last_actions[agent.player][unit_id][0]   


                            current_state = agent._state_representation(
                                last_obs[agent.player]["units"]["position"][agent.team_id][unit_id],
                                last_obs[agent.player]["units"]["energy"][agent.team_id][unit_id],
                                last_obs[agent.player]["relic_nodes"],
                                step,
                                last_obs[agent.player]["relic_nodes_mask"],
                                last_obs[agent.player]
                            )
                            next_state = agent._state_representation(
                                next_obs[agent.player]["units"]["position"][agent.team_id][unit_id],
                                next_obs[agent.player]["units"]["energy"][agent.team_id][unit_id],
                                next_obs[agent.player]["relic_nodes"],
                                step + 1,
                                next_obs[agent.player]["relic_nodes_mask"],
                                next_obs[agent.player]
                            )

                            agent.memory.push(
                                current_state,
                                last_actions[agent.player][unit_id][0],
                                reward,
                                next_state,
                                dones[agent.player]
                            )

                loss_0 = player_0.learn()
                loss_1 = player_1.learn()
                if loss_0 is not None:
                    losses_0.append(loss_0)
                if loss_1 is not None:
                    losses_1.append(loss_1)

            obs = next_obs
            if dones["player_0"] or dones["player_1"]:
                game_done = True
                if training:
                    player_0.save_model()
                    player_1.save_model()

            step += 1

        loss_player_0.append(np.mean(losses_0) if losses_0 else 0)
        loss_player_1.append(np.mean(losses_1) if losses_1 else 0)

    env.close()
    if training:
        player_0.save_model()
        player_1.save_model()

    return loss_player_0, loss_player_1

if __name__ == "__main__":
    loss_0, loss_1 = evaluate_agents(Agent, Agent, seed=42, training=True, games_to_play=150)
    print(f"Player 0 Losses: {loss_0}")
    print(f"Player 1 Losses: {loss_1}")