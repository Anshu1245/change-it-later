import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from gymnasium import spaces
import random
from collections import deque
from scipy.stats import ttest_rel


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)


class RandMDP(gym.Env):
    def __init__(self, seed=0, option='rand'):
        super(RandMDP, self).__init__()
        assert option in ('rand', 'semi_rand', 'fixed')
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(), dtype=np.float32)
        self.time = 0
        self.rng = np.random.RandomState(seed)
        self.obs = self.rng.rand()
        if option == 'rand':
            self.kinks = self.rng.rand(2, 2)
            self.kinks.sort(axis=1)
            self.values = self.rng.rand(2, 4)
        elif option == 'semi_rand':
            self.kinks = np.array([[1 / 3, 2 / 3],
                                   [1 / 3, 2 / 3]])
            self.values = np.array([[0.35 * self.rng.rand(), 0.65 + 0.35 * self.rng.rand(), 0.35 * self.rng.rand(),
                                     0.65 + 0.35 * self.rng.rand()],
                                    [0.35 * self.rng.rand(), 0.65 + 0.35 * self.rng.rand(), 0.35 * self.rng.rand(),
                                     0.65 + 0.35 * self.rng.rand()]])
        else:
            self.kinks = np.array([[1 / 3, 2 / 3],
                                   [1 / 3, 2 / 3]])
            self.values = np.array([[0.69, 0.131, 0.907, 0.079],
                                    [0.865, 0.134, 0.75, 0.053]])

    def step(self, action):
        self.time += 1
        kink = self.kinks[action]
        value = self.values[action]
        rew = np.copy(self.obs)

        if self.obs < kink[0]:
            self.obs = value[0] + (value[1] - value[0]) / kink[0] * self.obs
        elif self.obs >= kink[0] and self.obs < kink[1]:
            self.obs = value[1] + (value[2] - value[1]) / (kink[1] - kink[0]) * (self.obs - kink[0])
        else:
            self.obs = value[2] + (value[3] - value[2]) / (1 - kink[1]) * (self.obs - kink[1])
        assert 0 <= self.obs <= 1

        return self.obs, rew, (self.time >= 10), {}

    def reset(self):
        self.time = 0
        self.obs = np.array([self.rng.random()])
        return self.obs

    def get_discrete_mdp(self, num_states=100):
        states = np.linspace(0, 1, num_states)
        rewards = states[None].repeat(2, axis=0)
        dyn_mats = np.zeros((2, num_states, num_states))
        for state_id in range(num_states):
            this_state = states[state_id]
            for action in [0, 1]:
                self.obs = this_state
                self.step(action)
                dst = np.abs(states - self.obs)
                next_state_id = np.argmin(dst)
                dyn_mats[action, state_id, next_state_id] = 1.0

        return states, rewards, dyn_mats

def perform_vi(states, rewards, dyn_mats, gamma=0.9, eps=1e-5):
    # Assume discrete actions and states
    q_values = np.zeros(dyn_mats.shape[:2])

    deltas = []
    while not deltas or deltas[-1] >= eps:
        old = q_values
        q_max = q_values.max(axis=0)
        q_values = rewards + gamma * dyn_mats @ q_max

        deltas.append(np.abs(old - q_values).max())

    return q_values, deltas


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, 400)
        self.fc5 = nn.Linear(400, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
class smallDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(smallDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc5(x)
        return x

class tinyDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(tinyDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc5(x)
        return x


class AttnDQN(nn.Module):
    def __init__(self, input_dim, output_dim, temp):
        super(AttnDQN, self).__init__()
        self.temp = temp
        self.attn = nn.Parameter(torch.zeros(input_dim))
        # self.mask = nn.functional.sigmoid(self.attn)
        self.mask = sigmoid(self.attn, self.temp)
        print("initial mask", self.mask)
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, 400)
        self.fc5 = nn.Linear(400, output_dim)

    def forward(self, x):
        # self.mask = nn.functional.sigmoid(self.attn)
        self.mask = sigmoid(self.attn, self.temp)
        x = x * self.mask
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class smallAttnDQN(nn.Module):
    def __init__(self, input_dim, output_dim, temp):
        super(smallAttnDQN, self).__init__()
        self.temp = temp
        self.attn = nn.Parameter(torch.zeros(input_dim))
        # self.mask = nn.functional.sigmoid(self.attn)
        self.mask = sigmoid(self.attn, self.temp)
        print("initial mask", self.mask)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, output_dim)

    def forward(self, x):
        # self.mask = nn.functional.sigmoid(self.attn)
        self.mask = sigmoid(self.attn, self.temp)
        x = x * self.mask
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc5(x)
        return x


class tinyAttnDQN(nn.Module):
    def __init__(self, input_dim, output_dim, temp):
        super(tinyAttnDQN, self).__init__()
        self.temp = temp
        self.attn = nn.Parameter(torch.zeros(input_dim))
        # self.mask = nn.functional.sigmoid(self.attn)
        self.mask = sigmoid(self.attn, self.temp)
        print("initial mask", self.mask)
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, output_dim)

    def forward(self, x):
        # self.mask = nn.functional.sigmoid(self.attn)
        self.mask = sigmoid(self.attn, self.temp)
        x = x * self.mask
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc5(x)
        return x


def train_dqn(env, save_name, num_episodes=1000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, lr=0.001, batch_size=64, target_update=10):
    input_dim = 1  # Assuming the state is a single value
    output_dim = 2  # Assuming two actions

    # policy_net = DQN(input_dim, output_dim)
    # target_net = DQN(input_dim, output_dim)

    # policy_net = smallDQN(input_dim, output_dim)
    # target_net = smallDQN(input_dim, output_dim)
    policy_net = tinyDQN(input_dim, output_dim)
    target_net = tinyDQN(input_dim, output_dim)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = deque(maxlen=10000)
    epsilon = epsilon_start

    def select_action(state):
        if random.random() < epsilon:
            return random.randint(0, output_dim - 1)
        else:
            with torch.no_grad():
                return policy_net(torch.tensor([state], dtype=torch.float32)).argmax().item()

    def optimize_model():
        if len(memory) < batch_size:
            return
        transitions = random.sample(memory, batch_size)
        batch = list(zip(*transitions))

        states = torch.tensor(batch[0], dtype=torch.float32)
        actions = torch.tensor(batch[1], dtype=torch.int64)
        rewards = torch.tensor(batch[2], dtype=torch.float32)
        next_states = torch.tensor(batch[3], dtype=torch.float32)
        dones = torch.tensor(batch[4], dtype=torch.float32)

        q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = target_net(next_states).max(1)[0]
        expected_q_values = rewards.squeeze(1) + (gamma * next_q_values * (1 - dones))

        loss = nn.functional.mse_loss(q_values, expected_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            optimize_model()

        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}")

    torch.save(policy_net.state_dict(), "./models/dqn_"+save_name+".pth")
    return policy_net


def train_dqn_distractors(env, save_name, BIAS_LOW, BIAS_HIGH, num_controllable_distractors, num_uncontrollable_distractors, num_episodes=1000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, lr=0.001, batch_size=64, target_update=10):
    input_dim = 1 + num_controllable_distractors + num_uncontrollable_distractors  # Assuming the state and distractors are single values
    output_dim = 2  # Assuming two actions

    # policy_net = DQN(input_dim, output_dim)
    # target_net = DQN(input_dim, output_dim)

    # policy_net = smallDQN(input_dim, output_dim)
    # target_net = smallDQN(input_dim, output_dim)
    policy_net = tinyDQN(input_dim, output_dim)
    target_net = tinyDQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = deque(maxlen=10000)
    epsilon = epsilon_start

    def select_action(state):
        if random.random() < epsilon:
            return random.randint(0, output_dim - 1)
        else:
            with torch.no_grad():
                return policy_net(torch.tensor([state], dtype=torch.float32)).argmax().item()

    def optimize_model():
        if len(memory) < batch_size:
            return
        transitions = random.sample(memory, batch_size)
        batch = list(zip(*transitions))

        states = torch.tensor(batch[0], dtype=torch.float32)
        actions = torch.tensor(batch[1], dtype=torch.int64)
        rewards = torch.tensor(batch[2], dtype=torch.float32)
        next_states = torch.tensor(batch[3], dtype=torch.float32)
        dones = torch.tensor(batch[4], dtype=torch.float32)

        q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = target_net(next_states).max(1)[0]
        expected_q_values = rewards.squeeze(1) + (gamma * next_q_values * (1 - dones))

        loss = nn.functional.mse_loss(q_values, expected_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    controllable_variable_bias = np.random.uniform(low=BIAS_LOW, high=BIAS_HIGH, size=(1, num_controllable_distractors))
    random_variable_bias = np.random.uniform(BIAS_LOW, BIAS_HIGH, size=(1, num_uncontrollable_distractors))
    control_weights = np.random.uniform(low=0, high=1. / 2, 
                                                size=(1, num_controllable_distractors))


    uncontrollable_distractors = np.random.uniform(low=-1, high=1, size=(1, num_uncontrollable_distractors)) + random_variable_bias
    controllable_distractors = np.random.uniform(low=-1, high=1, size=(1, num_controllable_distractors)) + controllable_variable_bias


    for episode in range(num_episodes):
        state = env.reset()
        state = np.concatenate((state, controllable_distractors.squeeze(0), uncontrollable_distractors.squeeze(0)))
        total_reward = 0
        done = False

        while not done:
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            uncontrollable_distractors = np.random.uniform(low=-1, high=1, size=(1, num_uncontrollable_distractors)) + random_variable_bias
            controllable_distractors = np.matmul(np.array([[action]]), control_weights) + controllable_variable_bias
            next_state = np.concatenate((next_state, controllable_distractors.squeeze(0), uncontrollable_distractors.squeeze(0)))

            # print("state", state)
            # print("next_state", next_state)
            # print("action", action)
            # print("reward", reward)
            # print("done", done)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            optimize_model()

        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}")

    torch.save(policy_net.state_dict(), "./models/dqn_distractors_"+save_name+".pth")
    return policy_net


def sigmoid(x, temp): 
    return 1 / (1 + torch.exp(-x * temp))

def train_dqn_distractors_attn(env, save_name, sigmoid_temp, BIAS_LOW, BIAS_HIGH, num_controllable_distractors, num_uncontrollable_distractors, \
                               num_episodes=1000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, \
                                lr=0.001, batch_size=64, target_update=10):
    input_dim = 1 + num_controllable_distractors + num_uncontrollable_distractors  # Assuming the state and distractors are single values
    output_dim = 2  # Assuming two actions

    # policy_net = AttnDQN(input_dim, output_dim, sigmoid_temp)
    # target_net = AttnDQN(input_dim, output_dim, sigmoid_temp)
    # policy_net = smallAttnDQN(input_dim, output_dim, sigmoid_temp)
    # target_net = smallAttnDQN(input_dim, output_dim, sigmoid_temp)
    policy_net = tinyAttnDQN(input_dim, output_dim, sigmoid_temp)
    target_net = tinyAttnDQN(input_dim, output_dim, sigmoid_temp)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = deque(maxlen=10000)
    epsilon = epsilon_start

    def select_action(state):
        if random.random() < epsilon:
            return random.randint(0, output_dim - 1)
        else:
            with torch.no_grad():
                return policy_net(torch.tensor([state], dtype=torch.float32)).argmax().item()

    def optimize_model():
        if len(memory) < batch_size:
            return
        transitions = random.sample(memory, batch_size)
        batch = list(zip(*transitions))

        states = torch.tensor(batch[0], dtype=torch.float32)
        actions = torch.tensor(batch[1], dtype=torch.int64)
        rewards = torch.tensor(batch[2], dtype=torch.float32)
        next_states = torch.tensor(batch[3], dtype=torch.float32)
        dones = torch.tensor(batch[4], dtype=torch.float32)

        q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = target_net(next_states).max(1)[0]
        expected_q_values = rewards.squeeze(1) + (gamma * next_q_values * (1 - dones))

        mse_loss = nn.functional.mse_loss(q_values, expected_q_values)
        # l1_loss = policy_net.mask.abs().sum()  # L1 norm
        loss = mse_loss #+ 0.1 * l1_loss
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    controllable_variable_bias = np.random.uniform(low=BIAS_LOW, high=BIAS_HIGH, size=(1, num_controllable_distractors))
    random_variable_bias = np.random.uniform(BIAS_LOW, BIAS_HIGH, size=(1, num_uncontrollable_distractors))
    control_weights = np.random.uniform(low=0, high=1. / 2, 
                                                size=(1, num_controllable_distractors))


    uncontrollable_distractors = np.random.uniform(low=-1, high=1, size=(1, num_uncontrollable_distractors)) + random_variable_bias
    controllable_distractors = np.random.uniform(low=-1, high=1, size=(1, num_controllable_distractors)) + controllable_variable_bias


    for episode in range(num_episodes):
        state = env.reset()
        state = np.concatenate((state, controllable_distractors.squeeze(0), uncontrollable_distractors.squeeze(0)))
        total_reward = 0
        done = False

        while not done:
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            uncontrollable_distractors = np.random.uniform(low=-1, high=1, size=(1, num_uncontrollable_distractors)) + random_variable_bias
            controllable_distractors = np.matmul(np.array([[action]]), control_weights) + controllable_variable_bias
            next_state = np.concatenate((next_state, controllable_distractors.squeeze(0), uncontrollable_distractors.squeeze(0)))

            # print("state", state)
            # print("next_state", next_state)
            # print("action", action)
            # print("reward", reward)
            # print("done", done)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            optimize_model()

        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 100 == 0:
            print("mask", policy_net.mask)

        print(f"Episode {episode}, Total Reward: {total_reward}")

    torch.save(policy_net.state_dict(), "./models/dqn_distractors_attn_"+save_name+".pth")
    return policy_net




def plot_with_ci(states, error_data, label, color):
    print(error_data.shape)
    mean = np.mean(error_data, axis=0)  # shape: (num_states,)
    print(mean.shape)
    std = np.std(error_data, axis=0)
    ci = 1.96 * std / np.sqrt(error_data.shape[0])  # 95% confidence interval

    plt.plot(states, mean, label=label, color=color)
    plt.fill_between(states, mean - ci, mean + ci, alpha=0.3, color=color)

def compare_methods_per_state(errors_A, errors_B, alpha=0.05):
    """
    errors_A, errors_B: shape (num_seeds, num_states)
    Returns: binary array of shape (num_states,) where 1 = significant difference
    """
    num_states = errors_A.shape[1]
    p_values = np.array([
        ttest_rel(errors_A[:, i], errors_B[:, i]).pvalue
        for i in range(num_states)
    ])
    return p_values < alpha, p_values


if __name__=="__main__":
    num_states = 20
    torch.manual_seed(0)
    mdp = RandMDP(seed=0, option='rand')
    states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
    true_q_values, losses = perform_vi(states, rewards, dyn_mats)

    plt.plot(states, true_q_values[0], label='true Q(s, a=0)')
    plt.plot(states, true_q_values[1], label='true Q(s, a=1)')
    # plt.legend()
    # plt.savefig("true_q.pdf")

    #############################################################################
    eps = 2000
    
    states = np.linspace(0, 1, num_states)

    seeds = 20
    q_values_oracle = [[[], []] for seed in range(seeds)]
    q_values_full = [[[], []] for seed in range(seeds)]
    q_values_attn = [[[], []] for seed in range(seeds)]


    for seed in range(seeds):

        set_seed(seed)
        print("running seed", seed) 

        target_update = 100
        remark = "_fixed_distractors_tiny_dqn_rand0_mdp"

        save_name = "seed{}_{}_eps_{}_target_update_{}.pdf".format(seed, eps, target_update, remark)
        dqn = train_dqn(mdp, save_name, num_episodes=eps, target_update=target_update)
        for state in states:
            q_values_oracle[seed][0].append(dqn(torch.tensor([state], dtype=torch.float32))[0].item())
            q_values_oracle[seed][1].append(dqn(torch.tensor([state], dtype=torch.float32))[1].item())
        
        
        sigmoid_temp = 10
        BIAS_LOW, BIAS_HIGH = 1, 2
        num_controllable_distractors = 20
        num_uncontrollable_distractors = 20
        controllable_variable_bias = np.random.uniform(low=BIAS_LOW, high=BIAS_HIGH, size=(1, num_controllable_distractors))
        random_variable_bias = np.random.uniform(BIAS_LOW, BIAS_HIGH, size=(1, num_uncontrollable_distractors))
        control_weights = np.random.uniform(low=0, high=1. / 2, 
                                                    size=(1, num_controllable_distractors))


        uncontrollable_distractors = np.random.uniform(low=-1, high=1, size=(num_states, num_uncontrollable_distractors)) + random_variable_bias
        controllable_distractors = np.random.uniform(low=-1, high=1, size=(num_states, num_controllable_distractors)) + controllable_variable_bias

        uncontrollable_distractors = random_variable_bias * np.ones((num_states, num_uncontrollable_distractors))
        controllable_distractors = controllable_variable_bias * np.ones((num_states, num_controllable_distractors))

        states_d = np.concatenate((states.reshape(num_states, 1), controllable_distractors, uncontrollable_distractors), axis=1)

        save_name = "seed{}_{}_eps_{}_{}_distractors_{}_target_update_{}.pdf".format(seed, eps, num_controllable_distractors, num_uncontrollable_distractors, target_update, remark)
        dqn = train_dqn_distractors(mdp, save_name, BIAS_LOW, BIAS_HIGH, num_controllable_distractors, num_uncontrollable_distractors, num_episodes=eps, target_update=target_update)
        for state in states_d:
            q_values_full[seed][0].append(dqn(torch.tensor(state, dtype=torch.float32))[0].item())
            q_values_full[seed][1].append(dqn(torch.tensor(state, dtype=torch.float32))[1].item())
        
        save_name = "seed{}_{}_eps_{}_{}_distractors_{}_target_update_{}_sigtemp_{}.pdf".format(seed, eps, num_controllable_distractors, num_uncontrollable_distractors, target_update, sigmoid_temp, remark)
        dqn = train_dqn_distractors_attn(mdp, save_name, sigmoid_temp, BIAS_LOW, BIAS_HIGH, num_controllable_distractors, num_uncontrollable_distractors, num_episodes=eps, target_update=target_update)
        for state in states_d:
            q_values_attn[seed][0].append(dqn(torch.tensor(state, dtype=torch.float32))[0].item())
            q_values_attn[seed][1].append(dqn(torch.tensor(state, dtype=torch.float32))[1].item())
    
    # error_oracle = np.mean((true_q_values - np.array(q_values_oracle)) ** 2, axis=0)
    # error_full = np.mean((true_q_values - np.array(q_values_full)) ** 2, axis=0)
    # error_attn = np.mean((true_q_values - np.array(q_values_attn)) ** 2, axis=0)

    error_oracle = (true_q_values - np.array(q_values_oracle)) ** 2
    error_full = (true_q_values - np.array(q_values_full)) ** 2
    error_attn = (true_q_values - np.array(q_values_attn)) ** 2

    q_values_oracle = np.mean(np.array(q_values_oracle), axis=0)
    q_values_full = np.mean(np.array(q_values_full), axis=0)
    q_values_attn = np.mean(np.array(q_values_attn), axis=0)
    q_values_oracle_std = np.std(np.array(q_values_oracle), axis=0)
    q_values_full_std = np.std(np.array(q_values_full), axis=0)
    q_values_attn_std = np.std(np.array(q_values_attn), axis=0)

    plt.errorbar(states, q_values_oracle[0], \
                 yerr=q_values_oracle_std[0], \
                    label='dqn Q(s, a=0)',\
                  errorevery=1, \
                    alpha=0.7\
                        )


    plt.plot(states, q_values_oracle[0], label='dqn Q(s, a=0)')
    plt.plot(states, q_values_oracle[1], label='dqn Q(s, a=1)')

    plt.plot(states, q_values_full[0], label='dqn distractors Q(s, a=0)')
    plt.plot(states, q_values_full[1], label='dqn distractors Q(s, a=1)')

    plt.plot(states, q_values_attn[0], label='dqn distractors attn Q(s, a=0)')
    plt.plot(states, q_values_attn[1], label='dqn distractors attn Q(s, a=1)')
    plt.legend()
    plt.savefig("q_values_{}_states_{}_eps_{}_{}_distractors_{}_target_update_{}_sigtemp_{}_seeds_{}.pdf".format(num_states, eps, num_controllable_distractors, num_uncontrollable_distractors, target_update, sigmoid_temp, seeds, remark))
    plt.close()


    print(error_oracle.shape)


    # === Action a=0
    plt.figure()
    plot_with_ci(states, error_oracle[:, 0, :], label='dqn oracle error Q(s, a=0)', color='blue')
    plot_with_ci(states, error_full[:, 0, :], label='dqn full error Q(s, a=0)', color='green')
    plot_with_ci(states, error_attn[:, 0, :], label='dqn attn error Q(s, a=0)', color='red')

    # Compare attn vs full formally
    significant, pvals = compare_methods_per_state(error_attn[:, 0, :], error_full[:, 0, :])
    print(pvals)
    for s in range(len(states)):
        if significant[s]:
            plt.scatter(states[s], 0, color='black', marker='*', s=40)  # Mark significant points

    plt.xlabel('States')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig("error_a0_{}_states_{}_eps_{}_{}_distractors_{}_target_update_{}_sigtemp_{}_seeds_{}.pdf".format(
        num_states, eps, num_controllable_distractors, num_uncontrollable_distractors,
        target_update, sigmoid_temp, seeds, remark))
    plt.close()

    # === Action a=1
    plt.figure()
    plot_with_ci(states, error_oracle[:, 1, :], label='dqn oracle error Q(s, a=1)', color='blue')
    plot_with_ci(states, error_full[:, 1, :], label='dqn full error Q(s, a=1)', color='green')
    plot_with_ci(states, error_attn[:, 1, :], label='dqn attn error Q(s, a=1)', color='red')
    
    # Compare attn vs full formally
    significant, pvals = compare_methods_per_state(error_attn[:, 1, :], error_full[:, 1, :])
    print(pvals)
    for s in range(len(states)):
        if significant[s]:
            plt.scatter(states[s], 0, color='black', marker='*', s=40)  # Mark significant points

    
    plt.xlabel('States')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig("error_a1_{}_states_{}_eps_{}_{}_distractors_{}_target_update_{}_sigtemp_{}_seeds_{}.pdf".format(
        num_states, eps, num_controllable_distractors, num_uncontrollable_distractors,
        target_update, sigmoid_temp, seeds, remark))
    plt.close()