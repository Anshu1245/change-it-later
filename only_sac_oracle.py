import argparse
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["MUJOCO_GL"]="egl"

import random
import time
from distutils.util import strtobool
import pickle
import matplotlib.pyplot as plt

import gymnasium as gym
from dm_control import manipulation
from dm_control import suite, viewer
# from dmc_utils import *
from DMCGym_scaled import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt-dir", type=str, default='./saved_models/',
        help="models saved to this directory")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1331,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    # parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model-frequency", type=int, default=2e5,
        help="save model frequency") 
    parser.add_argument("--device", type=str, default="cuda",
        help="use cpu or gpu")

    # Algorithm specific arguments
    # parser.add_argument("--env-id", type=str, default="manipulator-bring_ball",
        # help="the id of the environment")
    # parser.add_argument("--env-id", type=str, default="manipulation-stack_2_bricks_moveable_base_features",
    #     help="the id of the environment")
    # parser.add_argument("--env-id", type=str, default="reacher-hard",
    #     help="the id of the environment")
    parser.add_argument("--env-id", type=str, default="walker",
        help="the id of the environment")
    parser.add_argument("--task-id", type=str, default="run",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(5e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=5e-3,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=1e4,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=1,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient. default=0.2")
    parser.add_argument("--max-grad-norm", type=float, default=10,
            help="clip gradient to this max norm")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args


# Set environment seed if supported
def set_env_seed(env, seed):
    if hasattr(env, 'seed'):
        env.seed(seed)
    if hasattr(env.action_space, 'seed'):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, 'seed'):
        env.observation_space.seed(seed)


def make_env(env_id, task_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            print("creating train env...")
            # env = gym.make(env_id, render_mode="rgb_array")
            # env = DMCGym('cartpole', 'balance')#, visualize_reward=False)#, seed=seed)
            env = DMCGym(env_id, task_id)
            set_env_seed(env, seed)
            # env = DMCGym('manipulator', 'bring_ball')
            # env = DMCGym('stack_2_bricks_moveable_base_features', '')
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            print("creating test env...")
            # env = gym.make(env_id)
            # env = DMCGym('cartpole', 'balance')
            # env = DMCGym('manipulator', 'bring_ball')
            # env = DMCGym('stack_2_bricks_moveable_base_features', '')
            env = DMCGym(env_id, task_id)
            set_env_seed(env, seed)
            # env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

# save model parameters
def save_checkpoint(actor, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer, run_name, global_step, ckpt_path=None):
        if not os.path.exists('{}/'.format(run_name)):
            os.makedirs('{}/'.format(run_name))
        if ckpt_path is None:
            ckpt_path = "{}/{}_steps".format(run_name, global_step)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': actor.state_dict(),
                    'critic1_state_dict': qf1.state_dict(),
                    'critic1_target_state_dict': qf1_target.state_dict(),
                    'critic2_state_dict': qf2.state_dict(),
                    'critic2_target_state_dict': qf2_target.state_dict(),
                    'critic_optimizer_state_dict': q_optimizer.state_dict(),
                    'policy_optimizer_state_dict': actor_optimizer.state_dict()}, ckpt_path)


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        action = torch.clamp(action, min=-1, max=1)
        # added the above line as action was becoming 1.000001 at some step
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = parse_args()
    print("running experiment in env", args.env_id, "and task", args.task_id)
    run_name = f"seed{args.seed}__{args.env_id}-{args.task_id}__{args.exp_name}" #__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    print("using seed", args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('using device:', device)

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.task_id, args.seed, 0, args.capture_video, run_name)])
    test_envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.task_id, args.seed, 1, args.capture_video, run_name)])
    
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    test_envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    alpha_start = 0.5
    alpha_finish = 0.1
    alpha_decay = 1
    alpha_step = 0
    rewards_over_episodes = []
    obs, _ = envs.reset()#seed=args.seed)

    if not os.path.exists('{}/'.format(run_name)):
        os.makedirs('{}/'.format(run_name))

    for global_step in range(args.total_timesteps):
        # envs.render()

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            # print("action from actor", actions, actions.dtype)
            actions = actions.detach().cpu().numpy()
            # print("action after detach", actions, actions.dtype)

        # TRY NOT TO MODIFY: execute the game and log data.
        # print("actions", actions)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        # print("truncations", truncations)
        # print(rewards) if rewards>0.01 else print("nope")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # rewards_over_episodes.append(info["episode"]["r"][0])
                # print("rewards over eps", rewards_over_episodes)
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

            # # EVALUATE for 50 episodes by _resetting episode_!
            # print("testing...")
            # mean_returns = []
            # for ep in range(50):
            #     # print("test episode {}".format(ep))
            #     obs, _ = test_envs.reset()
            #     ep_returns = 0
            #     step = 0
            #     while True:
            #         step += 1
            #         _, _, actions = actor.get_action(torch.Tensor(obs).to(device))
            #         actions = actions.detach().cpu().numpy()
            #         next_obs, rewards, terminations, truncations, infos = test_envs.step(actions)
            #         ep_returns += rewards[0]
            #         obs = next_obs

            #         if truncations[0]:
            #             break
                    
            #     # print("test episode returns {} after {} steps".format(ep_returns, step))
            #     mean_returns.append(ep_returns)
            # rewards_over_episodes.append(np.mean(mean_returns))
            # print("TEST RETURNS", rewards_over_episodes[-1])

            # EVALUATE for 50 episodes without reset!
            print("testing...")
            mean_returns = []
            ep = 0
            # print("test episode {}".format(ep))
            test_obs, _ = test_envs.reset()
            while True:
                _, _, test_actions = actor.get_action(torch.Tensor(test_obs).to(device))
                test_actions = test_actions.detach().cpu().numpy()
                test_next_obs, test_rewards, test_terminations, test_truncations, test_infos = test_envs.step(test_actions)

                if "final_info" in test_infos:
                    for test_info in test_infos["final_info"]:
                        ep += 1
                        # print(info["episode"]["r"][0])
                        mean_returns.append(test_info["episode"]["r"][0])
                        # print("rewards over eps", rewards_over_episodes)
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        break

                # real_next_obs = next_obs.copy()
                # for idx, trunc in enumerate(truncations):
                #     if trunc:
                #         real_next_obs[idx] = infos["final_observation"][idx]
                test_obs = test_next_obs

                if ep==50:
                    break
                
            # print("test episode returns {} after {} steps".format(ep_returns, step))
            rewards_over_episodes.append(np.mean(mean_returns))
            print("TEST RETURNS", rewards_over_episodes[-1])

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            # torch.nn.utils.clip_grad_norm_(qf1.parameters(), args.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(qf2.parameters(), args.max_grad_norm)
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                    actor_optimizer.step()

                    if args.autotune:
                        # print("AUTOTUNE ON!")
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        # torch.nn.utils.clip_grad_norm_(log_alpha.parameters(), args.max_grad_norm)
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

                    else:
                        alpha = (alpha_start - alpha_finish) * np.exp(-alpha_decay * alpha_step / args.total_timesteps) + alpha_finish                     
                        alpha_step += 1

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        if (global_step + 1) % args.save_model_frequency == 0:
            save_checkpoint(actor, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer, run_name, global_step) #+"__"+str(global_step))
            with open(run_name + '/rewards_over_episodes.pkl', 'wb') as f:
                pickle.dump(rewards_over_episodes, f)
            
        if (global_step+1) % 5000 == 0:
            
            # Compute moving average
            window_size = 1
            weights = np.repeat(1.0, window_size) / window_size
            rewards_over_episodes_smooth = np.convolve(rewards_over_episodes, weights, 'valid')

            # Plot the smoothed data
            plt.plot(rewards_over_episodes_smooth)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward (Moving Average)')
            plt.savefig(run_name + '/rewards_over_episodes.pdf')

    envs.close()
    writer.close()

    