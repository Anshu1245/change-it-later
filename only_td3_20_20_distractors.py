import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["MUJOCO_GL"]="egl"
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


import argparse

from distutils.util import strtobool
import pickle
import matplotlib.pyplot as plt

# from dm_control import manipulation
# from dm_control import suite, viewer
# from dmc_utils import *
from DMCGym_scaled import *

import warnings
warnings.filterwarnings("ignore")

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1331
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    device: str = "cuda"
    """device gpu or cpu"""
    save_model_frequency: int = 200000
    """save model freq"""
    

    # Algorithm specific arguments
    env_id: str = "walker"
    """the id of the environment"""
    task_id: str = "run"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""


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
            # idx=0 is the training env
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

def save_checkpoint(actor, actor_target, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer, run_name, global_step, ckpt_path=None):
        if not os.path.exists('{}/'.format(run_name)):
            os.makedirs('{}/'.format(run_name))
        if ckpt_path is None:
            ckpt_path = "{}/{}_steps".format(run_name, global_step)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': actor.state_dict(),
                    'policy_target_state_dict': actor_target.state_dict(),   
                    'critic1_state_dict': qf1.state_dict(),
                    'critic1_target_state_dict': qf1_target.state_dict(),
                    'critic2_state_dict': qf2.state_dict(),
                    'critic2_target_state_dict': qf2_target.state_dict(),
                    'critic_optimizer_state_dict': q_optimizer.state_dict(),
                    'policy_optimizer_state_dict': actor_optimizer.state_dict()}, ckpt_path)

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
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


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
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
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    print("running experiment in env", args.env_id, "and task", args.task_id)
    run_name = f"seed{args.seed}__{args.env_id}-{args.task_id}__{args.exp_name}"
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

    # device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('using device:', device)

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.task_id, args.seed, 0, args.capture_video, run_name)])
    test_envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.task_id, args.seed, 1, args.capture_video, run_name)])
    print("old envs.single_observation_space", envs.single_observation_space)
    # num_distractors = 40
    num_controllable_distractors = 20
    num_uncontrollable_distractors = 20

    print("num distractors", num_controllable_distractors+num_uncontrollable_distractors)
    envs.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(envs.single_observation_space.shape[0]+num_controllable_distractors+num_uncontrollable_distractors,), dtype=np.float32)
    print("new envs.single_observation_space", envs.single_observation_space)
    test_envs.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(test_envs.single_observation_space.shape[0]+num_controllable_distractors+num_uncontrollable_distractors,), dtype=np.float32)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

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

    rewards_over_episodes = []
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    BIAS_LOW = 1
    BIAS_HIGH = 2
    controllable_variable_bias = np.random.uniform(low=BIAS_LOW, high=BIAS_HIGH, size=(1, num_controllable_distractors))
    random_variable_bias = np.random.uniform(BIAS_LOW, BIAS_HIGH, size=(1, num_uncontrollable_distractors))
    control_weights = np.random.uniform(low=0, high=1. / np.prod(envs.single_action_space.shape), 
                                                size=(np.prod(envs.single_action_space.shape), num_controllable_distractors))


    uncontrollable_distractors = np.random.uniform(low=-1, high=1, size=(1, num_uncontrollable_distractors)) + random_variable_bias
    controllable_distractors = np.random.uniform(low=-1, high=1, size=(1, num_controllable_distractors)) + controllable_variable_bias
    obs = np.concatenate((obs, controllable_distractors, uncontrollable_distractors), axis=1)


    if not os.path.exists('{}/'.format(run_name)):
        os.makedirs('{}/'.format(run_name))

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        uncontrollable_distractors = np.random.uniform(low=-1, high=1, size=(1, num_uncontrollable_distractors)) + random_variable_bias
        controllable_distractors = np.matmul(actions, control_weights) + controllable_variable_bias
        next_obs = np.concatenate((next_obs, controllable_distractors, uncontrollable_distractors), axis=1)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

            # EVALUATE for 50 episodes without reset!
            print("testing...")
            mean_returns = []
            ep = 0
            # print("test episode {}".format(ep))
            test_obs, _ = test_envs.reset()

            test_controllable_variable_bias = np.random.uniform(BIAS_LOW, BIAS_HIGH, (1, num_controllable_distractors))
            test_random_variable_bias = np.random.uniform(BIAS_LOW, BIAS_HIGH, (1, num_uncontrollable_distractors))
            test_control_weights = np.random.uniform(0, 1. / np.prod(test_envs.single_action_space.shape), 
                                                        (np.prod(test_envs.single_action_space.shape), num_controllable_distractors))


            test_uncontrollable_distractors = np.random.uniform(low=-1, high=1, size=(1, num_uncontrollable_distractors)) + test_random_variable_bias
            test_controllable_distractors = np.random.uniform(low=-1, high=1, size=(1, num_controllable_distractors)) + test_controllable_variable_bias
            test_obs = np.concatenate((test_obs, test_controllable_distractors, test_uncontrollable_distractors), axis=1)
            
            while True:
                test_actions = actor(torch.Tensor(test_obs).to(device))
                test_actions = test_actions.detach().cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)
                test_next_obs, test_rewards, test_terminations, test_truncations, test_infos = test_envs.step(test_actions)
                
                test_uncontrollable_distractors = np.random.uniform(low=-1, high=1, size=(1, num_uncontrollable_distractors)) + test_random_variable_bias
                test_controllable_distractors = np.matmul(test_actions, test_control_weights) + test_controllable_variable_bias
                test_next_obs = np.concatenate((test_next_obs, test_controllable_distractors, test_uncontrollable_distractors), axis=1)


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
                # print("final obs", infos["final_observation"][idx].shape)
                infos["final_observation"][idx] = np.concatenate((infos["final_observation"][idx], controllable_distractors.squeeze(0), uncontrollable_distractors.squeeze(0)))
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale

                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
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
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    
        if (global_step + 1) % args.save_model_frequency == 0:
                save_checkpoint(actor, target_actor, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer, run_name, global_step) #+"__"+str(global_step))
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