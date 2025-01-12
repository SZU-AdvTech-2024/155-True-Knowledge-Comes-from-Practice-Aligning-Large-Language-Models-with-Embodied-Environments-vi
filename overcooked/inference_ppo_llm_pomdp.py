import os
import sys
import pathlib

root = str(pathlib.Path(__file__).parents[2])
sys.path.append(root)

import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper
# from policy_pomdp import LLMAgent

# def make_env(env_id, seed, idx, capture_video, run_name, env_params):
#     def thunk():

#         env = gym.make(env_id, **env_params)
#         if capture_video:
#             if idx == 0:
#                 env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         return env

#     return thunk

def make_env(env_id, seed, idx, capture_video, run_name, env_params):
    def thunk():

        env = gym.make(env_id, **env_params)
        env = MacEnvWrapper(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def main():

    device = torch.device("cuda")

    # env_params = {
    #     'seed': 10,
    #     'debug': False,
    # }
    
    # parser.add_argument('--env-reward',             action='store',        type=float, nargs=4,  default=[0.1, 1, 0, 0.001],    help='The reward list of the env')
    # 0.2 1 0.1 0.001
    rewardList = {"subtask finished": 0.2, "correct delivery": 1, "wrong delivery": 0.1, "step penalty": 0.001}
    TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]
    env_params = {'grid_dim': [7,7],
                'task': TASKLIST[0],
                'rewardList': rewardList,
                'map_type': "A",
                'n_agent': 1,
                'obs_radius': 2,
                'mode': "vector",
                'debug': False
            }  # 修改task

    print("play tomato salad v4")

    envs = gym.vector.SyncVectorEnv(
        [make_env("Overcooked-LLMA-v4", 10, 0, False, "tmp", env_params) for i in
         range(1)]
    )

    print("play agent")
    # workdir/VirtualHome-v1__heat_pancake_ppo_llm__10__20241206_17_28_33/saved_models/epoch_0001
    # load_path = os.path.join(root, "checkpoints", "food_preparation", "lora")
    load_path = os.path.join(root, "workdir/Overcooked-LLMA-v4__tomato_salad_llm__1__20250106_14_51_35/saved_models/epoch_0001")

    
    use_dropout = False  # 模型是否是使用mcdropout，需要修改
    
    if use_dropout:
        from policy_pomdp_with_dropout import LLMAgent
    else:
        from policy_pomdp import LLMAgent

    agent = LLMAgent(normalization_mode="word",
                     load_path=load_path,
                     load_8bit=False,
                     task=0)

    success_rate = 0

    reward_list = []
    step_list = []
    for i in range(100):
        steps = 0
        done = False
        rewards = 0
        discount = 1
        obs = envs.reset()
        while not done:
            steps += 1
            action = agent.get_action_and_value(obs, return_value= False)[0].cpu().numpy()
            print("action", action)
            obs, reward, done, info = envs.step(action)
            rewards += reward * discount
            discount *= 0.95
        reward_list.append(rewards)
        step_list.append(steps)
        if rewards > 0:
            success_rate += 1
    
        print(steps, rewards)

    print(np.mean(reward_list), np.std(reward_list))
    print(np.mean(step_list), np.std(step_list))
    print(success_rate)


if __name__ == '__main__':
    main()

