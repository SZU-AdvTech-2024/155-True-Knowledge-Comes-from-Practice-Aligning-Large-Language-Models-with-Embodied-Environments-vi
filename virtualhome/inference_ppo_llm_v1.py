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
import virtual_home
# from policy_v1 import LLMAgent

def make_env(env_id, seed, idx, capture_video, run_name, env_params):
    def thunk():

        env = gym.make(env_id, **env_params)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk

def main():

    device = torch.device("cuda")

    env_params = {
        'seed': 10,
        'debug': False,
    }

    print("play virtual home v1")

    envs = gym.vector.SyncVectorEnv(
        [make_env("VirtualHome-v1", 10, 0, False, "tmp", env_params) for i in
         range(1)]
    )

    print("play agent")
    # workdir/VirtualHome-v1__heat_pancake_ppo_llm__10__20241206_17_28_33/saved_models/epoch_0001
    # load_path = os.path.join(root, "checkpoints", "food_preparation", "lora")
    load_path = os.path.join(root, "workdir/VirtualHome-v1__heat_pancake_ppo_llm__dropout__10__20250105_16_04_30/saved_models/epoch_0002")
    # 保存好的模型路径，需要修改

    normalization_mode = "word"
    
    use_dropout = True  # 模型是否是使用mcdropout，需要修改
    
    if use_dropout:
        from policy_v1_with_dropout import LLMAgent
    else:
        from policy_v1 import LLMAgent
    
    agent = LLMAgent(normalization_mode=normalization_mode,
                     load_path=load_path,
                     load_8bit=False)
    
    # time_str = time.strftime("%Y%m%d_%H_%M_%S", time.localtime(time.time()))
    
    # run_name = f"food_preparation_original_{time_str}"
    
    # track = True
    # if track:
    #     import wandb

    #     wandb.init(
    #         sync_tensorboard=True,
    #         project="virtual_home_v1",
    #         name=run_name,  # 可根据实验命名
    #         config={
    #             "normalization_mode": normalization_mode
    #         }
    #     )

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
    #     # 记录单次实验的数据到 WandB
    #     wandb.log({"episode_reward": rewards, "episode_steps": steps})

    # # 总结性统计数据
    # wandb.log({
    #     "mean_reward": np.mean(reward_list),
    #     "std_reward": np.std(reward_list),
    #     "mean_steps": np.mean(step_list),
    #     "std_steps": np.std(step_list),
    #     "success_rate": success_rate / 100,
    # })
    
    # cumulative_rewards = np.cumsum(reward_list) / np.arange(1, len(reward_list) + 1)
    # wandb.log({"cumulative_rewards": cumulative_rewards[-1]})
    # for step, reward in enumerate(reward_list):
    #     wandb.log({"Total Return": reward, "Steps": step})
    
    print(np.mean(reward_list), np.std(reward_list))
    print(np.mean(step_list), np.std(step_list))
    print(success_rate)


if __name__ == '__main__':
    main()

