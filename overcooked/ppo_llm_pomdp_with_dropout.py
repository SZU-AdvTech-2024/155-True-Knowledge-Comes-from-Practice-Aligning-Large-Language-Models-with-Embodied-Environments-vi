# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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
from policy_pomdp_with_dropout import LLMAgent

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    
    parser.add_argument("--policy-learning-rate", type=float, default=1e-6,
        help="the learning rate of the optimizer")
    parser.add_argument("--value-learning-rate", type=float, default=3e-5,
        help="the learning rate of the optimizer")
    
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=32,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    # parser.add_argument("--num-minibatches", type=int, default=16,
    #     help="the number of mini-batches")
    parser.add_argument("--policy-num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--value-num-minibatches", type=int, default=4,
        help="the number of mini-batches")

    parser.add_argument("--update-epochs", type=int, default=1,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    parser.add_argument('--gradient-checkpointing-steps', action='store',  type=int,             default=8,                     help='The number of steps for gradient checkpointing')
    parser.add_argument('--critic-warm-up-steps',   action='store',        type=int,             default=5000,                  help='The number of time steps to warm up critic')
    
    # ?
    #env_parameter
    parser.add_argument('--env-id',                 action='store',        type=str,             default='Overcooked-LLMA-v3',  help='Domain name')
    parser.add_argument('--n-agent',                action='store',        type=int,             default=1,                     help='Number of agents')
    parser.add_argument('--grid-dim',               action='store',        type=int,   nargs=2,  default=[7,7],                 help='Grid world size')
    parser.add_argument('--task',                   action='store',        type=int,             default=3,                     help='The receipt agent cooks')
    parser.add_argument('--map-type',               action='store',        type=str,             default="A",                   help='The type of map')
    parser.add_argument('--obs-radius',             action='store',        type=int,             default=2,                     help='The radius of the agents')
    parser.add_argument('--env-reward',             action='store',        type=float, nargs=4,  default=[0.1, 1, 0, 0.001],    help='The reward list of the env')
    parser.add_argument('--mode',                   action='store',        type=str,             default="vector",              help='The type of the observation(vector/image)')    
    parser.add_argument('--debug',                  action='store',        type=bool,            default=False,                 help='Whehter print the debug information and render') 
    
    
    parser.add_argument('--load-8bit',              action='store',        type=bool,            default=False,                 help='Whether to convert model to 8bits')
    
    parser.add_argument('--save-path',              action='store',        type=str,             default="saved_models",        help='The path to save the checkpoint')
    parser.add_argument('--save-interval',          action='store',        type=int,             default=10,                    help='The interval for saving model for certain num_updates')
    parser.add_argument('--resume',                 action='store',        type=bool,            default=False,                 help='Whehter resume from previous checkpoint')
    parser.add_argument('--load-path',              action='store',        type=str,             default="saved_models",        help='The path to load the checkpoint')    
    parser.add_argument('--record-path',            action='store',        type=str,             default="llm5_runs",           help='The path to save the tensorboard results')    

    parser.add_argument('--normalization-mode',     action='store',        type=str,             default="token",               help='The normalization mode of how to deal with the logits of each token')    

    

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    
    args.policy_minibatch_size = int(args.batch_size // args.policy_num_minibatches)
    args.value_minibatch_size = int(args.batch_size // args.value_num_minibatches)
    
    # fmt: on
    return args


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

if __name__ == "__main__":
    args = parse_args()
    time_str = time.strftime("%Y%m%d_%H_%M_%S", time.localtime(time.time()))
    run_name = f"{args.env_id}__{args.exp_name}__dropout__{args.seed}__{time_str}"
    args.track = True
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
    writer = SummaryWriter(f"{args.record_path}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # If you're using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ?
    rewardList = {"subtask finished": args.env_reward[0], "correct delivery": args.env_reward[1], "wrong delivery": -args.env_reward[2], "step penalty": -args.env_reward[3]}
    TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]
    env_params = {'grid_dim': args.grid_dim,
                    'task': TASKLIST[args.task],
                    'rewardList': rewardList,
                    'map_type': args.map_type,
                    'n_agent': args.n_agent,
                    'obs_radius': args.obs_radius,
                    'mode': args.mode,
                    'debug': args.debug
                }

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, env_params) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    if args.resume:
        agent = LLMAgent(normalization_mode=args.normalization_mode, load_path=args.load_path, load_8bit=args.load_8bit, task=args.task)
    else:
        agent = LLMAgent(normalization_mode=args.normalization_mode, load_8bit=args.load_8bit, task=args.task)

    policy_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, agent.actor.parameters()), lr=args.policy_learning_rate, eps=1e-5, weight_decay=0)
    value_optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.critic.parameters()), lr=args.value_learning_rate, eps=1e-5)
        
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    steps = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    pre_global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size 
    num_critic_warm_up_updates = args.critic_warm_up_steps // args.batch_size
    
    is_warmup = True
    episodic_returns = []
    for update in range(1, num_updates + 1 + num_critic_warm_up_updates):
        if is_warmup and update > num_critic_warm_up_updates:
            is_warmup = False

        # Annealing the rate if instructed to do so.
        if args.anneal_lr and not is_warmup:
            frac = 1.0 - (update - 1.0 - num_critic_warm_up_updates) / num_updates
            policy_optimizer.param_groups[0]["lr"] = frac * args.policy_learning_rate
            value_optimizer.param_groups[0]["lr"] = frac * args.value_learning_rate
            

        #----------------Addition----------------    
        uncertainty_weight = 0.1
        #----------------------------------------
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # action, logprob, _, value = agent.get_action_and_value(next_obs)
                
                #----------------Addition----------------
                action, logprob, _, value, uncertainty = agent.get_action_and_value(next_obs)
                adjusted_value = value - uncertainty_weight * uncertainty
                values[step] = adjusted_value.flatten()
                #----------------------------------------
                
                # values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            #print("info", info)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            steps[step] = torch.Tensor([item['macro_action_steps'] for item in info]).to(device)

            for item in info:
                if "episode" in item.keys():
                    #---------------------Addition-------------------------
                    episodic_return = item['episode']['r']
                    # if len(total_returns) > 0:
                    #     total_return = (total_returns[len(total_returns) - 1] + episodic_return) / 2
                    # else
                    #     total_return = episodic_return
                    episodic_returns.append(episodic_return)
                    if len(episodic_returns) > 10:
                        window_size = 10  # You can change the window size as needed
                        recent_returns = episodic_returns[-window_size:]  # Last `window_size` episodes
                        total_return = sum(recent_returns) / len(recent_returns)  # Average
                    else:
                        total_return = sum(episodic_returns) / len(episodic_returns)  # Average
                    #---------------------Addition-------------------------
                    writer.add_scalar("charts/total_return", total_return, global_step)
                    
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}, episodic_length={item['episode']['l']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break
        
        # bootstrap value if not done
        with torch.no_grad():
            #-------------------Addition-------------------
            next_value, next_uncertainty = agent.get_value(next_obs)
            next_value = next_value.reshape(1, -1)
            next_uncertainty = next_uncertainty.reshape(1, -1)
            #----------------------------------------------
            
            # next_value = agent.get_value(next_obs).reshape(1, -1)
            
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                # ?
                discount = torch.pow(args.gamma, steps[t])
                delta = rewards[t] + discount * nextvalues * nextnonterminal - values[t]
                
                #--------------Addition----------------
                adjusted_delta = delta / (1 + next_uncertainty)
                advantages[t] = lastgaelam = adjusted_delta + discount * args.gae_lambda * nextnonterminal * lastgaelam
                #-------------------------------------
                
                # advantages[t] = lastgaelam = delta + discount * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        #----------------Addition----------------
        # b_uncertainties = uncertainties.reshape(-1)
        #---------------------------------------

        # Optimizing the policy and value network
        # 优化策略和价值网络
        b_inds = np.arange(args.batch_size) # 创建一个包含所有批次索引的数组
        clipfracs = [] # 存储被裁剪的策略比率的比例
        kl_explode = False # 标记是否 KL 散度爆炸
        policy_update_steps = 0 # 记录策略更新的步数
        pg_loss = torch.tensor(0) # 初始化策略损失
        entropy_loss = torch.tensor(0) # 初始化熵损失
        old_approx_kl = torch.tensor(0) # 旧的 KL 散度估计值
        approx_kl = torch.tensor(0) # 当前的 KL 散度估计值
        total_approx_kl = torch.tensor(0) # 总的 KL 散度
        
        # 开始 PPO 的更新循环，更新多个 epoch
        for epoch in range(args.update_epochs):
            if kl_explode: # 如果 KL 散度爆炸，提前停止
                break
            #update value
            # 更新价值网络
            np.random.shuffle(b_inds) # 打乱索引顺序以确保批次的随机性
            for start in range(0, args.batch_size, args.value_minibatch_size):
                end = start + args.value_minibatch_size
                mb_inds = b_inds[start:end] # 当前小批量的索引
                # print(f"b_obs[mb_inds].shape: {b_obs[mb_inds].shape}")
                # print(f"b_returns[mb_inds].shape: {b_returns[mb_inds].shape}")

                # newvalue = agent.get_value(b_obs[mb_inds]) # 使用价值网络预测当前状态的价值
                #-------------Addition----------------
                newvalue, newuncertainty = agent.get_value(b_obs[mb_inds])
                #-----------------------------------

                # Value loss
                # 计算价值损失
                newvalue = newvalue.view(-1) # 将输出调整为一维张量
                
                #-------------Addition----------------
                # newuncertainty = newuncertainty.view(-1)
                # assert newvalue.shape[0] == b_returns[mb_inds].shape[0], "Shape mismatch between newvalue and returns"
                #-----------------------------------
                
                if args.clip_vloss: # 是否对价值损失进行裁剪
                    # v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2 # 未裁剪的价值损失
                    # print(f"newvalue.shape: {newvalue.shape}")
                    # print(f"uncertainty.shape: {newuncertainty.shape}")

                    
                    #------------Addition----------------
                    v_loss_unclipped = ((newvalue - b_returns[mb_inds]) ** 2) / (1 + newuncertainty)
                    #------------------------------------
                    
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    # v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2 # 裁剪后的价值损失
                    
                    #------------Addition----------------
                    v_loss_clipped = ((v_clipped - b_returns[mb_inds]) ** 2) / (1 + newuncertainty)
                    #------------------------------------
                    
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped) # 取两者中较大的损失
                    v_loss = 0.5 * v_loss_max.mean() # 最终的价值损失
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = v_loss * args.vf_coef # 加权价值损失

                value_optimizer.zero_grad() # 清除梯度
                loss.backward() # 反向传播
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm) # 裁剪梯度防止爆炸
                value_optimizer.step() # 更新价值网络的参数
            
            if is_warmup: # 如果是预热阶段，跳过策略更新
                continue
            
            policy_optimizer.zero_grad() # 清除策略优化器的梯度
            #update policy
            # 更新策略网络
            for start in range(0, args.batch_size, args.policy_minibatch_size):
                if policy_update_steps % args.gradient_checkpointing_steps == 0: # 定期重置 KL 总和
                    total_approx_kl = 0
                policy_update_steps += 1 # 更新计数
                end = start + args.policy_minibatch_size
                mb_inds = b_inds[start:end] # 当前小批量的索引
                
                # 获取新策略下的 log 概率、熵和价值
                # _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds], is_warmup, return_value = False)
                #----------------Addition----------------
                _, newlogprob, entropy, newvalue, uncertainty = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds], is_warmup, return_value = False)
                #-----------------------------------
                
                logratio = newlogprob - b_logprobs[mb_inds] # 新旧策略 log 概率之差
                ratio = logratio.exp() # 策略概率比
                
                # 计算 KL 散度的估计值
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean() # 旧的 KL 散度
                    approx_kl = ((ratio - 1) - logratio).mean() # 当前 KL 散度
                    total_approx_kl += approx_kl / args.gradient_checkpointing_steps # 累计 KL 散度
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()] # 裁剪比率统计


                # 归一化优势函数
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                # 计算策略损失
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() # 策略损失

                entropy_loss = entropy.mean() # 熵损失
                loss = pg_loss - args.ent_coef * entropy_loss # 总损失（包含熵正则项）
                loss /= args.gradient_checkpointing_steps # 平均化损失
                
                loss.backward() # 反向传播
                
                # 检查是否 KL 爆炸
                if policy_update_steps % args.gradient_checkpointing_steps == 0:
                    if args.target_kl is not None:
                        if total_approx_kl > args.target_kl: # 超过目标 KL
                            policy_optimizer.zero_grad() # 清空梯度
                            kl_explode = True # 停止更新标志
                            policy_update_steps -= args.gradient_checkpointing_steps
                            break                    
                    
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm) # 裁剪梯度
                    policy_optimizer.step() # 更新策略网络的参数
                    policy_optimizer.zero_grad() # 清空优化器梯度


        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        if len(clipfracs) == 0:
            num_clipfracs = 0
        else:
            num_clipfracs = np.mean(clipfracs)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/policy_learning_rate", policy_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/value_learning_rate", value_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/total_approx_kl", total_approx_kl.item(), global_step)
        writer.add_scalar("losses/policy_update_times", policy_update_steps // args.gradient_checkpointing_steps, global_step)
        writer.add_scalar("losses/clipfrac", num_clipfracs, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", global_step, (time.time() - start_time))
        writer.add_scalar("charts/SPS", global_step / (time.time() - start_time), global_step)

        if global_step // 10000 != pre_global_step // 10000: 
            agent.save(global_step // 10000, f"{args.record_path}/{run_name}/{args.save_path}")
        pre_global_step = global_step
    
    agent.save(global_step // 10000 + 1, f"{args.record_path}/{run_name}/{args.save_path}")
    
    envs.close()
    writer.close()