import os
import argparse
import numpy as np
import torch
import random
import networkx as nx
import uuid
# 尝试导入matplotlib，如果不可用则使用替代方案
try:
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False
    print("警告: matplotlib未安装，将不会生成可视化图表。")
    print("如需完整功能，请安装matplotlib: pip install matplotlib")
import pickle
import time
import json
import sys
import shutil

# 尝试导入TensorBoard，如果不可用则使用替代方案
try:
    from torch.utils.tensorboard import SummaryWriter
    has_tensorboard = True
except ImportError:
    has_tensorboard = False
    print("警告: TensorBoard未安装，将不会记录详细的训练日志。")
    print("如需完整功能，请安装TensorBoard: pip install tensorboard")

    # 创建一个虚拟的SummaryWriter类
    class DummySummaryWriter:
        def __init__(self, log_dir=None):
            self.log_dir = log_dir
            print(f"日志将保存在: {log_dir}")

        def add_scalar(self, tag, scalar_value, global_step=None):
            pass

        def close(self):
            pass

    SummaryWriter = DummySummaryWriter

import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # experiment目录
sys.path.append(project_root)

from rl_moldel.graph_env import FixedDimGraphEnvironment
from rl_moldel.data_loader import load_all_entities
from rl_moldel.ppo import PPO
from casual_model.THP.train import DiscreteTopoHawkesModel, casual_model_graph

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def plot_curves(data, plot_type, save_path, episode_num=None):
    """
    绘制并保存各种曲线图

    参数:
        data: 要绘制的数据，可以是单个列表或字典
        plot_type: 图表类型，'reward'或'loss'
        save_path: 保存图表的路径
        episode_num: 当前回合数，用于文件名（可选）

    返回:
        保存的图表文件路径
    """
    print(f"开始绘制图表: 类型={plot_type}, 保存路径={save_path}, 回合数={episode_num}")

    if not has_matplotlib:
        print("matplotlib未安装，无法生成图表")
        return None


    if plot_type == 'reward':
        # 绘制奖励曲线
        plt.figure(figsize=(10, 5))
        plt.plot(data)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.tight_layout()

        # 确定保存路径
        if episode_num is not None:
            file_path = os.path.join(save_path, f'reward_curve_episode_{episode_num}.png')
        else:
            file_path = os.path.join(save_path, 'reward_curve.png')

        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            plt.savefig(file_path)
            plt.close()
            print(f"奖励曲线图已成功保存到: {file_path}")
            return file_path
        except Exception as e:
            print(f"保存奖励曲线图时出错: {e}")
            plt.close()
            return None

    elif plot_type == 'loss':
        # 绘制损失曲线
        plt.figure(figsize=(15, 10))

        # 总损失
        plt.subplot(2, 2, 1)
        plt.plot(data['total_losses'])
        plt.title('Total Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')

        # 策略损失
        plt.subplot(2, 2, 2)
        plt.plot(data['policy_losses'])
        plt.title('Policy Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')

        # 价值损失
        plt.subplot(2, 2, 3)
        plt.plot(data['value_losses'])
        plt.title('Value Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')

        # 熵损失
        plt.subplot(2, 2, 4)
        plt.plot(data['entropy_losses'])
        plt.title('Entropy Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')

        plt.tight_layout()

        # 确定保存路径
        if episode_num is not None:
            file_path = os.path.join(save_path, f'loss_curves_episode_{episode_num}.png')
        else:
            file_path = os.path.join(save_path, 'all_loss_curves.png')

        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            plt.savefig(file_path)
            plt.close()
            print(f"损失曲线图已成功保存到: {file_path}")
            return file_path
        except Exception as e:
            print(f"保存损失曲线图时出错: {e}")
            plt.close()
            return None

    else:
        print(f"不支持的图表类型: {plot_type}")
        return None

def main():
    parser = argparse.ArgumentParser(description="固定图谱PPO训练")
    # 训练参数
    parser.add_argument('--cuda', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--max_episodes', type=int, default=2000, help='最大训练回合数')
    parser.add_argument('--max_ep_len', type=int, default=64, help='每个回合的最大步数')
    parser.add_argument('--output_dir', type=str, default='rl_moldel/output', help='输出目录')
    parser.add_argument('--thp_model_path', type=str, default='casual_model/THP/models/discrete_topo_hawkes_model.npz', help='THP模型路径')
    parser.add_argument('--max_entity_num', type=int, default=64, help='固定实体数量')

    # 超参数调整
    parser.add_argument('--lr_actor', type=float, default=0.003, help='Actor网络学习率')
    parser.add_argument('--lr_critic', type=float, default=0.003, help='Critic网络学习率')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--hidden_units', type=int, default=128, help='隐藏层单元数')
    parser.add_argument('--k_epochs', type=int, default=10, help='每次更新的epoch数')

    # 学习率调度
    parser.add_argument('--lr_decay', type=float, default=0.95, help='学习率衰减因子')
    parser.add_argument('--lr_decay_freq', type=int, default=200, help='学习率衰减频率（回合数）')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='最小学习率')

    # 检查点保存
    parser.add_argument('--save_freq', type=int, default=10, help='保存检查点的频率（回合数）')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 生成唯一ID
    run_id = str(uuid.uuid4())[:8]
    print(f"本次训练的唯一ID: {run_id}")

    # 设置设备
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.cuda}" if use_cuda else "cpu")
    print(f"使用设备: {device}")

    # 打印训练配置
    print("\n====== 训练配置 ======")
    print(f"训练ID: {run_id}")
    print(f"最大训练回合数: {args.max_episodes}")
    print(f"每回合最大步数: {args.max_ep_len}")
    print(f"固定实体数量: {args.max_entity_num}")
    print(f"Actor学习率: {args.lr_actor}")
    print(f"Critic学习率: {args.lr_critic}")
    print(f"批量大小: {args.batch_size}")
    print(f"隐藏层单元数: {args.hidden_units}")
    print(f"每次更新的epoch数: {args.k_epochs}")
    print(f"学习率衰减因子: {args.lr_decay}")
    print(f"学习率衰减频率: {args.lr_decay_freq}回合")
    print(f"最小学习率: {args.min_lr}")
    print(f"检查点保存频率: {args.save_freq}回合")
    print(f"使用GPU: {'是' if use_cuda else '否'}")
    print("======================")

    # 创建输出目录
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出将保存在: {output_dir}")

    # 保存训练配置
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"训练配置已保存到: {config_path}")

    # 创建TensorBoard日志
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # 加载所有实体数据，确保达到目标实体数量
    print(f"加载所有实体数据，目标实体数量: {args.max_entity_num}...")
    all_entities, entity_relations = load_all_entities(target_entity_count=args.max_entity_num)

    # 验证实体数量
    if len(all_entities) < args.max_entity_num:
        print(f"警告: 实体数量({len(all_entities)})小于目标数量({args.max_entity_num})，将使用所有可用实体")
        # 更新max_entity_num为实际实体数量
        args.max_entity_num = len(all_entities)

    print(f"成功加载了 {len(all_entities)} 个实体，将使用 {args.max_entity_num} 个实体进行训练")

    # 保存实体和关系数据
    entity_data_path = os.path.join(output_dir, "entity_data.pkl")
    with open(entity_data_path, 'wb') as f:
        pickle.dump(all_entities, f)

    relation_data_path = os.path.join(output_dir, "relation_data.pkl")
    with open(relation_data_path, 'wb') as f:
        pickle.dump(entity_relations, f)

    print(f"实体数据已保存到: {entity_data_path}")
    print(f"关系数据已保存到: {relation_data_path}")

    # 加载THP模型
    # 确保THP模型路径是绝对路径
    thp_model_path = os.path.abspath(args.thp_model_path)
    print(f"加载THP模型: {thp_model_path}")

    # 检查文件是否存在
    if not os.path.exists(thp_model_path):
        print(f"错误: THP模型文件不存在: {thp_model_path}")
        print("尝试在当前目录下查找...")
        # 尝试在当前目录下查找
        base_name = os.path.basename(args.thp_model_path)
        alt_path = os.path.join(os.path.dirname(current_dir), "casual_model", "THP", "models", base_name)
        print(f"尝试备用路径: {alt_path}")
        if os.path.exists(alt_path):
            thp_model_path = alt_path
            print(f"找到模型文件: {thp_model_path}")
        else:
            print(f"错误: 备用路径也不存在: {alt_path}")
            return

    # 复制THP模型到输出目录
    thp_model_copy_path = os.path.join(output_dir, os.path.basename(thp_model_path))
    shutil.copy2(thp_model_path, thp_model_copy_path)
    print(f"THP模型已复制到: {thp_model_copy_path}")

    hawkes_model = DiscreteTopoHawkesModel.load_model(thp_model_path, casual_model_graph)

    # 创建环境
    print(f"创建固定图谱环境({args.max_entity_num}个实体)...")
    env = FixedDimGraphEnvironment(hawkes_model, all_entities, entity_relations=entity_relations, max_entity_num=args.max_entity_num, max_steps=args.max_ep_len)

    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")

    # PPO超参数
    ppo_config = {
        "gamma": 0.99,  # 折扣因子
        "lr_actor": args.lr_actor,  # Actor网络学习率
        "lr_critic": args.lr_critic,  # Critic网络学习率
        "eps_clip": 0.2,  # PPO裁剪参数
        "K_epochs": args.k_epochs,  # 每次更新的epoch数
        "batch_size": args.batch_size,  # 批量大小
        "hidden_units": args.hidden_units,  # 隐藏层单元数
        "has_continuous_action_space": False,  # 是否为连续动作空间
        "max_grad_norm": 0.5,  # 梯度裁剪参数
        "normalize_advantage": True  # 启用优势标准化
    }

    # 保存PPO配置
    ppo_config_path = os.path.join(output_dir, "ppo_config.json")
    with open(ppo_config_path, 'w') as f:
        json.dump(ppo_config, f, indent=4)
    print(f"PPO配置已保存到: {ppo_config_path}")

    # 创建PPO代理
    print("创建PPO代理...")
    agent = PPO(state_dim, action_dim, ppo_config, device)

    # 如果提供了恢复训练的检查点路径，则加载模型
    start_episode = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"从检查点恢复训练: {args.resume}")
            agent.load(args.resume)
            # 尝试从检查点文件名中提取回合数
            try:
                checkpoint_name = os.path.basename(args.resume)
                if 'episode_' in checkpoint_name:
                    start_episode = int(checkpoint_name.split('episode_')[1].split('.')[0])
                    print(f"从第 {start_episode} 回合继续训练")
            except:
                print("无法从检查点文件名中提取回合数，从头开始训练")
        else:
            print(f"警告: 检查点文件 {args.resume} 不存在，从头开始训练")

    # 训练循环
    print("开始训练...")
    time_step = 0
    i_episode = start_episode

    # 记录奖励和损失
    episode_rewards = []

    # 记录各种损失
    total_losses = []
    policy_losses = []
    value_losses = []
    entropy_losses = []

    # 记录每个回合的平均损失
    episode_total_losses = []
    episode_policy_losses = []
    episode_value_losses = []
    episode_entropy_losses = []

    # 创建检查点目录
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 创建损失数据目录
    loss_dir = os.path.join(output_dir, "losses")
    os.makedirs(loss_dir, exist_ok=True)

    # 训练PPO
    while i_episode < args.max_episodes:
        # 重置环境，但不重新采样实体（保持固定图谱）
        state = env.reset(resample=False)
        current_ep_reward = 0

        # 记录当前图谱的信息
        print(f"回合 {i_episode}: 使用固定的 {env.entity_num} 个实体，状态维度: {state.shape[0]}")

        for t in range(1, args.max_ep_len + 1):
            # 选择动作
            action, _ = agent.select_action(state)

            # 与环境交互
            next_state, reward, done, _ = env.step(action)

            # 注意：select_action已经将state、action和logprob添加到buffer中
            # 现在我们只需要添加reward和is_terminal
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            state = next_state
            current_ep_reward += reward
            time_step += 1

            # 更新PPO
            if len(agent.buffer.states) >= ppo_config["batch_size"]:
                loss, pg_loss, value_loss, entropy_loss = agent.update()

                # 记录每次更新的损失
                total_losses.append(loss)
                policy_losses.append(pg_loss)
                value_losses.append(value_loss)
                entropy_losses.append(entropy_loss)

                # 记录损失到TensorBoard
                writer.add_scalar('Loss/total', loss, time_step)
                writer.add_scalar('Loss/policy', pg_loss, time_step)
                writer.add_scalar('Loss/value', value_loss, time_step)
                writer.add_scalar('Loss/entropy', entropy_loss, time_step)

                # 打印损失信息
                if time_step % 100 == 0:
                    print(f"Step: {time_step}, Loss: {loss:.4f}, Policy Loss: {pg_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy Loss: {entropy_loss:.4f}")

            if done:
                break

        # 记录回合奖励
        episode_rewards.append(current_ep_reward)
        writer.add_scalar('Reward/episode', current_ep_reward, i_episode)

        # 计算并记录本回合的平均损失
        if len(total_losses) > 0:
            avg_total_loss = np.mean(total_losses)
            avg_policy_loss = np.mean(policy_losses)
            avg_value_loss = np.mean(value_losses)
            avg_entropy_loss = np.mean(entropy_losses)

            episode_total_losses.append(avg_total_loss)
            episode_policy_losses.append(avg_policy_loss)
            episode_value_losses.append(avg_value_loss)
            episode_entropy_losses.append(avg_entropy_loss)

            # 记录到TensorBoard
            writer.add_scalar('EpisodeLoss/total', avg_total_loss, i_episode)
            writer.add_scalar('EpisodeLoss/policy', avg_policy_loss, i_episode)
            writer.add_scalar('EpisodeLoss/value', avg_value_loss, i_episode)
            writer.add_scalar('EpisodeLoss/entropy', avg_entropy_loss, i_episode)

            # 清空损失列表，准备记录下一回合
            total_losses.clear()
            policy_losses.clear()
            value_losses.clear()
            entropy_losses.clear()
        else:
            # 如果本回合没有更新，记录零损失
            episode_total_losses.append(0.0)
            episode_policy_losses.append(0.0)
            episode_value_losses.append(0.0)
            episode_entropy_losses.append(0.0)

        # 打印训练信息
        if i_episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])

            # 如果有损失数据，也打印损失信息
            if len(episode_total_losses) >= 10:
                avg_total_loss = np.mean(episode_total_losses[-10:])
                avg_policy_loss = np.mean(episode_policy_losses[-10:])
                avg_value_loss = np.mean(episode_value_losses[-10:])
                avg_entropy_loss = np.mean(episode_entropy_losses[-10:])

                print(f"Episode: {i_episode}, Reward: {current_ep_reward:.3f}, Avg Reward: {avg_reward:.3f}")
                print(f"  Avg Losses - Total: {avg_total_loss:.4f}, Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}, Entropy: {avg_entropy_loss:.4f}")
            else:
                print(f"Episode: {i_episode}, Reward: {current_ep_reward:.3f}, Avg Reward: {avg_reward:.3f}")

        # 定期保存检查点
        if i_episode % args.save_freq == 0 and i_episode > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_episode_{i_episode}.pth")
            agent.save(checkpoint_path)
            print(f"检查点已保存到 {checkpoint_path}")

            # 保存当前的奖励数据
            reward_data_path = os.path.join(checkpoint_dir, f"rewards_episode_{i_episode}.pkl")
            with open(reward_data_path, 'wb') as f:
                pickle.dump(episode_rewards, f)

            # 保存损失数据
            loss_data = {
                'total_losses': episode_total_losses,
                'policy_losses': episode_policy_losses,
                'value_losses': episode_value_losses,
                'entropy_losses': episode_entropy_losses
            }
            loss_data_path = os.path.join(loss_dir, f"losses_episode_{i_episode}.pkl")
            with open(loss_data_path, 'wb') as f:
                pickle.dump(loss_data, f)
            print(f"损失数据已保存到 {loss_data_path}")

            # 绘制并保存损失曲线
            loss_data = {
                'total_losses': episode_total_losses,
                'policy_losses': episode_policy_losses,
                'value_losses': episode_value_losses,
                'entropy_losses': episode_entropy_losses
            }
            loss_curve_path = plot_curves(loss_data, 'loss', loss_dir, i_episode)
            if loss_curve_path:
                print(f"损失曲线已保存到 {loss_curve_path}")

            # 绘制并保存奖励曲线
            reward_curve_path = plot_curves(episode_rewards, 'reward', checkpoint_dir, i_episode)
            if reward_curve_path:
                print(f"奖励曲线已保存到 {reward_curve_path}")

        # 学习率调度
        if i_episode % args.lr_decay_freq == 0 and i_episode > 0:
            current_lr_actor, current_lr_critic = agent.update_learning_rate(args.lr_decay)

            # 确保学习率不低于最小值
            if current_lr_actor < args.min_lr:
                for param_group in agent.optimizer.param_groups:
                    if param_group['lr'] < args.min_lr:
                        param_group['lr'] = args.min_lr
                current_lr_actor, current_lr_critic = agent.optimizer.param_groups[0]['lr'], agent.optimizer.param_groups[1]['lr']

            print(f"学习率已更新 - Actor: {current_lr_actor:.6f}, Critic: {current_lr_critic:.6f}")

        i_episode += 1

    # 保存最终模型
    model_path = os.path.join(output_dir, "final_model.pth")
    agent.save(model_path)
    print(f"最终模型已保存到 {model_path}")

    # 保存图谱环境状态
    env_state = {
        'entity_data': env.entity_data,
        'entity_types': env.entity_types,
        'entity_to_tactics': env.entity_to_tactics,
        'entities': env.entities,
        'relation_types': env.relation_types,
        'graph': nx.node_link_data(env.graph)  # 将图转换为可序列化的格式
    }
    env_state_path = os.path.join(output_dir, "env_state.pkl")
    with open(env_state_path, 'wb') as f:
        pickle.dump(env_state, f)
    print(f"环境状态已保存到: {env_state_path}")

    # 绘制奖励曲线
    reward_curve_path = plot_curves(episode_rewards, 'reward', output_dir)
    if reward_curve_path:
        print(f"奖励曲线已保存到: {reward_curve_path}")
    else:
        print("matplotlib未安装，无法生成奖励曲线图")

    # 保存奖励数据
    reward_data_path = os.path.join(output_dir, 'episode_rewards.pkl')
    with open(reward_data_path, 'wb') as f:
        pickle.dump(episode_rewards, f)
    print(f"奖励数据已保存到: {reward_data_path}")

    # 保存完整的损失数据
    loss_data = {
        'total_losses': episode_total_losses,
        'policy_losses': episode_policy_losses,
        'value_losses': episode_value_losses,
        'entropy_losses': episode_entropy_losses
    }
    loss_data_path = os.path.join(output_dir, 'all_losses.pkl')
    with open(loss_data_path, 'wb') as f:
        pickle.dump(loss_data, f)
    print(f"完整损失数据已保存到: {loss_data_path}")

    # 绘制并保存完整的损失曲线
    loss_data = {
        'total_losses': episode_total_losses,
        'policy_losses': episode_policy_losses,
        'value_losses': episode_value_losses,
        'entropy_losses': episode_entropy_losses
    }
    loss_curve_path = plot_curves(loss_data, 'loss', output_dir)
    if loss_curve_path:
        print(f"完整损失曲线已保存到: {loss_curve_path}")

    # 创建训练完成标记文件
    with open(os.path.join(output_dir, "TRAINING_COMPLETE"), 'w') as f:
        f.write(f"训练完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总回合数: {i_episode}\n")
        f.write(f"总步数: {time_step}\n")
        f.write(f"最终奖励: {current_ep_reward}\n")
        f.write(f"平均奖励(最后10回合): {np.mean(episode_rewards[-10:])}\n")

    # 关闭TensorBoard写入器
    writer.close()

    print(f"训练完成! 所有输出已保存到: {output_dir}")

if __name__ == "__main__":
    main()
