import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
import random

class RolloutBuffer:
    """
    存储轨迹数据的缓冲区
    """
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def add(self, state, action, reward, next_state, done, logprob=None):
        """添加一个转换"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_terminals.append(done)
        if logprob is not None:
            self.logprobs.append(logprob)

    def clear(self):
        """清空缓冲区"""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    """
    Actor-Critic网络
    """
    def __init__(self, state_dim, action_dim, hidden_units, has_continuous_action_space, action_std_init, device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.has_continuous_action_space = has_continuous_action_space
        self.state_dim = state_dim
        self.action_dim = action_dim

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # Actor网络
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_units),
                nn.Tanh(),
                nn.Linear(hidden_units, hidden_units),
                nn.Tanh(),
                nn.Linear(hidden_units, action_dim),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_units),
                nn.Tanh(),
                nn.Linear(hidden_units, hidden_units),
                nn.Tanh(),
                nn.Linear(hidden_units, action_dim),
                nn.Softmax(dim=-1)
            )

        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1)
        )

    def act(self, state, node_order=None, event_order=None, subset_size=None, is_use_causal_mask=False):
        """
        根据状态选择动作

        参数:
            state: 状态
            node_order: 节点顺序
            event_order: 事件顺序
            subset_size: 子集大小
            is_use_causal_mask: 是否使用因果掩码

        返回:
            action: 选择的动作
            action_logprob: 动作的对数概率
            action_probs: 动作概率分布
        """
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)

            # 添加状态掩码 - 防止自环
            state_shape = state.shape[0] if len(state.shape) > 1 else 1
            node_num = int(np.sqrt(self.action_dim))
            mask = torch.ones(self.action_dim).to(self.device)

            # 将对角线元素（自环）设为0
            for i in range(node_num):
                mask[i * node_num + i] = 0

            # 使用克隆避免就地操作
            masked_probs = action_probs * mask
            action_probs = masked_probs / masked_probs.sum()

            # 添加因果掩码
            if is_use_causal_mask and node_order is not None and event_order is not None and subset_size is not None:
                node_num = len(node_order)
                event_num = len(event_order)

                # 因果掩码
                node_probs, _ = torch.sort(torch.zeros(node_num).uniform_(0, 1), descending=True)
                event_probs, _ = torch.sort(torch.zeros(event_num).uniform_(0.2, 1), descending=True)

                causal_mask = torch.zeros((node_num, node_num)).uniform_(0, 1).to(self.device)
                for i in range(node_num):
                    n = node_order[i]
                    if i < (subset_size*1.5):
                        causal_mask[n,:] = node_probs[0]
                    else:
                        causal_mask[n,:] = node_probs[-1]

                for j in range(event_num):
                    v = event_order[j]
                    if j < subset_size:
                        causal_mask[:,v] = causal_mask[:,v] * event_probs[j]
                    else:
                        causal_mask[:,v] = causal_mask[:,v] * event_probs[-1]

                causal_mask = causal_mask.flatten()

                # 使用克隆避免就地操作
                new_action_probs = action_probs * causal_mask
                if torch.max(new_action_probs) > 0:
                    action_probs = new_action_probs / new_action_probs.sum()

            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach(), action_probs

    def evaluate(self, state, action, node_order=None, event_order=None, subset_size=None):
        """
        评估动作

        参数:
            state: 状态
            action: 动作
            node_order: 节点顺序
            event_order: 事件顺序
            subset_size: 子集大小

        返回:
            action_logprobs: 动作的对数概率
            state_values: 状态值
            dist_entropy: 分布熵
        """
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)

            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)

            # 添加状态掩码 - 防止自环
            batch_size = state.shape[0]
            node_num = int(np.sqrt(self.action_dim))
            mask = torch.ones(batch_size, self.action_dim).to(self.device)

            # 将对角线元素（自环）设为0
            for b in range(batch_size):
                for i in range(node_num):
                    mask[b, i * node_num + i] = 0

            # 使用克隆避免就地操作
            masked_probs = action_probs * mask
            normalized_probs = masked_probs.clone()
            for b in range(batch_size):
                if masked_probs[b].sum() > 0:
                    normalized_probs[b] = masked_probs[b] / masked_probs[b].sum()
            action_probs = normalized_probs

            # 添加因果掩码
            if node_order is not None and event_order is not None and subset_size is not None:
                node_num = len(node_order)
                event_num = len(event_order)

                # 因果掩码
                node_probs, _ = torch.sort(torch.zeros(node_num).uniform_(0, 1), descending=True)
                event_probs, _ = torch.sort(torch.zeros(event_num).uniform_(0.2, 1), descending=True)

                causal_mask = torch.zeros((node_num, node_num)).uniform_(0, 1).to(self.device)
                for i in range(node_num):
                    n = node_order[i]
                    if i < (subset_size*1.5):
                        causal_mask[n,:] = node_probs[0]
                    else:
                        causal_mask[n,:] = node_probs[-1]

                for j in range(event_num):
                    v = event_order[j]
                    if j < subset_size:
                        causal_mask[:,v] = causal_mask[:,v] * event_probs[j]
                    else:
                        causal_mask[:,v] = causal_mask[:,v] * event_probs[-1]

                causal_mask = causal_mask.flatten()

                # 使用克隆避免就地操作
                causal_action_probs = action_probs.clone()
                for b in range(batch_size):
                    new_action_probs = action_probs[b] * causal_mask
                    if torch.max(new_action_probs) > 0:
                        causal_action_probs[b] = new_action_probs / new_action_probs.sum()
                action_probs = causal_action_probs

            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    """
    PPO算法
    """
    def __init__(self, state_dim, action_dim, config, device):
        self.name = "PPO"
        self.device = device

        self.has_continuous_action_space = config["has_continuous_action_space"]
        self.hidden_units = config["hidden_units"]
        self.gamma = config["gamma"]
        self.eps_clip = config["eps_clip"]
        self.K_epochs = config["K_epochs"]
        self.lr_actor = config["lr_actor"]
        self.lr_critic = config["lr_critic"]
        self.normalize_advantage = config["normalize_advantage"]
        self.max_grad_norm = config["max_grad_norm"]
        self.batch_size = config["batch_size"]
        self.action_std = 0.6

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, self.hidden_units, self.has_continuous_action_space,
                                 self.action_std, device).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, self.hidden_units, self.has_continuous_action_space,
                                     self.action_std, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, node_order=None, event_order=None, subset_size=None):
        """
        选择动作

        参数:
            state: 状态
            node_order: 节点顺序
            event_order: 事件顺序
            subset_size: 子集大小

        返回:
            action: 选择的动作
        """
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, _ = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, action_probs = self.policy_old.act(
                    state, node_order, event_order, subset_size, True
                )

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item(), action_probs

    def update(self, node_order=None, event_order=None, subset_size=None):
        """
        更新策略 - 完整的PPO更新逻辑

        参数:
            node_order: 节点顺序
            event_order: 事件顺序
            subset_size: 子集大小

        返回:
            loss_mean: 平均总损失
            pg_loss_mean: 平均策略梯度损失
            value_loss_mean: 平均价值损失
            entropy_loss_mean: 平均熵损失
        """
        # 如果缓冲区中的数据不足，直接返回
        if len(self.buffer.states) < self.batch_size:
            self.buffer.clear()
            return 0, 0, 0, 0

        # 记录损失
        loss_list = []
        pg_losses, value_losses, entropy_losses = [], [], []

        # 对策略进行K轮优化
        for _ in range(self.K_epochs):
            # 随机采样batch_size个样本
            batches = random.sample(range(len(self.buffer.states)), k=min(self.batch_size, len(self.buffer.states)))

            # 蒙特卡洛估计回报
            rewards = []
            discounted_reward = 0
            bc_rewards = [self.buffer.rewards[b] for b in batches]
            bc_is_terminals = [self.buffer.is_terminals[b] for b in batches]

            # 从后向前计算折扣回报
            for reward, is_terminal in zip(reversed(bc_rewards), reversed(bc_is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            # 将列表转换为张量
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

            # 获取状态、动作和对数概率
            bc_states = [self.buffer.states[b] for b in batches]
            bc_actions = [self.buffer.actions[b] for b in batches]
            bc_logprobs = [self.buffer.logprobs[b] for b in batches]

            old_states = torch.squeeze(torch.stack(bc_states, dim=0)).detach().to(self.device)
            old_actions = torch.squeeze(torch.stack(bc_actions, dim=0)).detach().to(self.device)
            old_logprobs = torch.squeeze(torch.stack(bc_logprobs, dim=0)).detach().to(self.device)

            # 评估旧动作和值
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, node_order, event_order, subset_size)

            # 匹配state_values张量维度与rewards张量
            state_values = torch.squeeze(state_values)

            # 计算比率 (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算优势函数
            advantages = rewards - state_values.detach()

            # 如果启用了优势标准化且样本数大于1，则进行标准化
            if self.normalize_advantage and len(advantages) > 1:
                # 使用克隆避免就地操作
                advantages_mean = advantages.mean()
                advantages_std = advantages.std() + 1e-8
                advantages = (advantages - advantages_mean) / advantages_std

            # 计算PPO的替代损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # 计算策略损失、价值损失和熵损失
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.MseLoss(state_values, rewards)
            entropy_loss = -torch.mean(dist_entropy)

            # 记录各种损失
            pg_losses.append(-policy_loss.item())  # 取负是因为我们在下面计算总损失时使用了负号
            value_losses.append(value_loss.item())
            entropy_losses.append(-entropy_loss.item())  # 取负是因为我们在下面计算总损失时使用了负号

            # 计算PPO的最终裁剪目标损失
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            loss_list.append(loss.item())

            # 梯度步骤
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)

            self.optimizer.step()

        # 将新权重复制到旧策略中
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空缓冲区
        self.buffer.clear()

        # 计算平均损失
        pg_loss_mean = np.mean(pg_losses)
        value_loss_mean = np.mean(value_losses)
        entropy_loss_mean = np.mean(entropy_losses)
        loss_mean = np.mean(loss_list)

        return loss_mean, pg_loss_mean, value_loss_mean, entropy_loss_mean

    def save(self, checkpoint_path):
        """保存模型"""
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path, strict=True):
        """
        加载模型

        参数:
            checkpoint_path: 检查点路径
            strict: 是否严格加载模型参数，如果为False，则允许加载部分参数
        """
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.policy_old.load_state_dict(state_dict, strict=strict)
        self.policy.load_state_dict(state_dict, strict=strict)

    def update_learning_rate(self, decay_factor=0.95):
        """
        更新学习率

        参数:
            decay_factor: 学习率衰减因子
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_factor

        # 获取当前学习率
        current_lr_actor = self.optimizer.param_groups[0]['lr']
        current_lr_critic = self.optimizer.param_groups[1]['lr']

        return current_lr_actor, current_lr_critic
