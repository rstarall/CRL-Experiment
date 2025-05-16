# 离散时间拓扑霍克斯模型预测与强化学习集成方案

## 第一部分：使用离散时间拓扑霍克斯模型预测战术节点因果性

### 1. **因果效应预测公式**
对于给定的战术节点对 $(j \to i)$，定义以下离散时间因果指标：

a. 瞬时因果效应 (Instantaneous Causal Effect, ICE)：
$$
ICE_{j \to i}(t) = \alpha_{j \to i} \cdot \sum_{s_j \in [t-\Delta t, t)} e^{-\eta_{j \to i}(t - s_j)}
$$

b. 累积因果效应 (Cumulative Causal Effect, CCE)：
$$
CCE_{j \to i}([t_1, t_2]) = \sum_{t=t_1}^{t_2} (\lambda_i(t) - \mu_i)
$$

c. 因果显著性检验：
使用似然比检验判断因果关系的显著性：
$$
\Lambda = 2 \left[ \log \mathcal{L}(\text{Full Model}) - \log \mathcal{L}(\text{Model without } j \to i) \right] \sim \chi^2(1)
$$

### 2. **预测代码实现**
```python
import torch
import copy
from scipy.stats import chi2
import numpy as np

class CausalAnalyzer:
    def __init__(self, model):
        self.model = model

    def ice(self, j, i, current_time, event_history, delta_t=5):
        """计算离散时间瞬时因果效应"""
        if f'alpha_{j}->{i}' not in self.model.params:
            return 0.0  # 如果因果边不存在，返回0

        alpha = self.model.params[f'alpha_{j}->{i}'].item()
        eta = self.model.params[f'eta_{j}->{i}'].item()
        events_j = event_history[j]

        # 筛选时间窗口内的事件
        valid_events = [e for e in events_j if current_time - delta_t <= e < current_time]

        # 计算时间衰减效应
        effect = 0.0
        for event_time in valid_events:
            effect += alpha * np.exp(-eta * (current_time - event_time))

        return effect

    def cce(self, i, t_start, t_end, event_history):
        """计算离散时间累积因果效应"""
        if t_start >= t_end:
            return 0.0

        mu_i = self.model.params[f'mu_{i}'].item()
        cumulative_effect = 0.0

        # 对离散时间点求和
        for t in range(t_start, t_end + 1):
            lambda_t = self.model.intensity(i, t, event_history).item()
            cumulative_effect += (lambda_t - mu_i)

        return cumulative_effect

    def causal_significance(self, j, i, event_data):
        """因果显著性检验"""
        # 确保因果边存在
        if f'alpha_{j}->{i}' not in self.model.params:
            return 1.0  # 不存在的边，p值为1

        # 计算全模型对数似然
        full_loglik = self._calculate_log_likelihood(self.model, event_data)

        # 创建移除j->i边的模型
        removed_model = copy.deepcopy(self.model)
        # 将alpha设为0（相当于移除边）
        if f'alpha_{j}->{i}' in removed_model.params:
            removed_model.params[f'alpha_{j}->{i}'].data.copy_(torch.tensor(0.0))

        # 计算移除边后的对数似然
        removed_loglik = self._calculate_log_likelihood(removed_model, event_data)

        # 计算似然比统计量
        lambda_ratio = 2 * (full_loglik - removed_loglik)

        # 计算p值（自由度为1）
        p_value = 1 - chi2.cdf(lambda_ratio, df=1)
        return p_value

    def _calculate_log_likelihood(self, model, event_data):
        """计算模型的对数似然"""
        log_lik = 0.0
        T = max([max(events) if events else 0 for events in event_data.values()]) + 1

        for node in model.nodes:
            events = event_data.get(node, [])

            # 事件项
            for t in events:
                lambda_t = model.intensity(node, t, event_data)
                log_lik += torch.log(lambda_t + 1e-10)

            # 积分项（离散时间求和）
            for t in range(T):
                lambda_t = model.intensity(node, t, event_data)
                log_lik -= lambda_t

        return log_lik.item()

    def att(self, j, i, event_history, t_start=None, t_end=None):
        """计算平均处理效应（反事实干预）"""
        if t_start is None or t_end is None:
            # 默认使用整个时间范围
            all_times = []
            for events in event_history.values():
                all_times.extend(events)
            if not all_times:
                return 0.0
            t_start = 0
            t_end = max(all_times)

        # 计算原始强度
        original_intensity = 0.0
        for t in range(t_start, t_end + 1):
            original_intensity += self.model.intensity(i, t, event_history).item()

        # 创建干预后的事件历史（移除j的所有事件）
        intervened_history = copy.deepcopy(event_history)
        intervened_history[j] = []

        # 计算干预后的强度
        intervened_intensity = 0.0
        for t in range(t_start, t_end + 1):
            intervened_intensity += self.model.intensity(i, t, intervened_history).item()

        # 计算平均处理效应
        att = (original_intensity - intervened_intensity) / (t_end - t_start + 1)
        return att

# 使用示例
def predict_example():
    # 加载训练好的模型
    import os
    from experiment.casual_model.THP.readme.THP_train import DiscreteTopoHawkesModel, casual_model_graph

    model_path = os.path.join("experiment", "casual_model", "THP", "models", "discrete_topo_hawkes_model.npz")
    trained_model = DiscreteTopoHawkesModel.load_model(model_path, casual_model_graph)

    # 创建分析器
    analyzer = CausalAnalyzer(trained_model)

    # 示例事件数据
    events = {
        'TA0043': [1, 3, 5],
        'TA0042': [2, 4, 6],
        'TA0001': [7, 9]
    }

    # 计算TA0043->TA0042在t=7时的瞬时效应
    ice_value = analyzer.ice('TA0043', 'TA0042', current_time=7, event_history=events)
    print(f"瞬时因果效应(TA0043->TA0042): {ice_value:.3f}")

    # 计算TA0042在[0,10]时间段的累积效应
    cce_value = analyzer.cce('TA0042', 0, 10, events)
    print(f"累积因果效应(TA0042): {cce_value:.3f}")

    # 检验TA0043->TA0042的显著性
    p_value = analyzer.causal_significance('TA0043', 'TA0042', events)
    print(f"因果显著性(TA0043->TA0042, p值): {p_value:.4f}")

    # 计算TA0043->TA0042的平均处理效应
    att_value = analyzer.att('TA0043', 'TA0042', events)
    print(f"平均处理效应(TA0043->TA0042): {att_value:.3f}")

if __name__ == "__main__":
    predict_example()
```

## 第二部分：接入强化学习的完整方案

### 1. **强化学习框架设计**
将离散时间拓扑霍克斯模型作为环境模型集成到PPO算法中：

| 组件          | 功能描述                                                                 |
|---------------|--------------------------------------------------------------------------|
| 状态空间  | 当前图谱状态（节点存在性、连接状态、离散事件时间戳）                     |
| 动作空间  | 添加/删除边、修改节点属性                                                |
| 奖励函数  | 基于因果模型的奖励：<br>- 图谱完整性奖励(r1)<br>- 因果一致性奖励(r2)<br>- 反事实有效性奖励(r3) |
| 环境模型  | 使用离散时间拓扑霍克斯模型预测动作后的因果状态变化                       |

### 2. **因果驱动的奖励函数公式**
$$
r(s,a,s') = \underbrace{\text{图完整性}(s')}_{r1} + \underbrace{\sum_{j \to i} \mathbb{I}(\alpha_{j \to i} > \gamma) \cdot \text{ICE}_{j \to i}(t)}_{r2} + \underbrace{\mathbb{E}[\text{ATT}_{\text{pred}} - \text{ATT}_{\text{real}}]}_{r3}
$$

其中：
• $r1$ 使用Jaccard相似度计算补全图谱与真实图谱的相似度
• $r2$ 对符合因果强度阈值（$\gamma$）的边给予奖励
• $r3$ 通过反事实干预前后的效应差异评估动作合理性

### 3. **代码实现**
```python
import gym
import torch
import numpy as np
import networkx as nx
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class CyberEnv(gym.Env):
    def __init__(self, hawkes_model, true_graph):
        super(CyberEnv, self).__init__()
        self.hawkes_model = hawkes_model
        self.true_graph = true_graph
        self.current_graph = self._init_graph()
        # 动作空间: [添加边, 删除边]
        self.action_space = gym.spaces.MultiBinary(self._get_action_dim())
        # 状态空间: 节点存在性 + 边存在性
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self._get_obs_shape())

    def _get_action_dim(self):
        """动作维度为所有可能边的数量"""
        return len(self.true_graph) * (len(self.true_graph)-1)

    def _get_obs_shape(self):
        """状态维度为节点数 + 边数"""
        num_nodes = len(self.true_graph)
        return (num_nodes + num_nodes*(num_nodes-1),)

    def reset(self):
        self.current_graph = self._init_graph()
        return self._get_obs()

    def step(self, action):
        # 执行动作修改图谱
        modified_graph = self._apply_action(action)
        # 使用霍克斯模型预测因果效应
        reward = self._calculate_reward(modified_graph)
        # 判断是否终止
        done = self._is_done(modified_graph)
        return self._get_obs(), reward, done, {}

    def _calculate_reward(self, graph):
        """计算因果驱动的综合奖励"""
        # 1. 图谱完整性奖励
        r1 = jaccard_similarity(graph, self.true_graph)

        # 2. 因果一致性奖励
        analyzer = CausalAnalyzer(self.hawkes_model)
        current_time = self._get_current_time()
        event_history = self._get_events()
        r2 = 0.0
        for j, i in graph.edges():
            ice = analyzer.ice(j, i, current_time, event_history)
            alpha = self.hawkes_model.params[f'alpha_{j}->{i}'].item()
            if alpha > 0.2:  # γ=0.2
                r2 += ice

        # 3. 反事实奖励
        r3 = self._counterfactual_validation(graph)

        return r1 + r2 + r3

    def _counterfactual_validation(self, graph):
        """反事实干预验证"""
        # 随机选择一条边进行干预
        j, i = random.choice(list(graph.edges()))
        # 计算真实效应
        real_att = self.hawkes_model.att(j, i, self._get_events())
        # 干预后的效应
        intervened_events = self._do_intervention(j)
        pred_att = self.hawkes_model.att(j, i, intervened_events)
        return -torch.abs(real_att - pred_att).item()

# 初始化环境
true_graph = {'TA0043': [], 'TA0042': ['TA0043'], ...}  # 真实有向图结构
env = CyberEnv(trained_model, true_graph)
check_env(env)  # 验证环境符合性

# 训练PPO代理
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# 保存模型
model.save("rl_cyber_detection")

# 使用训练好的代理进行预测
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()
```

### 4. **关键集成点**
| 模块          | 集成方法                                                                 |
|---------------|--------------------------------------------------------------------------|
| 状态编码  | 将霍克斯模型的离散时间强度值 $\lambda_i(t)$ 作为状态特征                  |
| 动作执行  | 修改图谱后立即调用霍克斯模型进行因果效应预测                             |
| 奖励计算  | 实时调用`CausalAnalyzer`计算ICE、CCE和反事实差异                         |
| 终止条件  | 当预测的因果效应与真实数据差异超过阈值时终止回合                         |

### 5. **性能优化建议**
• 并行计算：对多个边/节点的因果效应计算进行批处理
• 模型蒸馏：将复杂的霍克斯模型简化为轻量级代理模型用于RL训练
• 课程学习：从简单攻击场景逐步过渡到复杂多阶段攻击

---

## 总结
通过将离散时间拓扑霍克斯过程模型与强化学习深度融合，实现了：
1. 因果感知的动作决策：智能体能理解攻击战术间的动态因果关系
2. 反事实推理能力：评估假设性干预对攻击链路的影响
3. 可解释性保障：奖励机制与因果效应直接挂钩

该方法可广泛应用于APT攻击检测、网络防御策略优化等场景。实际部署时需根据具体业务需求调整奖励权重和状态编码方式。