import gym
import numpy as np
import torch
import networkx as nx
from gym import spaces
import random
import copy
import os
from action_env import action_space_constraint

class GraphEnvironment(gym.Env):
    """
    基于离散时间拓扑霍克斯过程的强化学习环境，
    用于威胁情报图谱关系推理补全
    """

    def __init__(self, hawkes_model, entity_data, max_steps=100):
        """
        初始化环境

        参数:
            hawkes_model: 离散时间拓扑霍克斯模型
            entity_data: 实体数据 (字典 {实体ID: 实体信息})
            max_steps: 每个回合的最大步数
        """
        super(GraphEnvironment, self).__init__()

        self.hawkes_model = hawkes_model
        self.entity_data = entity_data
        self.max_steps = max_steps

        # 提取实体和关系
        self.entities = list(entity_data.keys())
        self.entity_num = len(self.entities)

        # 提取实体类型
        self.entity_types = {}
        for entity_id, entity_info in entity_data.items():
            self.entity_types[entity_id] = entity_info.get('EntityType', 'unknown')

        # 构建图结构
        self.graph = nx.DiGraph()
        for entity_id in self.entities:
            self.graph.add_node(entity_id, type=self.entity_types[entity_id])

        # 定义动作空间
        # 动作编码: (source_idx, target_idx, relation_type_idx)
        self.relation_types = list(action_space_constraint.keys())
        self.relation_num = len(self.relation_types)
        self.action_dim = self.entity_num * self.entity_num * self.relation_num
        self.action_space = spaces.Discrete(self.action_dim)

        # 定义状态空间: 图的邻接矩阵展平 + 当前时间
        self.state_dim = self.entity_num * self.entity_num * self.relation_num + 1
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.state_dim,), dtype=np.float32
        )

        # 初始化状态
        self.current_time = 0
        self.current_step = 0
        self.adjacency_tensor = np.zeros((self.entity_num, self.entity_num, self.relation_num))

        # 构建实体到战术的映射
        self.entity_to_tactics = {}
        for entity_id, entity_info in entity_data.items():
            self.entity_to_tactics[entity_id] = entity_info.get('Labels', [])

        # 构建战术事件历史
        self.tactic_event_history = self._build_tactic_event_history()

    def _build_tactic_event_history(self):
        """
        构建战术事件历史

        返回:
            tactic_event_history: 字典 {战术: 时间点列表}
        """
        tactic_event_history = {}

        # 遍历所有实体
        for entity_id, entity_info in self.entity_data.items():
            tactics = entity_info.get('Labels', [])
            times = entity_info.get('Times', [])

            # 将时间转换为整数
            times = [int(t) for t in times]

            # 将时间点添加到对应的战术中
            for tactic, time in zip(tactics, times):
                if tactic not in tactic_event_history:
                    tactic_event_history[tactic] = []
                if time not in tactic_event_history[tactic]:
                    tactic_event_history[tactic].append(time)

        # 对每个战术的时间点进行排序
        for tactic in tactic_event_history:
            tactic_event_history[tactic].sort()

        return tactic_event_history

    def reset(self, partial_graph=None):
        """
        重置环境

        参数:
            partial_graph: 部分图结构，用于图谱补全任务

        返回:
            初始状态
        """
        # 重置图结构
        self.graph = nx.DiGraph()
        for entity_id in self.entities:
            self.graph.add_node(entity_id, type=self.entity_types[entity_id])

        # 重置邻接张量
        self.adjacency_tensor = np.zeros((self.entity_num, self.entity_num, self.relation_num))

        # 如果提供了部分图，则初始化为部分图结构
        if partial_graph is not None:
            for source, targets in partial_graph.items():
                source_idx = self.entities.index(source)
                for target, relation in targets:
                    if target in self.entities and relation in self.relation_types:
                        target_idx = self.entities.index(target)
                        relation_idx = self.relation_types.index(relation)
                        self.graph.add_edge(source, target, relation=relation)
                        self.adjacency_tensor[source_idx, target_idx, relation_idx] = 1

        # 重置时间和步数
        self.current_time = 0
        self.current_step = 0

        return self._get_state()

    def step(self, action):
        """
        执行动作

        参数:
            action: 动作索引

        返回:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 解码动作
        source_idx = action // (self.entity_num * self.relation_num)
        remaining = action % (self.entity_num * self.relation_num)
        target_idx = remaining // self.relation_num
        relation_idx = remaining % self.relation_num

        source_entity = self.entities[source_idx]
        target_entity = self.entities[target_idx]
        relation_type = self.relation_types[relation_idx]

        # 检查动作是否有效
        is_valid = self._is_valid_action(source_entity, target_entity, relation_type)

        # 执行动作: 添加或删除边
        if is_valid:
            if self.adjacency_tensor[source_idx, target_idx, relation_idx] == 0:
                # 添加边
                self.graph.add_edge(source_entity, target_entity, relation=relation_type)
                self.adjacency_tensor[source_idx, target_idx, relation_idx] = 1
            else:
                # 删除边
                if self.graph.has_edge(source_entity, target_entity):
                    self.graph.remove_edge(source_entity, target_entity)
                self.adjacency_tensor[source_idx, target_idx, relation_idx] = 0

        # 更新时间和步数
        self.current_time += 1
        self.current_step += 1

        # 计算奖励
        reward = self._calculate_reward(is_valid)

        # 检查是否结束
        done = self._is_done()

        # 获取下一个状态
        next_state = self._get_state()

        # 返回额外信息
        info = {
            'source_entity': source_entity,
            'target_entity': target_entity,
            'relation_type': relation_type,
            'is_valid': is_valid,
            'action_type': 'add' if self.adjacency_tensor[source_idx, target_idx, relation_idx] == 1 else 'remove'
        }

        return next_state, reward, done, info

    def _is_valid_action(self, source_entity, target_entity, relation_type):
        """
        检查动作是否有效

        参数:
            source_entity: 源实体
            target_entity: 目标实体
            relation_type: 关系类型

        返回:
            is_valid: 动作是否有效
        """
        # 检查自环
        if source_entity == target_entity:
            return False

        # 检查实体类型是否符合关系约束
        source_type = self.entity_types.get(source_entity, 'unknown')
        target_type = self.entity_types.get(target_entity, 'unknown')

        # 获取关系约束
        constraint = action_space_constraint.get(relation_type, {})
        valid_source_types = constraint.get('source_types', [])
        valid_target_types = constraint.get('target_types', [])

        # 检查实体类型是否符合约束
        if source_type not in valid_source_types or target_type not in valid_target_types:
            return False

        return True

    def _get_state(self):
        """
        获取当前状态

        返回:
            state: 当前状态向量
        """
        # 将邻接张量展平并添加当前时间
        state = np.concatenate([
            self.adjacency_tensor.flatten(),
            np.array([self.current_time / self.max_steps])
        ])
        return state

    def _calculate_reward(self, is_valid):
        """
        计算奖励

        参数:
            is_valid: 动作是否有效

        返回:
            reward: 奖励值
        """
        # 如果动作无效，给予负奖励
        if not is_valid:
            return -0.1

        # 1. 图谱完整性奖励
        r1 = self._graph_completeness_reward()

        # 2. 因果一致性奖励
        r2 = self._causal_consistency_reward()

        # 3. 反事实有效性奖励
        r3 = self._counterfactual_validation_reward()

        # 综合奖励
        reward = r1 + r2 + r3
        return reward

    def _graph_completeness_reward(self):
        """
        计算图谱完整性奖励

        返回:
            r1: 图谱完整性奖励
        """
        # 计算图的边数
        edge_count = self.graph.number_of_edges()

        # 计算可能的最大边数
        max_edge_count = self.entity_num * (self.entity_num - 1) * self.relation_num

        # 计算图的完整性
        if max_edge_count == 0:
            return 0

        completeness = edge_count / max_edge_count

        # 使用非线性函数，鼓励适度的边数
        return 0.1 * (1 - abs(completeness - 0.3))

    def _causal_consistency_reward(self):
        """
        计算因果一致性奖励

        返回:
            r2: 因果一致性奖励
        """
        r2 = 0.0
        gamma = 0.2  # 因果强度阈值

        # 遍历所有边
        for source, target, data in self.graph.edges(data=True):
            relation = data.get('relation', '')

            # 获取源实体和目标实体的战术
            source_tactics = self.entity_to_tactics.get(source, [])
            target_tactics = self.entity_to_tactics.get(target, [])

            # 计算战术间的因果效应
            for source_tactic in source_tactics:
                for target_tactic in target_tactics:
                    # 检查模型中是否存在该因果边
                    param_key = f'alpha_{source_tactic}->{target_tactic}'
                    if param_key in self.hawkes_model.params:
                        alpha = self.hawkes_model.params[param_key].item()
                        if alpha > gamma:
                            # 计算瞬时因果效应
                            ice = self._calculate_ice(source_tactic, target_tactic)
                            r2 += ice

        return r2

    def _calculate_ice(self, j, i, delta_t=3):
        """
        计算瞬时因果效应

        参数:
            j: 源战术
            i: 目标战术
            delta_t: 时间窗口大小

        返回:
            ice: 瞬时因果效应
        """
        if f'alpha_{j}->{i}' not in self.hawkes_model.params:
            return 0.0

        alpha = self.hawkes_model.params[f'alpha_{j}->{i}'].item()
        eta = self.hawkes_model.params[f'eta_{j}->{i}'].item()
        events_j = self.tactic_event_history.get(j, [])

        # 筛选时间窗口内的事件
        valid_events = [e for e in events_j if self.current_time - delta_t <= e < self.current_time]

        # 计算时间衰减效应
        effect = 0.0
        for event_time in valid_events:
            effect += alpha * np.exp(-eta * (self.current_time - event_time))

        return effect

    def _counterfactual_validation_reward(self):
        """
        计算反事实验证奖励

        返回:
            r3: 反事实验证奖励
        """
        # 如果图中没有边，返回0
        if self.graph.number_of_edges() == 0:
            return 0

        # 随机选择一条边进行干预
        edges = list(self.graph.edges(data=True))
        if not edges:
            return 0

        source, target, data = random.choice(edges)
        relation = data.get('relation', '')

        # 获取源实体和目标实体的战术
        source_tactics = self.entity_to_tactics.get(source, [])
        target_tactics = self.entity_to_tactics.get(target, [])

        # 如果没有战术，返回0
        if not source_tactics or not target_tactics:
            return 0

        # 随机选择一对战术
        j = random.choice(source_tactics)
        i = random.choice(target_tactics)

        # 计算真实效应
        real_att = self._calculate_att(j, i)

        # 干预后的效应
        intervened_history = self._do_intervention(j)
        pred_att = self._calculate_att(j, i, intervened_history)

        # 反事实奖励: 干预效应的差异越小越好
        return -abs(real_att - pred_att)

    def _calculate_att(self, j, i, custom_history=None):
        """
        计算平均处理效应

        参数:
            j: 源战术
            i: 目标战术
            custom_history: 自定义事件历史

        返回:
            att: 平均处理效应
        """
        history = custom_history if custom_history is not None else self.tactic_event_history
        t_start = 0
        t_end = self.current_time

        # 将历史数据转换为张量
        tensor_history = {k: torch.tensor(v, dtype=torch.float32) for k, v in history.items()}

        # 计算原始强度
        original_intensity = 0.0
        for t in range(t_start, t_end + 1):
            original_intensity += self.hawkes_model.intensity(i, t, tensor_history).item()

        # 创建干预后的事件历史
        intervened_history = copy.deepcopy(history)
        intervened_history[j] = []

        # 将干预后的历史数据转换为张量
        tensor_intervened_history = {k: torch.tensor(v, dtype=torch.float32) for k, v in intervened_history.items()}

        # 计算干预后的强度
        intervened_intensity = 0.0
        for t in range(t_start, t_end + 1):
            intervened_intensity += self.hawkes_model.intensity(i, t, tensor_intervened_history).item()

        # 计算平均处理效应
        if t_end - t_start + 1 > 0:
            att = (original_intensity - intervened_intensity) / (t_end - t_start + 1)
        else:
            att = 0

        return att

    def _do_intervention(self, tactic):
        """
        执行干预

        参数:
            tactic: 要干预的战术

        返回:
            intervened_history: 干预后的事件历史
        """
        intervened_history = copy.deepcopy(self.tactic_event_history)
        intervened_history[tactic] = []
        return intervened_history

    def _is_done(self):
        """
        检查是否结束

        返回:
            done: 是否结束
        """
        # 达到最大步数时结束
        if self.current_step >= self.max_steps:
            return True

        return False

    def get_current_graph(self):
        """
        获取当前图结构

        返回:
            current_graph: 当前图结构 (字典 {源实体: [(目标实体, 关系类型)]})
        """
        current_graph = {}
        for source in self.entities:
            current_graph[source] = []
            for target in self.graph.successors(source):
                relation = self.graph.edges[source, target].get('relation', '')
                current_graph[source].append((target, relation))
        return current_graph
