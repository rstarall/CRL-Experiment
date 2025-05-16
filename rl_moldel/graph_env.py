import os
import numpy as np
import torch
import networkx as nx
import random
import copy
import gym
from gym import spaces
from rl_moldel.action_env import action_space_constraint

class FixedDimGraphEnvironment(gym.Env):
    """
    固定维度的图谱环境，使用固定数量的实体构建图谱
    """

    def __init__(self, hawkes_model, all_entities, entity_relations=None, max_entity_num=64, max_steps=100):
        """
        初始化环境

        参数:
            hawkes_model: 离散时间拓扑霍克斯模型
            all_entities: 所有实体数据 (字典 {实体ID: 实体信息})
            entity_relations: 实体间的关系 (字典 {(源实体ID, 目标实体ID): 关系类型列表})
            max_entity_num: 固定的实体数量，用于固定维度
            max_steps: 每个回合的最大步数
        """
        super(FixedDimGraphEnvironment, self).__init__()

        self.hawkes_model = hawkes_model
        self.all_entities = all_entities
        self.all_entity_relations = entity_relations or {}

        # 设置最大实体数量
        self.max_entity_num = max_entity_num

        # 确保实体数量足够
        if len(all_entities) < self.max_entity_num:
            print(f"警告: 实体数量({len(all_entities)})小于指定的固定维度({self.max_entity_num})，将使用所有可用实体")
            # 使用所有可用实体
            self.max_entity_num = len(all_entities)
        self.max_steps = max_steps

        # 定义关系类型
        self.relation_types = list(action_space_constraint.keys())
        self.relation_num = len(self.relation_types)

        # 随机选择固定数量的实体
        self.sample_entities()

        # 构建参考图谱
        self.reference_graph = self._build_reference_graph()

        # 定义动作空间
        # 动作编码: (source_idx, target_idx, relation_type_idx)
        self.action_dim = self.max_entity_num * self.max_entity_num * self.relation_num
        self.action_space = spaces.Discrete(self.action_dim)

        # 定义状态空间: 图的邻接矩阵展平 + 当前时间
        self.state_dim = self.max_entity_num * self.max_entity_num * self.relation_num + 1
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.state_dim,), dtype=np.float32
        )

        print(f"创建了固定维度图环境: {self.max_entity_num}个实体, 状态维度={self.state_dim}, 动作维度={self.action_dim}")

        # 初始化状态
        self.current_time = 0
        self.current_step = 0
        self.adjacency_tensor = np.zeros((self.max_entity_num, self.max_entity_num, self.relation_num))

        # 构建战术事件历史
        self.tactic_event_history = self._build_tactic_event_history()

    def sample_entities(self):
        """
        从所有实体中随机抽样固定数量的实体
        """
        # 获取所有实体ID
        all_entity_ids = list(self.all_entities.keys())

        # 确保我们有足够的实体
        if len(all_entity_ids) < self.max_entity_num:
            raise ValueError(f"实体数量({len(all_entity_ids)})小于指定的固定维度({self.max_entity_num})，请先生成足够的实体")

        # 随机抽样固定数量的实体
        self.entities = random.sample(all_entity_ids, self.max_entity_num)

        # 确保实体数量正确
        self.entity_num = len(self.entities)
        assert self.entity_num == self.max_entity_num, f"实体数量({self.entity_num})不等于指定的固定维度({self.max_entity_num})"

        # 提取实体数据
        self.entity_data = {entity_id: self.all_entities[entity_id] for entity_id in self.entities}

        # 提取实体类型
        self.entity_types = {}
        for entity_id, entity_info in self.entity_data.items():
            self.entity_types[entity_id] = entity_info.get('EntityType', 'unknown')

        # 构建实体到战术的映射
        self.entity_to_tactics = {}
        for entity_id, entity_info in self.entity_data.items():
            self.entity_to_tactics[entity_id] = entity_info.get('Labels', [])

        # 构建图结构
        self.graph = nx.DiGraph()
        for entity_id in self.entities:
            self.graph.add_node(entity_id, type=self.entity_types[entity_id])

        print(f"采样了 {self.entity_num} 个实体，实体类型分布: {self._get_entity_type_distribution()}")

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

    def _build_reference_graph(self):
        """
        构建参考图谱，用于评估预测图谱的准确性

        返回:
            reference_graph: 参考图谱 (NetworkX DiGraph)
        """
        # 创建参考图谱
        reference_graph = nx.DiGraph()

        # 添加节点
        for entity_id in self.entities:
            reference_graph.add_node(entity_id, type=self.entity_types[entity_id])

        # 添加边
        for (source_id, target_id), relation_types in self.all_entity_relations.items():
            if source_id in self.entities and target_id in self.entities:
                for relation_type in relation_types:
                    if relation_type in self.relation_types:
                        reference_graph.add_edge(source_id, target_id, relation=relation_type)

        print(f"构建了参考图谱，包含 {reference_graph.number_of_nodes()} 个节点和 {reference_graph.number_of_edges()} 条边")
        return reference_graph

    def reset(self, resample=True):
        """
        重置环境

        参数:
            resample: 是否重新采样实体，如果为False，则保持当前实体不变

        返回:
            初始状态
        """
        # 如果需要重新采样实体
        if resample:
            self.sample_entities()
            # 重新构建参考图谱
            self.reference_graph = self._build_reference_graph()
            # 重新构建战术事件历史
            self.tactic_event_history = self._build_tactic_event_history()
            print(f"已重新采样 {self.entity_num} 个实体")
        else:
            print(f"保持固定的 {self.entity_num} 个实体")

        # 重置图结构
        self.graph = nx.DiGraph()
        for entity_id in self.entities:
            self.graph.add_node(entity_id, type=self.entity_types[entity_id])

        # 重置邻接张量
        self.adjacency_tensor = np.zeros((self.max_entity_num, self.max_entity_num, self.relation_num))

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
        source_idx = action // (self.max_entity_num * self.relation_num)
        remaining = action % (self.max_entity_num * self.relation_num)
        target_idx = remaining // self.relation_num
        relation_idx = remaining % self.relation_num

        # 检查索引是否有效
        if source_idx >= self.entity_num or target_idx >= self.entity_num:
            # 无效动作，给予负奖励
            reward = -0.1
            done = self._is_done()
            return self._get_state(), reward, done, {'is_valid': False}

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

        # 1. 图谱准确性奖励
        r1 = self._graph_accuracy_reward()

        # 2. 因果一致性奖励
        r2 = self._causal_consistency_reward()

        # 3. 反事实有效性奖励
        r3 = self._counterfactual_validation_reward()

        # 综合奖励
        reward = r1 + r2 + r3
        return reward

    def _graph_accuracy_reward(self):
        """
        计算图谱准确性奖励，评估预测图谱与参考图谱的一致性

        返回:
            r1: 图谱准确性奖励
        """
        # 如果参考图谱为空，返回0
        if self.reference_graph.number_of_edges() == 0:
            return 0

        # 计算真阳性(TP)、假阳性(FP)、假阴性(FN)
        tp = 0  # 正确预测的边
        fp = 0  # 错误预测的边
        fn = 0  # 未预测的边

        # 检查当前图中的每条边是否在参考图中
        for source, target, data in self.graph.edges(data=True):
            relation = data.get('relation', '')
            if self.reference_graph.has_edge(source, target) and self.reference_graph.edges[source, target].get('relation', '') == relation:
                tp += 1
            else:
                fp += 1

        # 检查参考图中的每条边是否在当前图中
        for source, target, data in self.reference_graph.edges(data=True):
            relation = data.get('relation', '')
            if not self.graph.has_edge(source, target) or self.graph.edges[source, target].get('relation', '') != relation:
                fn += 1

        # 计算精确率(Precision)和召回率(Recall)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # 计算F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 返回F1分数作为奖励
        return 0.5 * f1  # 缩放因子0.5，使奖励在合理范围内

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

    def _get_entity_type_distribution(self):
        """
        获取实体类型分布

        返回:
            type_distribution: 实体类型分布 (字典 {实体类型: 数量})
        """
        type_distribution = {}
        for entity_type in self.entity_types.values():
            if entity_type not in type_distribution:
                type_distribution[entity_type] = 0
            type_distribution[entity_type] += 1
        return type_distribution

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
