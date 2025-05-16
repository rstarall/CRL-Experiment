import os
import numpy as np
import torch
import networkx as nx
import random
import copy
import gym
from rl_moldel.graph_env import GraphEnvironment

class ExtendedGraphEnvironment(GraphEnvironment):
    """
    扩展的图谱环境，支持动态更新实体数据和重置环境
    """

    def update_entity_data(self, entity_data):
        """
        更新环境的实体数据

        参数:
            entity_data: 新的实体数据 (字典 {实体ID: 实体信息})
        """
        # 更新实体数据
        self.entity_data = entity_data

        # 提取实体和关系
        self.entities = list(entity_data.keys())
        self.entity_num = len(self.entities)

        # 提取实体类型
        self.entity_types = {}
        for entity_id, entity_info in entity_data.items():
            self.entity_types[entity_id] = entity_info.get('EntityType', 'unknown')

        # 更新动作空间
        self.action_dim = self.entity_num * self.entity_num * self.relation_num
        self.action_space.n = self.action_dim

        # 更新状态空间
        self.state_dim = self.entity_num * self.entity_num * self.relation_num + 1
        # 创建新的观察空间，因为Box的shape是只读的
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.state_dim,), dtype=np.float32
        )

        # 更新邻接张量
        self.adjacency_tensor = np.zeros((self.entity_num, self.entity_num, self.relation_num))

        # 构建实体到战术的映射
        self.entity_to_tactics = {}
        for entity_id, entity_info in entity_data.items():
            self.entity_to_tactics[entity_id] = entity_info.get('Labels', [])

        # 构建战术事件历史
        self.tactic_event_history = self._build_tactic_event_history()

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
                if source in self.entities:  # 确保源实体存在
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
