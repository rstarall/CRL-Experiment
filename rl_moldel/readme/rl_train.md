# 基于离散时间拓扑霍克斯过程的威胁情报图谱关系推理补全方案

本文档描述了如何使用离散时间拓扑霍克斯过程(THP)作为环境模型，结合离线图谱数据，通过PPO算法进行强化学习训练，实现威胁情报图谱中缺失关系的预测与补全。

## 1. 概述

威胁情报图谱中的实体和关系往往存在缺失，本方案将离散时间拓扑霍克斯过程模型与强化学习深度融合，实现：
1. **因果感知的关系预测**：智能体能理解图谱节点间的动态因果关系
2. **反事实推理能力**：评估假设性干预对攻击链路的影响
3. **图谱补全能力**：预测并补全缺失的实体关系
4. **可解释性保障**：奖励机制与因果效应直接挂钩

## 2. 数据准备

### 2.1 离线图谱数据简化处理

从`rl_moldel/dataset/ner_data`目录中的JSON文件提取以下信息：

1. **实体信息**：提取每个实体的ID、类型和战术标签
2. **时序信息**：提取每个实体的战术发生时间
3. **实体关系**：提取实体间的关系类型

简化处理步骤：
```python
def load_entity_data(data_dir="rl_moldel/dataset/ner_data"):
    entity_data = {}

    for year_dir in os.listdir(data_dir):
        year_path = os.path.join(data_dir, year_dir)
        if os.path.isdir(year_path):
            for file_name in os.listdir(year_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(year_path, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "Entities" in data:
                            for entity in data["Entities"]:
                                entity_id = entity.get("EntityId", "")
                                if entity_id:
                                    entity_data[entity_id] = entity

    return entity_data
```

### 2.2 构建实体-战术映射

将实体与其对应的战术标签关联起来：
```python
entity_to_tactics = {}
for entity_id, entity_info in entity_data.items():
    entity_to_tactics[entity_id] = entity_info.get('Labels', [])
```

## 3. 离散时间拓扑霍克斯模型

### 3.1 模型定义

对于战术节点 $i \in \{TA_1,...,TA_{14}\}$，其离散时间强度函数为：
$$
\lambda_i(t) = \mu_i + \sum_{j \in Pa(i)} \sum_{s \in H_j(t)} \alpha_{j \to i} \cdot e^{-\eta_{j \to i}(t-s)}
$$

其中：
- $\mu_i$ 为战术节点i的基线强度
- $Pa(i)$ 表示因果模型图中节点i的父战术集合
- $\alpha_{j \to i}$ 表示战术j对i的激发效应强度
- $\eta_{j \to i}$ 为时间衰减系数
- $H_j(t)$ 表示战术j在时间t之前的所有离散事件时间点集合

### 3.2 使用预训练模型

本方案使用`casual_model\THP\models\discrete_topo_hawkes_model.npz`中已训练好的THP模型，该模型包含了战术间的因果关系参数。

```python
# 加载THP模型
hawkes_model = DiscreteTopoHawkesModel.load_model(
    'experiment/casual_model/THP/models/discrete_topo_hawkes_model.npz',
    casual_model_graph
)
```

## 4. 强化学习环境设计

### 4.1 环境组件

| 组件 | 描述 |
|------|------|
| **状态空间** | 当前图谱状态（节点存在性、连接状态、关系类型） |
| **动作空间** | 基于`action_env.py`中定义的关系类型添加/删除边 |
| **奖励函数** | 基于因果模型的奖励：图谱完整性(r1) + 因果一致性(r2) + 反事实有效性(r3) |
| **环境模型** | 使用离散时间拓扑霍克斯模型预测动作后的因果状态变化 |

### 4.2 动作空间定义

使用`rl_moldel\action_env.py`中定义的动作空间，该空间定义了不同类型实体之间的关系类型：

```python
action_space_constraint = {
    "use": {
        "source_types": ["attcker"],
        "target_types": ["tool", "vul", "ioc"]
    },
    "trigger": {
        "source_types": ["victim"],
        "target_types": ["file", "env", "ioc"]
    },
    # 更多关系类型...
}
```

### 4.3 因果驱动的奖励函数

$$
r(s,a,s') = \underbrace{\text{图完整性}(s')}_{r1} + \underbrace{\sum_{j \to i} \mathbb{I}(\alpha_{j \to i} > \gamma) \cdot \text{ICE}_{j \to i}(t)}_{r2} + \underbrace{\mathbb{E}[\text{ATT}_{\text{pred}} - \text{ATT}_{\text{real}}]}_{r3}
$$

- **r1**: 使用图谱完整性度量补全图谱的合理性
- **r2**: 对符合因果强度阈值（$\gamma$）的边给予奖励
- **r3**: 通过反事实干预前后的效应差异评估动作合理性

## 5. PPO算法实现

### 5.1 算法概述

PPO (Proximal Policy Optimization) 是一种策略梯度算法，通过限制策略更新的步长来提高训练稳定性。

### 5.2 关键特性

1. **动作有效性检查**：根据`action_env.py`中的约束检查动作是否有效
2. **实体-战术映射**：考虑一个实体可能处于多个战术阶段
3. **反事实评估**：使用THP模型进行反事实干预评估

### 5.3 训练流程

1. **加载预训练THP模型**：使用已训练好的THP模型
2. **环境初始化**：使用实体数据和THP模型初始化环境
3. **PPO训练**：使用PPO算法进行训练
4. **模型评估**：使用反事实干预评估模型性能

## 6. 图谱关系补全

### 6.1 补全流程

1. **输入缺失图谱**：提供包含部分关系的威胁情报图谱
2. **状态编码**：将图谱转换为模型可处理的状态表示
3. **关系预测**：使用训练好的PPO模型预测缺失的关系
4. **结果验证**：通过反事实干预验证预测关系的合理性
5. **图谱更新**：将预测的关系添加到原图谱中



## 7. 使用指南

### 7.1 训练命令

```bash
python rl_moldel/train_graph_ppo.py --cuda 0 --seed 42 --max_episodes 1000 --thp_model_path experiment/casual_model/THP/models/discrete_topo_hawkes_model.npz
```

参数说明：
- `--cuda`: 使用的GPU设备ID
- `--seed`: 随机种子
- `--max_episodes`: 最大训练回合数
- `--thp_model_path`: THP模型路径

### 7.2 图谱补全命令

```bash
python rl_moldel/complete_graph_relations.py --model_path output/graph_ppo_model.pth --thp_model_path experiment/casual_model/THP/models/discrete_topo_hawkes_model.npz --input_graph input/partial_graph.json --output_graph output/completed_graph.json
```

参数说明：
- `--model_path`: 训练好的PPO模型路径
- `--thp_model_path`: THP模型路径
- `--input_graph`: 输入的部分图谱路径
- `--output_graph`: 输出的补全图谱路径

### 7.3 超参数设置

| 参数 | 值 | 说明 |
|------|-----|------|
| gamma | 0.99 | 折扣因子 |
| lr_actor | 0.0003 | Actor网络学习率 |
| lr_critic | 0.0003 | Critic网络学习率 |
| eps_clip | 0.2 | PPO裁剪参数 |
| eps_causal | 0.2 | 因果探索参数 |
| K_epochs | 50 | 每次更新的epoch数 |
| batch_size | 64 | 批量大小 |
| hidden_units | 128 | 隐藏层单元数 |

## 8. 性能优化建议

1. **并行计算**：对多个边/节点的因果效应计算进行批处理
2. **模型蒸馏**：将复杂的霍克斯模型简化为轻量级代理模型用于RL训练
3. **课程学习**：从简单攻击场景逐步过渡到复杂多阶段攻击
4. **时间窗口优化**：使用较短的时间跨度（每个阶段时间在5之内）
5. **测试用例设计**：生成更合理、更有意义的结果，避免极端值（全0或全1）

## 9. 应用场景

该方法可广泛应用于：
1. APT攻击检测与关系补全
2. 网络防御策略优化
3. 安全事件关联分析
4. 攻击链路预测与重构
5. 威胁情报图谱知识库扩充

## 10. 注意事项

1. 确保使用的是有向图(directed graph)进行因果建模
2. 测试用例应使用较短的时间跨度，每个阶段时间应在5之内
3. 测试用例应产生更合理、更有意义的结果，避免极端值
4. 实际部署时需根据具体业务需求调整奖励权重和状态编码方式
5. 一个图谱节点可能处于多个战术阶段，需要考虑这种多对多的映射关系
