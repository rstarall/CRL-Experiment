# 离散时间拓扑霍克斯过程模型训练方案

以下是为离散时间多节点拓扑霍克斯过程设计的完整参数学习代码，包含模型保存和加载功能：

```python
import torch
import numpy as np
import json
from collections import defaultdict
import os

# ATT&CK战术层的有向图结构
casual_model_graph = {
    "TA0043": ["TA0042"],
    "TA0042": ["TA0001"],
    "TA0001": ["TA0002", "TA0003"],
    "TA0002": ["TA0003", "TA0004", "TA0005", "TA0006", "TA0007", "TA0008", "TA0009", "TA0011"],
    "TA0003": ["TA0004", "TA0005", "TA0006", "TA0007", "TA0008", "TA0009", "TA0011"],
    "TA0004": ["TA0005", "TA0006", "TA0007", "TA0008", "TA0009", "TA0011"],
    "TA0005": ["TA0006", "TA0007", "TA0008", "TA0009", "TA0011"],
    "TA0006": ["TA0007", "TA0008", "TA0009", "TA0011"],
    "TA0007": ["TA0008", "TA0009", "TA0011"],
    "TA0008": ["TA0009", "TA0011"],
    "TA0009": ["TA0011"],
    "TA0011": ["TA0010", "TA0040","TA0002"],
    "TA0010": ["TA0040"],
    "TA0040": ["TA0043"]
}

# ===================== 模型定义 =====================
class DiscreteTopoHawkesModel:
    def __init__(self, graph_structure):
        """
        graph_structure: 字典 {节点: [父节点列表]}
        示例: {'TA0001': [], 'TA0002': ['TA0001'], ...}
        """
        self.graph = graph_structure
        self.nodes = list(graph_structure.keys())
        self.params = {}
        
        # 初始化参数
        for node in self.nodes:
            # 基线强度
            self.params[f'mu_{node}'] = torch.nn.Parameter(torch.tensor(0.1))
            # 父节点参数 (alpha和eta)
            parents = graph_structure[node]
            for parent in parents:
                self.params[f'alpha_{parent}->{node}'] = torch.nn.Parameter(torch.randn(1).abs() * 0.1)
                self.params[f'eta_{parent}->{node}'] = torch.nn.Parameter(torch.randn(1).abs() * 0.1 + 0.5)
        
        # 转换为参数列表
        self.param_list = list(self.params.values())
    
    def intensity(self, node, t, event_history):
        """
        计算节点在离散时间t的强度
        event_history: 字典 {节点: 事件时间列表}
        """
        parents = self.graph[node]
        lambda_t = self.params[f'mu_{node}']
        
        for parent in parents:
            alpha = self.params[f'alpha_{parent}->{node}']
            eta = self.params[f'eta_{parent}->{node}']
            # 获取父节点事件时间
            parent_events = event_history[parent]
            if len(parent_events) > 0:
                # 筛选出t之前的事件
                prev_events = [e for e in parent_events if e < t]
                if prev_events:
                    # 计算时间差
                    time_diffs = torch.tensor([t - e for e in prev_events], dtype=torch.float32)
                    # 计算激发效应
                    excitation = alpha * torch.sum(torch.exp(-eta * time_diffs))
                    lambda_t += excitation
        
        return lambda_t
    
    def save_model(self, path):
        state_dict = {k: v.detach().numpy() for k, v in self.params.items()}
        np.savez(path, **state_dict)
    
    @classmethod
    def load_model(cls, path, graph_structure):
        data = np.load(path)
        model = cls(graph_structure)
        for k in model.params:
            if k in data:
                model.params[k].data.copy_(torch.from_numpy(data[k]))
        return model

# ===================== 数据加载 =====================
def load_tactics_data(file_path):
    """
    从JSON文件加载战术序列数据
    返回: 字典 {节点: 事件时间列表}
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 初始化事件历史
    event_history = defaultdict(list)
    
    # 处理每个序列
    for sequence in data:
        for timestamp, tactic in enumerate(sequence):
            if tactic in casual_model_graph:  # 确保战术在我们的图中
                event_history[tactic].append(timestamp)
    
    return event_history

# ===================== 参数学习 =====================
def train_model(graph_structure, event_data, epochs=1000, lr=0.01, T=100):
    """
    训练离散时间拓扑霍克斯模型
    graph_structure: 图结构
    event_data: 事件数据 {节点: 事件时间列表}
    epochs: 训练轮数
    lr: 学习率
    T: 最大时间
    """
    model = DiscreteTopoHawkesModel(graph_structure)
    optimizer = torch.optim.Adam(model.param_list, lr=lr)
    
    # 转换事件数据为张量
    tensor_event_data = {k: torch.tensor(v, dtype=torch.float32) for k, v in event_data.items()}
    
    for epoch in range(epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        
        # 遍历所有节点
        for node in model.nodes:
            events = tensor_event_data.get(node, torch.tensor([]))
            if len(events) == 0:
                continue
                
            # 计算事件项（对数似然的正项）
            event_term = 0.0
            for t in events:
                lambda_t = model.intensity(node, t.item(), event_data)
                event_term += torch.log(lambda_t + 1e-6)
            
            # 计算积分项（对数似然的负项）
            # 对于离散时间，我们对每个时间点计算强度并求和
            integral = 0.0
            for t in range(T):
                lambda_t = model.intensity(node, t, event_data)
                integral += lambda_t
            
            # 累计损失（负对数似然）
            loss_node = -(event_term - integral)
            total_loss += loss_node
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss.item():.4f}')
    
    return model

# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 1. 加载ATT&CK战术序列数据
    data_path = os.path.join("experiment", "casual_model", "dataset", "tactics_sequence_data.json")
    event_data = load_tactics_data(data_path)
    
    # 2. 训练模型
    trained_model = train_model(casual_model_graph, event_data, epochs=2000, lr=0.01, T=100)
    
    # 3. 保存模型
    model_path = os.path.join("experiment", "casual_model", "THP", "models", "discrete_topo_hawkes_model.npz")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    trained_model.save_model(model_path)
    
    # 4. 加载模型
    loaded_model = DiscreteTopoHawkesModel.load_model(model_path, casual_model_graph)
    
    # 5. 查看参数示例
    print("\nLearned Parameters:")
    for node in casual_model_graph:
        print(f"Node {node}:")
        print(f"  mu: {loaded_model.params[f'mu_{node}'].item():.3f}")
        for parent in casual_model_graph[node]:
            alpha = loaded_model.params[f'alpha_{parent}->{node}'].item()
            eta = loaded_model.params[f'eta_{parent}->{node}'].item()
            print(f"  {parent}->{node}: alpha={alpha:.3f}, eta={eta:.3f}")
```

## 离散时间模型修改说明

| 修改点 | 原连续时间模型 | 离散时间模型 |
|--------|--------------|-------------|
| 时间表示 | 连续实数值 | 离散整数时间戳 |
| 数据加载 | 模拟生成连续时间事件 | 从JSON文件加载离散序列 |
| 强度计算 | 连续时间激发函数 | 离散时间点上的激发效应 |
| 积分计算 | 数值积分近似 | 离散时间点求和 |
| 参数初始化 | 随机初始化 | 更稳定的初始化范围 |

## 离散时间模型优势

1. **直接适配离散数据**: 无需连续时间近似，直接处理离散时间序列
2. **计算效率更高**: 避免了连续模型中的数值积分计算
3. **更符合实际**: 安全事件通常以离散时间记录（如日志时间戳）
4. **参数解释性**: α表示父节点对子节点的影响强度，η表示影响的衰减速率

## 使用说明

1. 确保数据文件路径正确: `experiment/casual_model/dataset/tactics_sequence_data.json`
2. 模型将保存在: `experiment/casual_model/THP/models/discrete_topo_hawkes_model.npz`
3. 可通过调整`epochs`和`lr`参数优化训练效果
4. 参数`T`应设置为数据集中最大时间戳+1