import torch
import numpy as np
import json
import random
from collections import defaultdict
import os

# ATT&CK战术层的有向图结构
# 定义为 {节点: [父节点列表]}，表示哪些节点会影响当前节点
casual_model_graph = {
    "TA0043": ["TA0040"],       # 侦察 <- 影响
    "TA0042": ["TA0043"],       # 资源开发 <- 侦察
    "TA0001": ["TA0042"],       # 初始访问 <- 资源开发
    "TA0002": ["TA0001", "TA0011"],  # 执行 <- 初始访问, 命令与控制
    "TA0003": ["TA0001", "TA0002"],  # 持久化 <- 初始访问, 执行
    "TA0004": ["TA0002", "TA0003"],  # 权限提升 <- 执行, 持久化
    "TA0005": ["TA0002", "TA0003", "TA0004"],  # 防御规避 <- 执行, 持久化, 权限提升
    "TA0006": ["TA0002", "TA0003", "TA0004", "TA0005"],  # 凭证获取 <- 执行, 持久化, 权限提升, 防御规避
    "TA0007": ["TA0002", "TA0003", "TA0004", "TA0005", "TA0006"],  # 发现 <- 执行, 持久化, 权限提升, 防御规避, 凭证获取
    "TA0008": ["TA0002", "TA0003", "TA0004", "TA0005", "TA0006", "TA0007"],  # 横向移动 <- 执行, 持久化, 权限提升, 防御规避, 凭证获取, 发现
    "TA0009": ["TA0002", "TA0003", "TA0004", "TA0005", "TA0006", "TA0007", "TA0008"],  # 收集 <- 执行, 持久化, 权限提升, 防御规避, 凭证获取, 发现, 横向移动
    "TA0011": ["TA0002", "TA0003", "TA0004", "TA0005", "TA0006", "TA0007", "TA0008", "TA0009"],  # 命令与控制 <- 执行, 持久化, 权限提升, 防御规避, 凭证获取, 发现, 横向移动, 收集
    "TA0010": ["TA0011"],       # 数据渗漏 <- 命令与控制
    "TA0040": ["TA0010"]        # 影响 <- 数据渗漏
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
            # 基线强度 - 使用小的正值初始化
            self.params[f'mu_{node}'] = torch.nn.Parameter(torch.tensor(0.05))
            # 父节点参数 (alpha和eta)
            parents = graph_structure[node]
            for parent in parents:
                # alpha (激发强度) - 使用小的正值初始化
                self.params[f'alpha_{parent}->{node}'] = torch.nn.Parameter(torch.tensor(0.1))
                # eta (时间衰减) - 使用适中的正值初始化
                self.params[f'eta_{parent}->{node}'] = torch.nn.Parameter(torch.tensor(0.5))

        # 转换为参数列表
        self.param_list = list(self.params.values())

    def intensity(self, node, t, event_history):
        """
        计算节点在离散时间t的强度
        event_history: 字典 {节点: 事件时间列表}
        """
        parents = self.graph[node]
        # 确保基线强度为正值
        mu = torch.abs(self.params[f'mu_{node}'])  # 使用绝对值确保正值
        lambda_t = mu  # 基线强度

        # 如果没有父节点，直接返回基线强度
        if not parents:
            return torch.clamp(lambda_t, min=1e-8)

        # 计算所有父节点的激发效应
        for parent in parents:
            # 确保alpha和eta为正值
            alpha = torch.abs(self.params[f'alpha_{parent}->{node}'])
            eta = torch.abs(self.params[f'eta_{parent}->{node}']) + 0.1  # 添加小的正值避免eta太接近0

            # 获取父节点事件时间
            parent_events = event_history.get(parent, torch.tensor([]))

            # 如果父节点没有事件，跳过
            if len(parent_events) == 0:
                continue

            # 筛选出t之前的事件，使用向量化操作
            mask = parent_events < t
            prev_events = parent_events[mask]

            # 如果没有t之前的事件，跳过
            if len(prev_events) == 0:
                continue

            # 计算时间差
            time_diffs = t - prev_events

            # 为了提高计算效率，只考虑最近的5个事件
            if len(time_diffs) > 5:
                # 获取最小的5个时间差（最近的5个事件）
                _, indices = torch.topk(time_diffs, 5, largest=False)
                time_diffs = time_diffs[indices]

            # 计算激发效应，使用clamp避免数值溢出
            exp_terms = torch.exp(-eta * torch.clamp(time_diffs, max=20.0))
            excitation = alpha * torch.sum(exp_terms)
            lambda_t = lambda_t + excitation

        # 确保强度为正值且不会太小，也不会太大
        return torch.clamp(lambda_t, min=1e-8, max=100.0)

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
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        # 尝试在当前工作目录下查找
        alt_path = os.path.join(os.getcwd(), "casual_model", "dataset", "tactics_sequence_data.json")
        print(f"尝试备用路径: {alt_path}")
        with open(alt_path, 'r') as f:
            data = json.load(f)

    # 初始化事件历史
    event_history = defaultdict(list)

    # 处理数据，格式为 {年份: {战术: {时间点: [战术列表]}}}
    for year, year_data in data.items():
        for tactic, tactic_data in year_data.items():
            for time_point, tactics_list in tactic_data.items():
                # 将时间点转换为整数
                time_point_int = int(time_point)
                if tactic in casual_model_graph:  # 确保战术在我们的图中
                    event_history[tactic].append(time_point_int)

                # 处理关联的战术
                for related_tactic in tactics_list:
                    if related_tactic in casual_model_graph:  # 确保战术在我们的图中
                        # 为了避免重复，检查是否已经添加
                        if time_point_int not in event_history[related_tactic]:
                            event_history[related_tactic].append(time_point_int)

    # 对每个战术的时间点进行排序
    for tactic in event_history:
        event_history[tactic].sort()

    # 打印每个战术的事件数量，用于调试
    print("加载的事件数据统计:")
    for tactic in sorted(event_history.keys()):
        print(f"  {tactic}: {len(event_history[tactic])} 个事件")

    return event_history

# ===================== 参数学习 =====================
def train_model(graph_structure, event_data, epochs=1000, lr=0.01, T=25):
    """
    训练离散时间拓扑霍克斯模型
    graph_structure: 图结构
    event_data: 事件数据 {节点: 事件时间列表}
    epochs: 训练轮数
    lr: 学习率
    T: 最大时间
    """
    # 确保T不超过数据中的最大时间点，但也不要太大
    max_time = 0
    for node, times in event_data.items():
        if times and max(times) > max_time:
            max_time = max(times)

    # 使用较小的T值，最大不超过10
    T = min(10, max_time + 1)
    print(f"使用T值: {T}")

    model = DiscreteTopoHawkesModel(graph_structure)

    # 初始化参数为更合理的值
    for node in model.nodes:
        # 基线强度初始化为较小的正值
        model.params[f'mu_{node}'].data.fill_(0.05)
        # 父节点参数初始化
        parents = graph_structure[node]
        for parent in parents:
            # alpha初始化为较小的正值
            model.params[f'alpha_{parent}->{node}'].data.fill_(0.1)
            # eta初始化为适中的正值
            model.params[f'eta_{parent}->{node}'].data.fill_(0.5)

    optimizer = torch.optim.Adam(model.param_list, lr=lr)

    # 转换事件数据为张量
    tensor_event_data = {k: torch.tensor(v, dtype=torch.float32) for k, v in event_data.items()}

    losses = []  # 记录损失值
    best_loss = float('inf')
    patience = 5  # 早停的耐心值
    patience_counter = 0

    # 为了提高效率，预先计算一些常用值
    time_points = list(range(T))

    for epoch in range(epochs):
        total_loss = torch.tensor(0.0, requires_grad=True)  # 使用Tensor初始化
        optimizer.zero_grad()

        node_losses = {}  # 记录每个节点的损失，用于调试

        # 遍历所有节点
        for node in model.nodes:
            events = tensor_event_data.get(node, torch.tensor([]))

            # 计算事件项（对数似然的正项）
            event_term = torch.tensor(0.0, requires_grad=True)
            if len(events) > 0:
                # 为了提高效率，只使用最多50个事件
                if len(events) > 50:
                    # 随机选择50个事件
                    indices = torch.randperm(len(events))[:50]
                    sampled_events = events[indices]
                else:
                    sampled_events = events

                for t in sampled_events:
                    lambda_t = model.intensity(node, t.item(), tensor_event_data)
                    # 确保强度为正且不为零
                    lambda_t = torch.clamp(lambda_t, min=1e-8)
                    event_term = event_term + torch.log(lambda_t)

                # 根据采样比例调整事件项
                if len(events) > 50:
                    event_term = event_term * (len(events) / 50)

            # 计算积分项（对数似然的负项）
            integral = torch.tensor(0.0, requires_grad=True)
            # 为了提高效率，只采样部分时间点
            sampled_time_points = time_points
            if T > 5:
                # 随机选择5个时间点
                sampled_time_points = sorted(random.sample(time_points, min(5, T)))

            for t in sampled_time_points:
                lambda_t = model.intensity(node, t, tensor_event_data)
                integral = integral + lambda_t

            # 根据采样比例调整积分项
            if T > 5:
                integral = integral * (T / len(sampled_time_points))

            # 累计损失（负对数似然）
            loss_node = -(event_term - integral)
            node_losses[node] = loss_node.item()
            total_loss = total_loss + loss_node

        # 检查损失是否为NaN或无穷大
        if isinstance(total_loss, torch.Tensor):
            is_nan = torch.isnan(total_loss) or torch.isinf(total_loss)
        else:
            import math
            is_nan = math.isnan(total_loss) or math.isinf(total_loss)

        if is_nan:
            print(f"警告: 第{epoch}轮损失为NaN或无穷大，尝试重新初始化优化器")
            lr = lr * 0.5
            optimizer = torch.optim.Adam(model.param_list, lr=lr)
            continue

        # 反向传播
        total_loss.backward()

        # 梯度裁剪以防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.param_list, max_norm=1.0)

        optimizer.step()

        # 记录损失
        current_loss = total_loss.item()
        losses.append(current_loss)

        # 早停检查
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停: 损失在{patience}轮内没有改善")
            break

        # 打印训练进度
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {current_loss:.4f}')
            # 每100轮打印一次详细的节点损失
            if epoch % 100 == 0:
                print("节点损失详情:")
                for node, loss in sorted(node_losses.items()):
                    print(f"  {node}: {loss:.4f}")
    # 保存损失数据
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thp_losses.npy'), losses)

    # 绘制损失曲线
    if epochs > 10:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(losses)
            plt.title('THP Casual Model Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thp_loss_curve.png'))
            print("损失曲线已保存")
        except ImportError:
            print("无法导入matplotlib，跳过绘制损失曲线")

    return model

# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 1. 加载ATT&CK战术序列数据
    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建数据文件的绝对路径
    data_path = os.path.join(os.path.dirname(script_dir), "dataset", "tactics_sequence_data.json")
    print(f"尝试加载数据文件: {data_path}")
    event_data = load_tactics_data(data_path)

    # 2. 训练模型 - 使用更合适的参数
    # 减少epochs以避免过拟合，使用较小的学习率以稳定训练
    # T会根据数据自动调整，不需要手动设置
    trained_model = train_model(casual_model_graph, event_data, epochs=500, lr=0.005, T=10)

    # 3. 保存模型
    model_dir = os.path.join(script_dir, "models")
    model_path = os.path.join(model_dir, "discrete_topo_hawkes_model.npz")
    os.makedirs(model_dir, exist_ok=True)
    print(f"保存模型到: {model_path}")
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

    # 6. 计算和打印每个节点的平均强度，用于验证模型
    print("\n各节点平均强度:")
    tensor_event_data = {k: torch.tensor(v, dtype=torch.float32) for k, v in event_data.items()}
    max_time = 0
    for times in event_data.values():
        if times and max(times) > max_time:
            max_time = max(times)

    for node in loaded_model.nodes:
        total_intensity = 0.0
        for t in range(max_time + 1):
            intensity = loaded_model.intensity(node, t, tensor_event_data).item()
            total_intensity += intensity
        avg_intensity = total_intensity / (max_time + 1)
        print(f"  {node}: {avg_intensity:.4f}")
