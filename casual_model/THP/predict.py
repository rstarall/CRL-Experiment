import torch
import copy
from scipy.stats import chi2
import numpy as np

class CausalAnalyzer:
    def __init__(self, model):
        self.model = model

    def ice(self, j, i, current_time, event_history, delta_t=3):
        """
        计算离散时间瞬时因果效应

        参数:
        j: 源节点（父节点）
        i: 目标节点（子节点）
        current_time: 当前时间点
        event_history: 事件历史（字典，值为PyTorch张量或列表）
        delta_t: 时间窗口大小，默认为3（适合紧凑的时间跨度）
        """
        # 检查是否存在直接因果关系
        direct_effect = 0.0
        if f'alpha_{j}->{i}' in self.model.params:
            alpha = self.model.params[f'alpha_{j}->{i}'].item()
            eta = self.model.params[f'eta_{j}->{i}'].item()
            events_j = event_history.get(j, torch.tensor([]))

            # 确保events_j是PyTorch张量
            if not isinstance(events_j, torch.Tensor):
                events_j = torch.tensor(events_j, dtype=torch.float32)

            # 筛选时间窗口内的事件
            if len(events_j) > 0:
                # 使用PyTorch的布尔索引
                mask = (current_time - delta_t <= events_j) & (events_j < current_time)
                valid_events = events_j[mask]

                # 计算时间衰减效应
                for event_time in valid_events:
                    direct_effect += alpha * np.exp(-eta * (current_time - event_time.item()))

        # 检查是否存在间接因果关系（通过中间节点）
        indirect_effect = 0.0
        # 查找所有可能的中间节点
        for k in self.model.nodes:
            if (f'alpha_{j}->{k}' in self.model.params and
                f'alpha_{k}->{i}' in self.model.params and
                k != j and k != i):

                # 计算j->k的效应
                alpha_jk = self.model.params[f'alpha_{j}->{k}'].item()
                eta_jk = self.model.params[f'eta_{j}->{k}'].item()

                # 计算k->i的效应
                alpha_ki = self.model.params[f'alpha_{k}->{i}'].item()
                eta_ki = self.model.params[f'eta_{k}->{i}'].item()

                events_j = event_history.get(j, torch.tensor([]))
                events_k = event_history.get(k, torch.tensor([]))

                # 确保events_j和events_k是PyTorch张量
                if not isinstance(events_j, torch.Tensor):
                    events_j = torch.tensor(events_j, dtype=torch.float32)
                if not isinstance(events_k, torch.Tensor):
                    events_k = torch.tensor(events_k, dtype=torch.float32)

                # 筛选j的有效事件
                if len(events_j) > 0:
                    mask_j = (current_time - delta_t <= events_j) & (events_j < current_time)
                    valid_j_events = events_j[mask_j]
                else:
                    valid_j_events = torch.tensor([])

                # 筛选k的有效事件（这些事件可能是由j引起的）
                if len(events_k) > 0:
                    mask_k = (current_time - delta_t/2 <= events_k) & (events_k < current_time)
                    valid_k_events = events_k[mask_k]
                else:
                    valid_k_events = torch.tensor([])

                # 计算间接效应
                for j_time in valid_j_events:
                    j_time_val = j_time.item()
                    # j对k的效应
                    effect_jk = alpha_jk * np.exp(-eta_jk * (current_time - j_time_val))

                    for k_time in valid_k_events:
                        k_time_val = k_time.item()
                        if k_time_val > j_time_val:  # 确保k的事件发生在j之后
                            # k对i的效应
                            effect_ki = alpha_ki * np.exp(-eta_ki * (current_time - k_time_val))
                            # 累积间接效应
                            indirect_effect += effect_jk * effect_ki * 0.1  # 缩小间接效应的权重

        # 返回总效应（直接效应 + 间接效应）
        return direct_effect + indirect_effect

    def cce(self, i, t_start, t_end, event_history):
        """
        计算离散时间累积因果效应

        参数:
        i: 目标节点
        t_start: 开始时间
        t_end: 结束时间
        event_history: 事件历史（字典，值为PyTorch张量或列表）
        """
        if t_start >= t_end:
            return 0.0

        # 添加调试信息
        print(f"CCE调试: 节点={i}, t_start={t_start}, t_end={t_end}")
        print(f"CCE调试: event_history类型={type(event_history)}")
        for node, events in event_history.items():
            print(f"CCE调试: 节点={node}, 事件类型={type(events)}, 事件值={events}")

        # 确保event_history中的值是PyTorch张量
        tensor_event_history = {}
        for node, events in event_history.items():
            if not isinstance(events, torch.Tensor):
                print(f"CCE调试: 转换节点{node}的事件为张量")
                tensor_event_history[node] = torch.tensor(events, dtype=torch.float32)
            else:
                tensor_event_history[node] = events

        mu_i = self.model.params[f'mu_{i}'].item()
        cumulative_effect = 0.0

        # 对离散时间点求和
        for t in range(t_start, t_end + 1):
            try:
                lambda_t = self.model.intensity(i, t, tensor_event_history).item()
                # 计算超出基线强度的部分
                excess_intensity = lambda_t - mu_i
                if excess_intensity > 0:  # 只考虑正效应
                    cumulative_effect += excess_intensity
            except Exception as e:
                print(f"CCE调试: 在t={t}计算强度时出错: {str(e)}")
                # 打印当前节点的事件
                if i in tensor_event_history:
                    print(f"CCE调试: 节点{i}的事件={tensor_event_history[i]}")
                raise

        # 计算平均效应
        time_span = t_end - t_start + 1
        if time_span > 0:
            return cumulative_effect / time_span
        return 0.0

    def causal_significance(self, j, i, event_data):
        """因果显著性检验"""
        # 确保因果边存在
        if f'alpha_{j}->{i}' not in self.model.params:
            return 1.0  # 不存在的边，p值为1

        # 确保event_data中的值是PyTorch张量
        tensor_event_data = {}
        for node, events in event_data.items():
            if not isinstance(events, torch.Tensor):
                tensor_event_data[node] = torch.tensor(events, dtype=torch.float32)
            else:
                tensor_event_data[node] = events

        # 计算全模型对数似然
        full_loglik = self._calculate_log_likelihood(self.model, tensor_event_data)

        # 创建移除j->i边的模型
        removed_model = copy.deepcopy(self.model)
        # 将alpha设为0（相当于移除边）
        if f'alpha_{j}->{i}' in removed_model.params:
            removed_model.params[f'alpha_{j}->{i}'].data.copy_(torch.tensor(0.0))

        # 计算移除边后的对数似然
        removed_loglik = self._calculate_log_likelihood(removed_model, tensor_event_data)

        # 计算似然比统计量
        lambda_ratio = 2 * (full_loglik - removed_loglik)

        # 计算p值（自由度为1）
        p_value = 1 - chi2.cdf(lambda_ratio, df=1)
        return p_value

    def _calculate_log_likelihood(self, model, event_data):
        """计算模型的对数似然"""
        log_lik = 0.0

        # 确保event_data中的值是PyTorch张量，并找出最大时间
        max_time = 0
        for node, events in event_data.items():
            if not isinstance(events, torch.Tensor):
                event_data[node] = torch.tensor(events, dtype=torch.float32)

            if len(event_data[node]) > 0:
                node_max = event_data[node].max().item()
                if node_max > max_time:
                    max_time = node_max

        T = int(max_time) + 1

        for node in model.nodes:
            events = event_data.get(node, torch.tensor([]))
            if not isinstance(events, torch.Tensor):
                events = torch.tensor(events, dtype=torch.float32)

            # 事件项
            for t in events:
                lambda_t = model.intensity(node, t.item(), event_data)
                log_lik += torch.log(lambda_t + 1e-10)

            # 积分项（离散时间求和）
            for t in range(T):
                lambda_t = model.intensity(node, t, event_data)
                log_lik -= lambda_t

        return log_lik.item()

    def att(self, j, i, event_history, t_start=None, t_end=None):
        """
        计算平均处理效应（反事实干预）

        参数:
        j: 源节点（被干预的节点）
        i: 目标节点（观察效应的节点）
        event_history: 事件历史（字典，值为PyTorch张量或列表）
        t_start: 开始时间
        t_end: 结束时间
        """
        # 确保event_history中的值是PyTorch张量
        tensor_event_history = {}
        for node, events in event_history.items():
            if not isinstance(events, torch.Tensor):
                tensor_event_history[node] = torch.tensor(events, dtype=torch.float32)
            else:
                tensor_event_history[node] = events

        if t_start is None or t_end is None:
            # 默认使用整个时间范围
            all_times = []
            for events in tensor_event_history.values():
                if len(events) > 0:
                    all_times.extend(events.tolist())
            if not all_times:
                return 0.0
            t_start = 0
            t_end = int(max(all_times))

        # 计算原始强度
        original_intensity = 0.0
        for t in range(t_start, t_end + 1):
            lambda_t = self.model.intensity(i, t, tensor_event_history).item()
            original_intensity += lambda_t

        # 创建干预后的事件历史（移除j的所有事件）
        intervened_history = copy.deepcopy(tensor_event_history)
        intervened_history[j] = torch.tensor([], dtype=torch.float32)

        # 计算干预后的强度
        intervened_intensity = 0.0
        for t in range(t_start, t_end + 1):
            lambda_t = self.model.intensity(i, t, intervened_history).item()
            intervened_intensity += lambda_t

        # 计算平均处理效应
        time_span = t_end - t_start + 1
        if time_span > 0:
            att = (original_intensity - intervened_intensity) / time_span

            # 归一化处理效应（相对于原始强度）
            if original_intensity > 0:
                normalized_att = att / (original_intensity / time_span)
                # 返回归一化后的效应，限制在[0,1]范围内
                return min(max(normalized_att, 0.0), 1.0)

        return 0.0

# 辅助函数：打印模型参数
def print_model_parameters(model):
    """打印模型的参数，按照因果关系排序"""
    # 打印基线强度
    print("基线强度 (mu):")
    for node in sorted(model.nodes):
        mu = model.params[f'mu_{node}'].item()
        print(f"  {node}: {mu:.3f}")

    # 打印因果关系参数
    print("\n因果关系参数 (alpha, eta):")
    causal_params = []
    for node in model.nodes:
        for parent in model.graph.get(node, []):
            if f'alpha_{parent}->{node}' in model.params:
                alpha = model.params[f'alpha_{parent}->{node}'].item()
                eta = model.params[f'eta_{parent}->{node}'].item()
                causal_params.append((parent, node, alpha, eta))

    # 按alpha值排序（因果强度）
    for parent, node, alpha, eta in sorted(causal_params, key=lambda x: -x[2]):
        print(f"  {parent} -> {node}: alpha={alpha:.3f}, eta={eta:.3f}")

# 辅助函数：将事件历史转换为PyTorch张量
def convert_to_tensor(event_history):
    """将事件历史转换为PyTorch张量"""
    tensor_event_history = {}
    for node, events in event_history.items():
        if not isinstance(events, torch.Tensor):
            tensor_event_history[node] = torch.tensor(events, dtype=torch.float32)
        else:
            tensor_event_history[node] = events
    return tensor_event_history

# 使用示例
def predict_example():
    # 加载训练好的模型
    import os
    import sys

    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 导入同目录下的train.py中的类和变量
    sys.path.append(script_dir)
    from train import DiscreteTopoHawkesModel, casual_model_graph

    # 构建模型路径
    model_path = os.path.join(script_dir, "models", "discrete_topo_hawkes_model.npz")
    print(f"尝试加载模型: {model_path}")
    trained_model = DiscreteTopoHawkesModel.load_model(model_path, casual_model_graph)

    # 创建分析器
    analyzer = CausalAnalyzer(trained_model)

    # 创建一个超紧凑的测试用例，所有事件都在0-5的时间范围内，可能有多个事件在同一时间点
    ultra_compact_events_raw = {
        # 侦察 (Reconnaissance)
        'TA0043': [0, 1],
        # 资源开发 (Resource Development)
        'TA0042': [1, 2],
        # 初始访问 (Initial Access)
        'TA0001': [2, 3],
        # 执行 (Execution)
        'TA0002': [2, 3],
        # 持久化 (Persistence)
        'TA0003': [3],
        # 权限提升 (Privilege Escalation)
        'TA0004': [3, 4],
        # 防御规避 (Defense Evasion)
        'TA0005': [3, 4],
        # 凭证获取 (Credential Access)
        'TA0006': [4],
        # 发现 (Discovery)
        'TA0007': [4],
        # 横向移动 (Lateral Movement)
        'TA0008': [4, 5],
        # 收集 (Collection)
        'TA0009': [4, 5],
        # 命令与控制 (Command and Control)
        'TA0011': [5],
        # 数据渗漏 (Exfiltration)
        'TA0010': [5],
        # 影响 (Impact)
        'TA0040': [5]
    }

    # 将事件列表转换为PyTorch张量
    ultra_compact_events = {k: torch.tensor(v, dtype=torch.float32) for k, v in ultra_compact_events_raw.items()}

    # 创建另一个测试用例，使用随机值但保持在0-5范围内
    import random
    random.seed(42)  # 使用固定种子以确保结果可重现

    random_events_raw = {}
    for tactic in casual_model_graph.keys():
        # 为每个战术生成1-3个随机时间点
        num_events = random.randint(1, 3)
        events = sorted([random.randint(0, 5) for _ in range(num_events)])
        random_events_raw[tactic] = events

    # 将随机事件列表转换为PyTorch张量
    random_events = {k: torch.tensor(v, dtype=torch.float32) for k, v in random_events_raw.items()}

    print("\n===== 使用超紧凑事件序列测试 (0-5时间范围) =====")
    print("所有事件都在0-5的时间范围内，可能有多个事件在同一时间点...")

    # 打印事件序列
    print("\n事件序列:")
    for tactic, times in sorted(ultra_compact_events_raw.items()):
        print(f"{tactic}: {times}")

    print("\n===== 测试强因果关系 =====")
    # 测试直接因果关系：侦察->资源开发 (根据修正后的图结构，TA0043是TA0042的父节点)
    ice_value = analyzer.ice('TA0043', 'TA0042', current_time=3, event_history=ultra_compact_events, delta_t=3)
    print(f"瞬时因果效应(侦察->资源开发): {ice_value:.3f}")

    # 测试直接因果关系：资源开发->初始访问 (根据修正后的图结构，TA0042是TA0001的父节点)
    ice_value2 = analyzer.ice('TA0042', 'TA0001', current_time=4, event_history=ultra_compact_events, delta_t=3)
    print(f"瞬时因果效应(资源开发->初始访问): {ice_value2:.3f}")

    # 测试直接因果关系：初始访问->执行 (根据修正后的图结构，TA0001是TA0002的父节点)
    ice_value3 = analyzer.ice('TA0001', 'TA0002', current_time=4, event_history=ultra_compact_events, delta_t=3)
    print(f"瞬时因果效应(初始访问->执行): {ice_value3:.3f}")

    print("\n===== 测试累积效应 =====")
    # 计算资源开发在[0,5]时间段的累积效应
    cce_value = analyzer.cce('TA0042', 0, 5, ultra_compact_events)
    print(f"累积因果效应(资源开发[0-5]): {cce_value:.3f}")

    # 计算执行在[0,5]时间段的累积效应
    cce_value2 = analyzer.cce('TA0002', 0, 5, ultra_compact_events)
    print(f"累积因果效应(执行[0-5]): {cce_value2:.3f}")

    print("\n===== 测试因果显著性 =====")
    # 检验侦察->资源开发的显著性 (根据修正后的图结构)
    p_value = analyzer.causal_significance('TA0043', 'TA0042', ultra_compact_events)
    print(f"因果显著性(侦察->资源开发, p值): {p_value:.4f}")

    # 检验资源开发->初始访问的显著性 (根据修正后的图结构)
    p_value2 = analyzer.causal_significance('TA0042', 'TA0001', ultra_compact_events)
    print(f"因果显著性(资源开发->初始访问, p值): {p_value2:.4f}")

    # 检验初始访问->执行的显著性 (根据修正后的图结构)
    p_value3 = analyzer.causal_significance('TA0001', 'TA0002', ultra_compact_events)
    print(f"因果显著性(初始访问->执行, p值): {p_value3:.4f}")

    print("\n===== 测试平均处理效应 =====")
    # 计算侦察->资源开发的平均处理效应 (根据修正后的图结构)
    att_value = analyzer.att('TA0043', 'TA0042', ultra_compact_events, t_start=0, t_end=5)
    print(f"平均处理效应(侦察->资源开发): {att_value:.3f}")

    # 计算资源开发->初始访问的平均处理效应 (根据修正后的图结构)
    att_value2 = analyzer.att('TA0042', 'TA0001', ultra_compact_events, t_start=0, t_end=5)
    print(f"平均处理效应(资源开发->初始访问): {att_value2:.3f}")

    # 计算初始访问->执行的平均处理效应 (根据修正后的图结构)
    att_value3 = analyzer.att('TA0001', 'TA0002', ultra_compact_events, t_start=0, t_end=5)
    print(f"平均处理效应(初始访问->执行): {att_value3:.3f}")

    print("\n===== 测试间接因果关系 =====")
    # 测试间接因果关系：影响->资源开发 (TA0040->TA0043->TA0042)
    ice_value4 = analyzer.ice('TA0040', 'TA0042', current_time=3, event_history=ultra_compact_events, delta_t=3)
    print(f"瞬时因果效应(影响->资源开发): {ice_value4:.3f}")

    # 测试间接因果关系：侦察->初始访问 (TA0043->TA0042->TA0001)
    ice_value5 = analyzer.ice('TA0043', 'TA0001', current_time=4, event_history=ultra_compact_events, delta_t=3)
    print(f"瞬时因果效应(侦察->初始访问): {ice_value5:.3f}")

    # 测试间接因果关系：资源开发->执行 (TA0042->TA0001->TA0002)
    ice_value6 = analyzer.ice('TA0042', 'TA0002', current_time=5, event_history=ultra_compact_events, delta_t=4)
    print(f"瞬时因果效应(资源开发->执行): {ice_value6:.3f}")

    print("\n\n===== 使用随机事件序列测试 (0-5时间范围) =====")
    print("使用随机生成的事件序列，每个战术有1-3个随机时间点...")

    # 打印随机事件序列
    print("\n随机事件序列:")
    for tactic, times in sorted(random_events_raw.items()):
        print(f"{tactic}: {times}")

    print("\n===== 测试强因果关系（随机数据）=====")
    # 测试直接因果关系：侦察->资源开发 (根据修正后的图结构)
    ice_value_r = analyzer.ice('TA0043', 'TA0042', current_time=3, event_history=random_events, delta_t=5)
    print(f"瞬时因果效应(侦察->资源开发): {ice_value_r:.3f}")

    # 测试直接因果关系：资源开发->初始访问 (根据修正后的图结构)
    ice_value2_r = analyzer.ice('TA0042', 'TA0001', current_time=3, event_history=random_events, delta_t=5)
    print(f"瞬时因果效应(资源开发->初始访问): {ice_value2_r:.3f}")

    # 测试直接因果关系：初始访问->执行 (根据修正后的图结构)
    ice_value3_r = analyzer.ice('TA0001', 'TA0002', current_time=4, event_history=random_events, delta_t=5)
    print(f"瞬时因果效应(初始访问->执行): {ice_value3_r:.3f}")

    # 测试间接因果关系：侦察->初始访问 (TA0043->TA0042->TA0001)
    ice_value4_r = analyzer.ice('TA0043', 'TA0001', current_time=4, event_history=random_events, delta_t=5)
    print(f"瞬时因果效应(侦察->初始访问): {ice_value4_r:.3f}")

    # 打印模型参数
    print("\n===== 模型参数 =====")
    print_model_parameters(trained_model)

if __name__ == "__main__":
    predict_example()