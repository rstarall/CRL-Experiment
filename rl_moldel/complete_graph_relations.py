import os
import argparse
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import json
import random
import sys

# 检查matplotlib是否可用
try:
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False
    print("警告: matplotlib未安装，将不会生成可视化图表。")
    print("如需完整功能，请安装matplotlib: pip install matplotlib")

# 添加项目根目录到路径
sys.path.append("experiment")

from graph_env import GraphEnvironment
from ppo import PPO
from casual_model.THP.train import DiscreteTopoHawkesModel, casual_model_graph

# 加载实体数据
def load_entity_data(data_dir="rl_moldel/dataset/ner_data"):
    """
    加载实体数据

    参数:
        data_dir: 数据目录

    返回:
        entity_data: 实体数据 (字典 {实体ID: 实体信息})
    """
    entity_data = {}

    # 遍历数据目录
    for year_dir in os.listdir(data_dir):
        year_path = os.path.join(data_dir, year_dir)
        if os.path.isdir(year_path):
            for file_name in os.listdir(year_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(year_path, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            if "Entities" in data:
                                for entity in data["Entities"]:
                                    entity_id = entity.get("EntityId", "")
                                    if entity_id:
                                        entity_data[entity_id] = entity
                        except json.JSONDecodeError:
                            print(f"Error loading {file_path}")

    return entity_data

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def visualize_graph(graph, output_path, title="Graph Visualization"):
    """
    可视化图结构

    参数:
        graph: 图结构 (字典 {源实体: [(目标实体, 关系类型)]})
        output_path: 输出路径
        title: 图标题
    """
    if not has_matplotlib:
        print(f"无法生成可视化图表 {output_path}，matplotlib未安装")
        return

    G = nx.DiGraph()

    # 添加节点
    for source in graph:
        G.add_node(source)

    # 添加边
    for source, targets in graph.items():
        for target, relation in targets:
            G.add_edge(source, target, relation=relation)

    # 绘制图
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=1500, arrowsize=20, font_size=12,
            font_weight='bold', arrows=True)

    # 添加边标签
    edge_labels = {(u, v): d['relation'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.title(title)
    plt.savefig(output_path)
    plt.close()

    print(f"图表已保存到: {output_path}")

def create_partial_graph(entity_data, missing_rate=0.3):
    """
    创建部分图结构

    参数:
        entity_data: 实体数据
        missing_rate: 缺失率

    返回:
        partial_graph: 部分图结构
    """
    # 构建完整图
    full_graph = {}

    # 遍历实体数据
    for entity_id, entity_info in entity_data.items():
        full_graph[entity_id] = []

        # 获取实体类型
        entity_type = entity_info.get('EntityType', 'unknown')

        # 获取实体关系
        relations = entity_info.get('Relations', [])

        for relation in relations:
            target_id = relation.get('TargetId', '')
            relation_type = relation.get('RelationType', '')

            if target_id and relation_type and target_id in entity_data:
                full_graph[entity_id].append((target_id, relation_type))

    # 创建部分图
    partial_graph = {}
    for entity_id, targets in full_graph.items():
        partial_graph[entity_id] = []
        for target in targets:
            if random.random() > missing_rate:
                partial_graph[entity_id].append(target)

    return full_graph, partial_graph

def calculate_metrics(true_graph, predicted_graph):
    """
    计算评估指标

    参数:
        true_graph: 真实图结构
        predicted_graph: 预测图结构

    返回:
        metrics: 评估指标
    """
    # 提取所有边
    true_edges = set()
    for source, targets in true_graph.items():
        for target, relation in targets:
            true_edges.add((source, target, relation))

    predicted_edges = set()
    for source, targets in predicted_graph.items():
        for target, relation in targets:
            predicted_edges.add((source, target, relation))

    # 计算指标
    tp = len(true_edges & predicted_edges)
    fp = len(predicted_edges - true_edges)
    fn = len(true_edges - predicted_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def complete_graph(partial_graph, entity_data, ppo_model, hawkes_model, max_steps=100, device='cpu'):
    """
    补全图结构

    参数:
        partial_graph: 部分图结构
        entity_data: 实体数据
        ppo_model: PPO模型
        hawkes_model: 霍克斯模型
        max_steps: 最大步数
        device: 设备

    返回:
        completed_graph: 补全后的图结构
    """
    # 创建环境
    env = GraphEnvironment(hawkes_model, entity_data, max_steps=max_steps)

    # 重置环境，使用部分图作为初始状态
    state = env.reset(partial_graph)

    # 预测缺失关系
    done = False
    step = 0

    while not done and step < max_steps:
        # 选择动作
        action, _ = ppo_model.select_action(state)

        # 执行动作
        next_state, reward, done, info = env.step(action)

        # 更新状态
        state = next_state
        step += 1

    # 获取补全后的图结构
    completed_graph = env.get_current_graph()

    return completed_graph

def main():
    parser = argparse.ArgumentParser(description="威胁情报图谱关系补全")
    parser.add_argument('--model_path', type=str, default='experiment/rl_moldel/output/graph_ppo_model.pth', help='PPO模型路径')
    parser.add_argument('--thp_model_path', type=str, default='experiment/casual_model/THP/models/discrete_topo_hawkes_model.npz', help='THP模型路径')
    parser.add_argument('--input_graph', type=str, default='', help='输入的部分图谱路径')
    parser.add_argument('--output_graph', type=str, default='experiment/rl_moldel/output/completed_graph.json', help='输出的补全图谱路径')
    parser.add_argument('--missing_rate', type=float, default=0.3, help='缺失率')
    parser.add_argument('--max_steps', type=int, default=100, help='最大步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--cuda', type=int, default=0, help='GPU设备ID')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = os.path.dirname(args.output_graph)
    os.makedirs(output_dir, exist_ok=True)

    # 加载实体数据
    print("加载实体数据...")
    entity_data = load_entity_data()
    print(f"加载了 {len(entity_data)} 个实体")

    # 加载THP模型
    print(f"加载THP模型: {args.thp_model_path}")
    hawkes_model = DiscreteTopoHawkesModel.load_model(args.thp_model_path, casual_model_graph)

    # 创建部分图结构
    if args.input_graph:
        # 从文件加载部分图
        with open(args.input_graph, 'r') as f:
            partial_graph = json.load(f)
        # 构建真实图（如果有）
        true_graph = None
    else:
        # 随机创建部分图
        true_graph, partial_graph = create_partial_graph(entity_data, args.missing_rate)

    # 创建环境
    env = GraphEnvironment(hawkes_model, entity_data, max_steps=args.max_steps)

    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # PPO超参数
    ppo_config = {
        "gamma": 0.99,
        "lr_actor": 0.0003,
        "lr_critic": 0.0003,
        "eps_clip": 0.2,
        "K_epochs": 50,
        "batch_size": 64,
        "hidden_units": 128,
        "has_continuous_action_space": False,
        "max_grad_norm": 0.5,
        "normalize_advantage": True
    }

    # 创建PPO代理
    print("创建PPO代理...")
    agent = PPO(state_dim, action_dim, ppo_config, device)

    # 加载模型
    print(f"加载模型: {args.model_path}")
    agent.load(args.model_path)

    # 补全图结构
    print("开始补全图结构...")
    completed_graph = complete_graph(partial_graph, entity_data, agent, hawkes_model, args.max_steps, device)

    # 计算评估指标
    if true_graph:
        metrics = calculate_metrics(true_graph, completed_graph)
        print(f"评估指标: {metrics}")

    # 保存补全后的图结构
    with open(args.output_graph, 'w') as f:
        json.dump(completed_graph, f, indent=4)
    print(f"补全后的图结构已保存到: {args.output_graph}")

    # 可视化图结构
    if true_graph:
        visualize_graph(true_graph, os.path.join(output_dir, "true_graph.png"), "True Graph")
    visualize_graph(partial_graph, os.path.join(output_dir, "partial_graph.png"), "Partial Graph")
    visualize_graph(completed_graph, os.path.join(output_dir, "completed_graph.png"), "Completed Graph")

    # 保存评估结果
    results = {
        'partial_graph': partial_graph,
        'completed_graph': completed_graph
    }

    if true_graph:
        results['true_graph'] = true_graph
        results['metrics'] = metrics

    with open(os.path.join(output_dir, "completion_results.pkl"), 'wb') as f:
        pickle.dump(results, f)

    print("图谱补全完成!")

if __name__ == "__main__":
    main()
