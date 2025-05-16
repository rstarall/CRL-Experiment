import os
import json
import random
import networkx as nx

def load_all_entities(data_dir="rl_moldel/dataset/ner_data", target_entity_count=256):
    """
    加载所有实体数据，并确保实体数量达到目标数量

    参数:
        data_dir: 数据目录
        target_entity_count: 目标实体数量

    返回:
        all_entities: 所有实体数据 (字典 {实体ID: 实体信息})
        entity_relations: 实体间的关系 (字典 {(源实体ID, 目标实体ID): 关系类型列表})
    """
    all_entities = {}
    entity_relations = {}
    entity_name_to_id = {}  # 用于将实体名称映射到实体ID
    file_count = 0
    entity_count = 0
    relation_count = 0

    # 获取所有年份目录
    year_dirs = []
    for year_dir in os.listdir(data_dir):
        year_path = os.path.join(data_dir, year_dir)
        if os.path.isdir(year_path):
            year_dirs.append(year_dir)

    # 按顺序处理年份目录
    year_dirs.sort()

    # 获取所有JSON文件路径
    all_json_files = []
    for year_dir in year_dirs:
        year_path = os.path.join(data_dir, year_dir)
        file_names = [f for f in os.listdir(year_path) if f.endswith('.json')]
        file_names.sort()  # 按顺序处理文件
        for file_name in file_names:
            file_path = os.path.join(year_path, file_name)
            all_json_files.append(file_path)

    # 处理单个文件的函数
    def process_file(file_path, file_index):
        nonlocal file_count, entity_count, relation_count

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                file_count += 1

                # 创建文件内部的实体ID到全局实体ID的映射
                local_id_to_global_id = {}
                file_entity_name_to_id = {}  # 文件内部的实体名称到实体ID的映射

                # 提取实体
                entities_in_file = []
                if "Entities" in data:
                    for entity in data["Entities"]:
                        local_entity_id = entity.get("EntityId", "")
                        entity_name = entity.get("EntityName", "")

                        if local_entity_id and entity_name:
                            # 创建全局唯一的实体ID
                            global_entity_id = f"file_{file_index}_{local_entity_id}"

                            # 更新实体的ID
                            entity["EntityId"] = global_entity_id

                            # 添加到全局实体集合
                            all_entities[global_entity_id] = entity

                            # 更新映射
                            local_id_to_global_id[local_entity_id] = global_entity_id
                            entity_name_to_id[entity_name] = global_entity_id
                            file_entity_name_to_id[entity_name] = global_entity_id

                            entity_count += 1
                            entities_in_file.append(global_entity_id)

                    # 为同一文件中的实体创建关系
                    # 这里我们假设同一文件中的实体之间可能存在关系
                    # 我们根据实体类型创建关系
                    from rl_moldel.action_env import action_space_constraint

                    for i, source_id in enumerate(entities_in_file):
                        for target_id in entities_in_file[i+1:]:
                            source_type = all_entities[source_id].get("EntityType", "unknown")
                            target_type = all_entities[target_id].get("EntityType", "unknown")

                            # 根据实体类型确定可能的关系类型
                            possible_relations = []
                            for relation_type, constraint in action_space_constraint.items():
                                valid_source_types = constraint.get("source_types", [])
                                valid_target_types = constraint.get("target_types", [])

                                if source_type in valid_source_types and target_type in valid_target_types:
                                    possible_relations.append(relation_type)

                            # 如果有可能的关系，随机选择一个
                            if possible_relations:
                                relation_type = random.choice(possible_relations)
                                relation_key = (source_id, target_id)
                                if relation_key not in entity_relations:
                                    entity_relations[relation_key] = []
                                if relation_type not in entity_relations[relation_key]:
                                    entity_relations[relation_key].append(relation_type)
                                    relation_count += 1

                # 如果数据中有明确的关系字段，也提取它们
                if "Relationships" in data:
                    for relation in data["Relationships"]:
                        source_name = relation.get("Source", "")
                        target_name = relation.get("Target", "")
                        relation_type = relation.get("RelationshipType", "")

                        # 将实体名称转换为实体ID
                        source_id = file_entity_name_to_id.get(source_name, "")
                        target_id = file_entity_name_to_id.get(target_name, "")

                        if source_id and target_id and relation_type:
                            relation_key = (source_id, target_id)
                            if relation_key not in entity_relations:
                                entity_relations[relation_key] = []
                            if relation_type not in entity_relations[relation_key]:
                                entity_relations[relation_key].append(relation_type)
                                relation_count += 1

                # 兼容旧版本的关系字段名称
                if "Relations" in data:
                    for relation in data["Relations"]:
                        local_source_id = relation.get("SourceId", "")
                        local_target_id = relation.get("TargetId", "")
                        relation_type = relation.get("RelationType", "")

                        # 将本地实体ID转换为全局实体ID
                        source_id = local_id_to_global_id.get(local_source_id, "")
                        target_id = local_id_to_global_id.get(local_target_id, "")

                        if source_id and target_id and relation_type:
                            relation_key = (source_id, target_id)
                            if relation_key not in entity_relations:
                                entity_relations[relation_key] = []
                            if relation_type not in entity_relations[relation_key]:
                                entity_relations[relation_key].append(relation_type)
                                relation_count += 1

                return True
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return False

    # 处理文件直到达到目标实体数量或处理完所有文件
    for file_index, file_path in enumerate(all_json_files):
        process_file(file_path, file_index)

        # 检查是否已达到目标实体数量
        if len(all_entities) >= target_entity_count:
            break

    # 如果处理完所有文件后，实体数量仍然不足目标数量，则发出警告
    if len(all_entities) < target_entity_count:
        print(f"警告: 处理完所有文件后，实际实体数量({len(all_entities)})仍不足目标数量({target_entity_count})。")
        print(f"请考虑减小目标实体数量或添加更多实体数据文件。")

    print(f"加载了 {file_count} 个文件，共 {len(all_entities)} 个实体，{relation_count} 个关系")
    return all_entities, entity_relations

def build_graph_from_entities(entities, entity_relations, max_entity_num=1024):
    """
    从实体和关系构建图

    参数:
        entities: 实体数据 (字典 {实体ID: 实体信息})
        entity_relations: 实体间的关系 (字典 {(源实体ID, 目标实体ID): 关系类型列表})
        max_entity_num: 最大实体数量

    返回:
        graph: 构建的图 (NetworkX DiGraph)
        sampled_entities: 采样的实体 (列表)
    """
    # 创建图
    graph = nx.DiGraph()

    # 获取所有实体ID
    all_entity_ids = list(entities.keys())

    # 如果实体总数小于等于最大实体数，则使用所有实体
    if len(all_entity_ids) <= max_entity_num:
        sampled_entity_ids = all_entity_ids
    else:
        # 随机抽样固定数量的实体
        sampled_entity_ids = random.sample(all_entity_ids, max_entity_num)

    # 添加节点
    for entity_id in sampled_entity_ids:
        entity_info = entities[entity_id]
        entity_type = entity_info.get('EntityType', 'unknown')
        graph.add_node(entity_id, type=entity_type, info=entity_info)

    # 添加边
    for (source_id, target_id), relation_types in entity_relations.items():
        if source_id in sampled_entity_ids and target_id in sampled_entity_ids:
            for relation_type in relation_types:
                graph.add_edge(source_id, target_id, relation=relation_type)

    return graph, sampled_entity_ids

def sample_entities_with_relations(all_entities, entity_relations, max_entity_num=1024):
    """
    采样实体并保留关系

    参数:
        all_entities: 所有实体数据 (字典 {实体ID: 实体信息})
        entity_relations: 实体间的关系 (字典 {(源实体ID, 目标实体ID): 关系类型列表})
        max_entity_num: 最大实体数量

    返回:
        sampled_entities: 采样的实体 (字典 {实体ID: 实体信息})
        sampled_relations: 采样的关系 (字典 {(源实体ID, 目标实体ID): 关系类型列表})
    """
    # 获取所有实体ID
    all_entity_ids = list(all_entities.keys())

    # 如果实体总数小于等于最大实体数，则使用所有实体
    if len(all_entity_ids) <= max_entity_num:
        sampled_entity_ids = all_entity_ids
    else:
        # 随机抽样固定数量的实体
        sampled_entity_ids = random.sample(all_entity_ids, max_entity_num)

    # 提取采样的实体
    sampled_entities = {entity_id: all_entities[entity_id] for entity_id in sampled_entity_ids}

    # 提取采样的关系
    sampled_relations = {}
    for (source_id, target_id), relation_types in entity_relations.items():
        if source_id in sampled_entity_ids and target_id in sampled_entity_ids:
            sampled_relations[(source_id, target_id)] = relation_types

    return sampled_entities, sampled_relations
