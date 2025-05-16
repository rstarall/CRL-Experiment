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
    file_count = 0
    entity_count = 0
    relation_count = 0
    synthetic_count = 0

    # 遍历数据目录
    for year_dir in os.listdir(data_dir):
        year_path = os.path.join(data_dir, year_dir)
        if os.path.isdir(year_path):
            for file_name in os.listdir(year_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(year_path, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            file_count += 1

                            # 提取实体
                            if "Entities" in data:
                                entities_in_file = []
                                for entity in data["Entities"]:
                                    entity_id = entity.get("EntityId", "")
                                    if entity_id:
                                        if entity_id not in all_entities:
                                            all_entities[entity_id] = entity
                                            entity_count += 1
                                        entities_in_file.append(entity_id)

                                # 为同一文件中的实体创建关系
                                # 这里我们假设同一文件中的实体之间可能存在关系
                                # 我们根据实体类型创建关系
                                from action_env import action_space_constraint

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
                            if "Relations" in data:
                                for relation in data["Relations"]:
                                    source_id = relation.get("SourceId", "")
                                    target_id = relation.get("TargetId", "")
                                    relation_type = relation.get("RelationType", "")

                                    if source_id and target_id and relation_type:
                                        relation_key = (source_id, target_id)
                                        if relation_key not in entity_relations:
                                            entity_relations[relation_key] = []
                                        if relation_type not in entity_relations[relation_key]:
                                            entity_relations[relation_key].append(relation_type)
                                            relation_count += 1
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

    # 如果实体数量不足目标数量，生成合成实体
    original_entity_count = len(all_entities)
    if original_entity_count < target_entity_count:
        print(f"实际实体数量({original_entity_count})不足目标数量({target_entity_count})，生成合成实体...")

        # 获取所有实体类型
        entity_types = {}
        for entity_id, entity in all_entities.items():
            entity_type = entity.get("EntityType", "unknown")
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity_id)

        # 生成合成实体
        original_entities = list(all_entities.keys())
        while len(all_entities) < target_entity_count:
            # 随机选择一个原始实体作为模板
            template_id = random.choice(original_entities)
            template_entity = all_entities[template_id]

            # 创建一个新的合成实体
            synthetic_id = f"synthetic_entity_{synthetic_count}"
            synthetic_entity = template_entity.copy()
            synthetic_entity["EntityId"] = synthetic_id

            # 修改实体名称
            if "EntityName" in synthetic_entity:
                synthetic_entity["EntityName"] = f"{synthetic_entity['EntityName']}_syn_{synthetic_count}"

            # 保持实体类型不变
            entity_type = synthetic_entity.get("EntityType", "unknown")

            # 随机修改一些属性，使合成实体有所不同
            if "Labels" in synthetic_entity and synthetic_entity["Labels"]:
                # 随机删除或添加一些标签
                if random.random() < 0.5 and len(synthetic_entity["Labels"]) > 1:
                    synthetic_entity["Labels"].pop(random.randrange(len(synthetic_entity["Labels"])))
                else:
                    all_labels = set()
                    for e in all_entities.values():
                        if "Labels" in e:
                            all_labels.update(e["Labels"])
                    if all_labels:
                        new_label = random.choice(list(all_labels))
                        if new_label not in synthetic_entity["Labels"]:
                            synthetic_entity["Labels"].append(new_label)

            # 随机修改时间
            if "Times" in synthetic_entity and synthetic_entity["Times"]:
                synthetic_entity["Times"] = [str(int(t) + random.randint(-2, 2)) for t in synthetic_entity["Times"]]

            # 添加合成实体
            all_entities[synthetic_id] = synthetic_entity
            synthetic_count += 1

            # 为合成实体创建关系
            # 与同类型的实体创建关系
            if entity_type in entity_types:
                for other_id in entity_types[entity_type]:
                    if other_id != synthetic_id and random.random() < 0.3:  # 30%的概率创建关系
                        from action_env import action_space_constraint

                        # 根据实体类型确定可能的关系类型
                        possible_relations = []
                        for relation_type, constraint in action_space_constraint.items():
                            valid_source_types = constraint.get("source_types", [])
                            valid_target_types = constraint.get("target_types", [])

                            if entity_type in valid_source_types and entity_type in valid_target_types:
                                possible_relations.append(relation_type)

                        # 如果有可能的关系，随机选择一个
                        if possible_relations:
                            relation_type = random.choice(possible_relations)

                            # 创建双向关系
                            relation_key1 = (synthetic_id, other_id)
                            if relation_key1 not in entity_relations:
                                entity_relations[relation_key1] = []
                            if relation_type not in entity_relations[relation_key1]:
                                entity_relations[relation_key1].append(relation_type)
                                relation_count += 1

                            relation_key2 = (other_id, synthetic_id)
                            if relation_key2 not in entity_relations:
                                entity_relations[relation_key2] = []
                            if relation_type not in entity_relations[relation_key2]:
                                entity_relations[relation_key2].append(relation_type)
                                relation_count += 1

            # 更新实体类型列表
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(synthetic_id)

    print(f"加载了 {file_count} 个文件，共 {original_entity_count} 个原始实体，{synthetic_count} 个合成实体，总计 {len(all_entities)} 个实体，{relation_count} 个关系")
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
