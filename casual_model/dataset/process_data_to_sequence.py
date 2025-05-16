"""
    处理成的格式：
    {
        "year":{
            "tactic1":{
                "time1":[tactic1,...]
                "time2":[tactic2,...]
            }
        }
    }
    某战术tactic1(出现时间time1),关联到time2的一个战术tactic2,...
"""
import os
import json
import glob
from collections import defaultdict

def extract_tactics_from_file(file_path):
    """
    从单个JSON文件中提取战术序列数据

    参数:
        file_path: JSON文件路径

    返回:
        year: 文件对应的年份
        tactics_data: 提取的战术序列数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 从文件名中提取年份
        file_name = os.path.basename(file_path)
        year = file_name.split('_')[0]

        # 创建战术时间映射
        tactics_time_map = defaultdict(lambda: defaultdict(list))

        # 处理实体
        if "Entities" in data:
            for entity in data["Entities"]:
                if "Labels" in entity and "Times" in entity:
                    tactics = entity["Labels"]
                    times = entity["Times"]

                    # 对每个战术和时间进行映射
                    for tactic in tactics:
                        for time in times:
                            # 记录当前战术在当前时间点出现
                            tactics_time_map[tactic][time].append(tactic)

                            # 查找与当前实体相关的其他实体的战术
                            if "Relationships" in data:
                                for relation in data["Relationships"]:
                                    if relation["Source"] == entity["EntityName"]:
                                        # 查找目标实体
                                        for target_entity in data["Entities"]:
                                            if target_entity["EntityName"] == relation["Target"]:
                                                # 添加目标实体的战术到当前战术的后续时间点
                                                if "Labels" in target_entity:
                                                    target_tactics = target_entity["Labels"]
                                                    for target_time in target_entity.get("Times", []):
                                                        if int(target_time) > int(time):  # 只添加时间点更晚的战术
                                                            for target_tactic in target_tactics:
                                                                if target_tactic not in tactics_time_map[tactic][target_time]:
                                                                    tactics_time_map[tactic][target_time].append(target_tactic)

        return year, dict(tactics_time_map)

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None, None

def process_all_data(data_dir):
    """
    处理指定目录下所有JSON文件，生成战术序列数据

    参数:
        data_dir: 数据目录路径

    返回:
        result_data: 按年份组织的战术序列数据
    """
    result_data = defaultdict(lambda: defaultdict(dict))

    # 获取所有JSON文件
    json_files = []
    print(f"正在扫描目录: {data_dir}")
    for year_dir in os.listdir(data_dir):
        year_path = os.path.join(data_dir, year_dir)
        if os.path.isdir(year_path):
            year_files = glob.glob(os.path.join(year_path, "*.json"))
            json_files.extend(year_files)
            print(f"年份 {year_dir}: 找到 {len(year_files)} 个文件")

    print(f"总共找到 {len(json_files)} 个JSON文件")

    # 处理每个文件
    processed_count = 0
    for file_path in json_files:
        processed_count += 1
        if processed_count % 50 == 0:
            print(f"已处理 {processed_count}/{len(json_files)} 个文件...")

        year, tactics_data = extract_tactics_from_file(file_path)
        if year and tactics_data:
            # 合并同一年份的数据
            for tactic, time_data in tactics_data.items():
                for time, tactics_list in time_data.items():
                    if time not in result_data[year][tactic]:
                        result_data[year][tactic][time] = []

                    # 合并战术列表，避免重复
                    for t in tactics_list:
                        if t not in result_data[year][tactic][time]:
                            result_data[year][tactic][time].append(t)

    print(f"所有文件处理完成，共处理 {processed_count} 个文件")
    return dict(result_data)

def save_result(result_data, output_file):
    """
    保存结果数据到JSON文件

    参数:
        result_data: 处理后的结果数据
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
    print(f"结果已保存到 {output_file}")

def main():
    # 数据目录路径
    data_dir = "ner_data"

    # 输出文件路径
    output_file = "tactics_sequence_data.json"

    # 处理数据
    result_data = process_all_data(data_dir)

    # 保存结果
    save_result(result_data, output_file)

    # 统计战术关系对
    total_pairs, unique_pairs, pair_counts = count_tactic_pairs(result_data)

    # 输出统计结果
    print("\n战术关系对统计:")
    print(f"总关系对数量: {total_pairs}")
    print(f"唯一关系对数量: {unique_pairs}")

    # 输出出现频率最高的前10个关系对
    print("\n出现频率最高的前10个关系对:")
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    for i, ((source, target), count) in enumerate(sorted_pairs[:10], 1):
        print(f"{i}. {source} -> {target}: {count}次")

# 统计战术关系对的函数
def count_tactic_pairs(result_data):
    """
    统计战术关系对的数量

    参数:
        result_data: 处理后的结果数据

    返回:
        total_pairs: 战术关系对的总数
        unique_pairs: 唯一战术关系对的数量
        pair_counts: 每个唯一战术关系对的出现次数
    """
    total_pairs = 0
    unique_pairs = set()
    pair_counts = defaultdict(int)

    for year, tactics in result_data.items():
        for source_tactic, time_data in tactics.items():
            for time, target_tactics in time_data.items():
                for target_tactic in target_tactics:
                    if source_tactic != target_tactic:  # 排除自身关系
                        pair = (source_tactic, target_tactic)
                        unique_pairs.add(pair)
                        pair_counts[pair] += 1
                        total_pairs += 1

    return total_pairs, len(unique_pairs), dict(pair_counts)

if __name__ == "__main__":
    main()
