# 定义动作空间约束
# 每种关系类型都有其对应的源实体类型和目标实体类型约束

action_space_constraint = {
    "use": {
        "source_types": ["attacker"],
        "target_types": ["tool", "vul", "ioc"]
    },
    "trigger": {
        "source_types": ["victim"],
        "target_types": ["file", "env", "ioc"]
    },
    "involve": {
        "source_types": ["event"],
        "target_types": ["attacker", "victim"]
    },
    "target": {
        "source_types": ["attacker"],
        "target_types": ["victim", "asset", "env"]
    },
    "has": {
        "source_types": ["victim"],
        "target_types": ["asset", "env"]
    },
    "exploit": {
        "source_types": ["vul"],
        "target_types": ["asset", "env"]
    },
    "affect": {
        "source_types": ["file"],
        "target_types": ["asset", "env"]
    },
    "related_to": {
        "source_types": ["tool"],
        "target_types": ["vul", "ioc", "file"]
    },
    "belong_to": {
        "source_types": ["file", "ioc"],
        "target_types": ["asset", "env"]
    }
}

# 检查关系是否有效
def is_valid_relation(source_type, target_type, relation_type):
    """
    检查关系是否有效
    
    参数:
        source_type: 源实体类型
        target_type: 目标实体类型
        relation_type: 关系类型
        
    返回:
        is_valid: 关系是否有效
    """
    if relation_type not in action_space_constraint:
        return False
    
    constraint = action_space_constraint[relation_type]
    valid_source_types = constraint.get('source_types', [])
    valid_target_types = constraint.get('target_types', [])
    
    return source_type in valid_source_types and target_type in valid_target_types
