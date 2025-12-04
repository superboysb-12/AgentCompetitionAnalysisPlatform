"""
将三元组格式转换为实体+属性格式

用法:
python scripts/convert_to_entity_attributes.py knowledge_graph.json
"""

import json
import sys
from collections import defaultdict

# 定义哪些关系应该被视为属性
ATTRIBUTE_RELATIONS = {
    '具有参数', '参数值为', '使用制冷剂', '符合能效', '具有功能',
    '获得认证', '具有工程便利性'
}

def convert_triplets_to_entities(triplets):
    """将三元组转换为实体+属性格式"""

    # 第一步：收集所有实体
    entities_dict = {}  # {(name, type): {name, type, attributes: []}}

    # 第二步：分离属性关系和普通关系
    attribute_triplets = []
    regular_relations = []

    for triplet in triplets:
        relation = triplet['relation']
        subject = triplet['subject']
        subject_type = triplet['subject_type']
        obj = triplet['object']
        obj_type = triplet['object_type']

        # 确保主体实体存在
        entity_key = (subject, subject_type)
        if entity_key not in entities_dict:
            entities_dict[entity_key] = {
                'name': subject,
                'entity_type': subject_type,
                'attributes': [],
                'source_url': triplet.get('source_url', ''),
                'doc_id': triplet.get('doc_id', '')
            }

        # 确保客体实体存在
        obj_key = (obj, obj_type)
        if obj_key not in entities_dict:
            entities_dict[obj_key] = {
                'name': obj,
                'entity_type': obj_type,
                'attributes': [],
                'source_url': triplet.get('source_url', ''),
                'doc_id': triplet.get('doc_id', '')
            }

        # 判断是属性还是关系
        if relation in ATTRIBUTE_RELATIONS:
            attribute_triplets.append(triplet)
        else:
            regular_relations.append(triplet)

    # 第三步：将属性三元组转换为属性
    for triplet in attribute_triplets:
        relation = triplet['relation']
        subject = triplet['subject']
        subject_type = triplet['subject_type']
        obj = triplet['object']

        entity_key = (subject, subject_type)

        # 判断属性类型
        if relation == '参数值为':
            # 这是参数的值，subject是参数名，object是值
            attr_key = subject
            attr_value = obj
            value_type = '数值' if any(c.isdigit() for c in obj) else '文本'
        elif relation == '使用制冷剂':
            attr_key = '制冷剂'
            attr_value = obj
            value_type = '枚举'
        elif relation == '符合能效':
            attr_key = '能效等级'
            attr_value = obj
            value_type = '枚举'
        elif relation == '具有功能':
            attr_key = '功能'
            attr_value = obj
            value_type = '文本'
        elif relation == '获得认证':
            attr_key = '认证'
            attr_value = obj
            value_type = '文本'
        elif relation == '具有参数':
            # 这种情况下，obj是参数名，需要找对应的参数值
            # 暂时跳过，因为参数值会在'参数值为'中处理
            continue
        else:
            attr_key = relation
            attr_value = obj
            value_type = '文本'

        # 添加属性到实体
        attribute = {
            'key': attr_key,
            'value': attr_value,
            'value_type': value_type,
            'confidence': triplet.get('confidence', 0.9),
            'evidence': triplet.get('evidence', ''),
            'evidence_spans': triplet.get('evidence_spans', [])
        }

        entities_dict[entity_key]['attributes'].append(attribute)

    # 处理"具有参数"+"参数值为"的组合
    param_values = {}  # {参数名: 参数值}
    for triplet in attribute_triplets:
        if triplet['relation'] == '参数值为':
            param_name = triplet['subject']
            param_value = triplet['object']
            param_values[param_name] = {
                'value': param_value,
                'confidence': triplet.get('confidence', 0.9),
                'evidence': triplet.get('evidence', ''),
                'evidence_spans': triplet.get('evidence_spans', [])
            }

    for triplet in attribute_triplets:
        if triplet['relation'] == '具有参数':
            subject = triplet['subject']
            subject_type = triplet['subject_type']
            param_name = triplet['object']

            entity_key = (subject, subject_type)

            if param_name in param_values:
                param_info = param_values[param_name]
                attribute = {
                    'key': param_name,
                    'value': param_info['value'],
                    'value_type': '数值' if any(c.isdigit() for c in param_info['value']) else '文本',
                    'confidence': param_info['confidence'],
                    'evidence': param_info['evidence'],
                    'evidence_spans': param_info['evidence_spans']
                }
                entities_dict[entity_key]['attributes'].append(attribute)

    entities = list(entities_dict.values())
    return entities, regular_relations

def main():
    if len(sys.argv) < 2:
        print("用法: python convert_to_entity_attributes.py <input_json>")
        sys.exit(1)

    input_file = sys.argv[1]

    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取三元组
    if 'triplets' in data:
        if isinstance(data['triplets'], dict):
            triplets = data['triplets'].get('all', [])
        else:
            triplets = data['triplets']
    else:
        print("未找到triplets字段")
        sys.exit(1)

    # 转换
    entities, relations = convert_triplets_to_entities(triplets)

    # 构建输出
    output = {
        'metadata': data.get('metadata', {}),
        'entities': entities,
        'relations': relations,
        'summary': {
            'total_entities': len(entities),
            'total_relations': len(relations),
            'entities_with_attributes': len([e for e in entities if e['attributes']]),
            'total_attributes': sum(len(e['attributes']) for e in entities)
        }
    }

    # 保存
    output_file = input_file.replace('.json', '_with_attributes.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✓ 转换完成！")
    print(f"  实体数: {len(entities)}")
    print(f"  关系数: {len(relations)}")
    print(f"  有属性的实体: {len([e for e in entities if e['attributes']])}")
    print(f"  总属性数: {sum(len(e['attributes']) for e in entities)}")
    print(f"  输出文件: {output_file}")

if __name__ == '__main__':
    main()
