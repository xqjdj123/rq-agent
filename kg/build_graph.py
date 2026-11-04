"""
知识图谱构建模块
基于NetworkX构建实体关系图谱
实体：业务/意图/问题/答案/法规/流程/材料/渠道/安全事项
"""
import json
import os
import networkx as nx

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
KB_PATH = os.path.join(DATA_DIR, 'knowledge_base.json')
KG_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graph_export.json')


class KnowledgeGraph:
    """燃气客服知识图谱"""

    def __init__(self):
        self.g = nx.MultiDiGraph()
        self.kb = self._load_kb()

    def _load_kb(self):
        with open(KB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    def build(self):
        """构建知识图谱"""
        print("Building knowledge graph...")

        # 法规层级映射 + 真实法规名称（从知识库数据推断）
        regulation_info = {
            'FG-FL-001': {'name': '中华人民共和国安全生产法', 'level': '法律'},
            'FG-FL-002': {'name': '中华人民共和国消防法', 'level': '法律'},
            'FG-FL-003': {'name': '中华人民共和国管道保护法', 'level': '法律'},
            'FG-XZ-001': {'name': '城镇燃气管理条例', 'level': '行政法规'},
            'FG-XZ-002': {'name': '特种设备安全监察条例', 'level': '行政法规'},
            'FG-BM-001': {'name': '燃气燃烧器具安装维修管理规定', 'level': '部门规章'},
            'FG-BM-002': {'name': '湖南省燃气经营许可管理实施办法', 'level': '部门规章'},
            'FG-DF-001': {'name': '湖南省燃气管理条例', 'level': '地方规范'},
            'FG-DF-002': {'name': '某市燃气管理规定', 'level': '地方规范'},
            'FG-HY-001': {'name': '城镇燃气设计规范(GB 50028)', 'level': '行业规范'},
            'FG-CZ-001': {'name': '某市燃气收费标准与操作规程', 'level': '操作规范'},
        }
        level_map = {k: v['level'] for k, v in regulation_info.items()}

        # 收集所有实体
        regulations = {}  # {code: {name, level}}
        intents = {}      # {code: {name, cat1}}
        businesses = set()  # 一级域
        questions = []
        answers = []
        safety_items = []

        for item in self.kb:
            cat1 = item['一级域']
            cat2 = item['二级标签']
            fg_code = item['法规编码']

            businesses.add(cat1)

            # 意图实体
            intent_key = f"{cat1}>{cat2}"
            if intent_key not in intents:
                intents[intent_key] = {
                    'name': f"{cat1}-{cat2}",
                    'cat1': cat1,
                    'cat2': cat2
                }

            # 法规实体
            if fg_code and fg_code not in regulations:
                info = regulation_info.get(fg_code, {'name': f'法规{fg_code}', 'level': '其他'})
                regulations[fg_code] = {
                    'code': fg_code,
                    'name': info['name'],
                    'level': info['level']
                }

            # 问题实体
            questions.append({
                'id': item['id'],
                'text': item['标准问'],
                'intent_key': intent_key
            })

            # 答案实体
            answers.append({
                'id': f"A-{item['id']}",
                'text': item['标准答'][:100],
                'intent_key': intent_key,
                'fg_code': fg_code,
                'is_safety': item['安全敏感']
            })

            if item['安全敏感']:
                safety_items.append({
                    'id': f"S-{item['id']}",
                    'text': item['标准答'][:100],
                    'intent_key': intent_key
                })

        # 添加节点
        print(f"  Adding {len(businesses)} Business nodes")
        for b in businesses:
            self.g.add_node(f"Business:{b}", type='Business', name=b)

        print(f"  Adding {len(intents)} Intent nodes")
        for key, val in intents.items():
            self.g.add_node(f"Intent:{key}", type='Intent', name=val['name'],
                          category=val['cat1'])

        print(f"  Adding {len(regulations)} Regulation nodes")
        for code, val in regulations.items():
            self.g.add_node(f"Regulation:{code}", type='Regulation',
                          name=val['name'], level=val['level'])

        print(f"  Adding {len(questions)} Question nodes")
        for q in questions:
            self.g.add_node(f"Question:{q['id']}", type='Question', text=q['text'][:50])

        print(f"  Adding {len(answers)} Answer nodes")
        for a in answers:
            self.g.add_node(f"Answer:{a['id']}", type='Answer', text=a['text'][:50])

        print(f"  Adding {len(safety_items)} SafetyItem nodes")
        for s in safety_items:
            self.g.add_node(f"SafetyItem:{s['id']}", type='SafetyItem', text=s['text'][:50])

        # 添加关系
        print("  Adding relationships...")
        for q in questions:
            # Question -> Intent
            self.g.add_edge(f"Question:{q['id']}", f"Intent:{q['intent_key']}",
                          relation='属于')

        for a in answers:
            # Intent -> Answer
            self.g.add_edge(f"Intent:{a['intent_key']}", f"Answer:{a['id']}",
                          relation='对应答案')
            # Answer -> Regulation
            if a['fg_code']:
                self.g.add_edge(f"Answer:{a['id']}", f"Regulation:{a['fg_code']}",
                              relation='依据')

        # Business -> Intent
        for key, val in intents.items():
            self.g.add_edge(f"Business:{val['cat1']}", f"Intent:{key}",
                          relation='包含')

        for s in safety_items:
            self.g.add_edge(f"Intent:{s['intent_key']}", f"SafetyItem:{s['id']}",
                          relation='需强制动作')

        print(f"Graph built: {self.g.number_of_nodes()} nodes, {self.g.number_of_edges()} edges")
        return self.g

    def query_regulation(self, intent_name: str) -> list:
        """查询某个意图关联的法规"""
        results = []
        intent_node = f"Intent:{intent_name}"
        if intent_node in self.g:
            for _, answer_node in self.g.out_edges(intent_node):
                if self.g.nodes[answer_node].get('type') == 'Answer':
                    for _, reg_node in self.g.out_edges(answer_node):
                        if self.g.nodes[reg_node].get('type') == 'Regulation':
                            results.append(self.g.nodes[reg_node])
        return results

    def export(self):
        """导出为前端可视化JSON"""
        data = {
            'nodes': [],
            'edges': []
        }

        node_types = {
            'Business': {'color': '#4CAF50', 'size': 30},
            'Intent': {'color': '#2196F3', 'size': 25},
            'Question': {'color': '#FF9800', 'size': 15},
            'Answer': {'color': '#9C27B0', 'size': 15},
            'Regulation': {'color': '#F44336', 'size': 20},
            'SafetyItem': {'color': '#FF5722', 'size': 20},
        }

        for node_id, attrs in self.g.nodes(data=True):
            ntype = attrs.get('type', 'Unknown')
            style = node_types.get(ntype, {'color': '#999', 'size': 10})
            data['nodes'].append({
                'id': node_id,
                'label': attrs.get('name', attrs.get('text', node_id))[:30],
                'type': ntype,
                'color': style['color'],
                'size': style['size'],
                'category': attrs.get('category', ''),
                'level': attrs.get('level', '')
            })

        for u, v, attrs in self.g.edges(data=True):
            data['edges'].append({
                'source': u,
                'target': v,
                'relation': attrs.get('relation', '')
            })

        with open(KG_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Graph exported to {KG_OUTPUT}")
        print(f"  Nodes: {len(data['nodes'])}, Edges: {len(data['edges'])}")
        return data

    def get_stats(self) -> dict:
        """获取图谱统计信息"""
        type_counts = {}
        for _, attrs in self.g.nodes(data=True):
            t = attrs.get('type', 'Unknown')
            type_counts[t] = type_counts.get(t, 0) + 1

        rel_counts = {}
        for _, _, attrs in self.g.edges(data=True):
            r = attrs.get('relation', 'Unknown')
            rel_counts[r] = rel_counts.get(r, 0) + 1

        return {
            'total_nodes': self.g.number_of_nodes(),
            'total_edges': self.g.number_of_edges(),
            'node_types': type_counts,
            'relation_types': rel_counts
        }


def main():
    kg = KnowledgeGraph()
    kg.build()
    kg.export()
    stats = kg.get_stats()
    print("\n=== 知识图谱统计 ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
