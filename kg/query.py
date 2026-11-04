"""
知识图谱查询模块
接入主对话流程，提供法规追溯和关联推理
"""
import json
import os
import networkx as nx

KG_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(KG_DIR, 'graph_export.json')


class GraphQuery:
    """知识图谱查询器"""

    def __init__(self):
        self.g = nx.MultiDiGraph()
        self._load()

    def _load(self):
        """从导出的JSON加载图谱"""
        with open(GRAPH_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for node in data['nodes']:
            self.g.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})

        for edge in data['edges']:
            self.g.add_edge(edge['source'], edge['target'], relation=edge.get('relation', ''))

        print(f"Graph loaded: {self.g.number_of_nodes()} nodes, {self.g.number_of_edges()} edges")

    def lookup_regulation(self, fg_code: str) -> dict:
        """
        根据FG编码查询法规详情
        解决当前只能显示"FG-FL-001"而看不到法规名称的问题
        """
        node_id = f"Regulation:{fg_code}"
        if node_id in self.g:
            attrs = self.g.nodes[node_id]
            return {
                'code': fg_code,
                'name': attrs.get('label', ''),
                'level': attrs.get('level', ''),
                'color': attrs.get('color', '')
            }
        return {'code': fg_code, 'name': '', 'level': ''}

    def get_regulation_for_answer(self, answer_text: str) -> str:
        """
        通过答案内容反查法规（遍历图谱匹配）
        """
        # 按文本前缀匹配 Answer 节点
        for node_id, attrs in self.g.nodes(data=True):
            if attrs.get('type') == 'Answer' and attrs.get('label', '')[:30] in answer_text:
                # 找到 Answer -> Regulation 的边
                for _, target, edge_attrs in self.g.out_edges(node_id, data=True):
                    if edge_attrs.get('relation') == '依据':
                        return target.replace('Regulation:', '')
        return ''

    def get_related_questions(self, intent_name: str, limit: int = 5) -> list:
        """
        查询某意图下的关联问题（用于推荐追问）
        intent_name 格式: "安全用气-燃气泄漏应急处置" 或 "安全用气>燃气泄漏应急处置"
        """
        # 兼容两种分隔符
        if '>' in intent_name:
            node_id = f"Intent:{intent_name}"
        else:
            node_id = f"Intent:{intent_name.replace('-', '>', 1)}"

        if node_id not in self.g:
            return []

        questions = []
        for source, _, attrs in self.g.in_edges(node_id, data=True):
            if attrs.get('relation') == '属于':
                source_attrs = self.g.nodes[source]
                if source_attrs.get('type') == 'Question':
                    questions.append(source_attrs.get('label', ''))

        return questions[:limit]

    def get_regulations_by_intent(self, cat1: str, cat2: str = '') -> list:
        """
        查询某意图下所有关联的法规（用于法规溯源展示）
        从 Intent 找到所有 Answer，再找到所有 Regulation
        """
        intent_key = f"{cat1}>{cat2}" if cat2 else cat1
        intent_node = f"Intent:{intent_key}"

        if intent_node not in self.g:
            return []

        regulations = set()
        for _, answer_node, _ in self.g.out_edges(intent_node, data=True):
            if self.g.nodes[answer_node].get('type') == 'Answer':
                for _, reg_node, _ in self.g.out_edges(answer_node, data=True):
                    if self.g.nodes[reg_node].get('type') == 'Regulation':
                        node = self.g.nodes[reg_node]
                        regulations.add((reg_node.replace('Regulation:', ''),
                                        node.get('label', ''),
                                        node.get('level', '')))

        return [{'code': c, 'name': n, 'level': l} for c, n, l in sorted(regulations)]

    def get_intent_chain(self, cat1: str) -> list:
        """
        查询某业务域下完整的: 业务 → 意图 → 法规 链
        用于答辩展示业务全貌
        """
        business_node = f"Business:{cat1}"
        if business_node not in self.g:
            return []

        chain = []
        for _, intent_node, _ in self.g.out_edges(business_node, data=True):
            intent_attrs = self.g.nodes[intent_node]
            intent_name = intent_attrs.get('label', '')

            # 找该意图下的法规
            regs = set()
            for _, answer_node, _ in self.g.out_edges(intent_node, data=True):
                if self.g.nodes[answer_node].get('type') == 'Answer':
                    for _, reg_node, _ in self.g.out_edges(answer_node, data=True):
                        if self.g.nodes[reg_node].get('type') == 'Regulation':
                            regs.add(self.g.nodes[reg_node].get('label', ''))

            chain.append({
                'intent': intent_name,
                'regulations': sorted(regs)
            })

        return chain

    def get_graph_summary(self) -> dict:
        """获取图谱概览（用于前端展示）"""
        type_counts = {}
        for _, attrs in self.g.nodes(data=True):
            t = attrs.get('type', '?')
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            'total_nodes': self.g.number_of_nodes(),
            'total_edges': self.g.number_of_edges(),
            'node_types': type_counts,
            'business_count': type_counts.get('Business', 0),
            'intent_count': type_counts.get('Intent', 0),
            'regulation_count': type_counts.get('Regulation', 0),
            'qa_pairs': type_counts.get('Question', 0),
        }


# 单例
_graph_query = None

def get_graph_query() -> GraphQuery:
    global _graph_query
    if _graph_query is None:
        _graph_query = GraphQuery()
    return _graph_query
