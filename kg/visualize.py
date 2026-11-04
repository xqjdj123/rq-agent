# -*- coding: utf-8 -*-
"""
知识图谱可视化 — 生成交互式HTML
用法: python kg/visualize.py
输出: eval/graph_visualization.html (全量图) + eval/graph_business.html (业务域分层图)
"""
import json, os, sys
import networkx as nx
from pyvis.network import Network

KG_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(KG_DIR, 'graph_export.json')
OUT_DIR = os.path.join(os.path.dirname(KG_DIR), 'eval')

# 节点颜色方案
COLORS = {
    'Business': '#4CAF50',    # 绿色 — 业务域
    'Intent': '#2196F3',      # 蓝色 — 意图
    'Question': '#FF9800',    # 橙色 — 问题
    'Answer': '#9C27B0',      # 紫色 — 答案
    'Regulation': '#F44336',  # 红色 — 法规
}

def load_graph():
    with open(GRAPH_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    g = nx.MultiDiGraph()
    for node in data['nodes']:
        g.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
    for edge in data['edges']:
        g.add_edge(edge['source'], edge['target'], relation=edge.get('relation', ''))
    return g


def build_high_level_graph(g):
    """构建高层视图：Business → Intent → Regulation（聚合问答节点）"""
    hlg = nx.DiGraph()

    for nid, attrs in g.nodes(data=True):
        ntype = attrs.get('type', '')
        label = attrs.get('label', nid)
        if ntype in ('Business', 'Intent', 'Regulation'):
            hlg.add_node(nid, label=label, type=ntype, color=COLORS.get(ntype, '#999'))

    # 聚合关系：Business → Intent, Intent → Regulation（去重计数）
    edge_weights = {}
    for src, dst, attrs in g.edges(data=True):
        src_type = g.nodes[src].get('type', '')
        dst_type = g.nodes[dst].get('type', '')
        rel = attrs.get('relation', '')

        if src_type == 'Business' and dst_type == 'Intent':
            key = (src, dst)
            edge_weights[key] = edge_weights.get(key, 0) + 1
        elif src_type == 'Intent' and dst_type == 'Answer' and rel == '对应答':
            # Intent → Answer → Regulation 间接关系
            for _, reg, _ in g.out_edges(dst, data=True):
                if g.nodes[reg].get('type') == 'Regulation':
                    key = (src, reg)
                    edge_weights[key] = edge_weights.get(key, 0) + 1

    for (src, dst), weight in edge_weights.items():
        hlg.add_edge(src, dst, weight=weight, title=f'{weight}条')

    return hlg


def build_full_graph(g):
    """构建完整可视化图（含所有节点类型）"""
    fg = nx.DiGraph()
    for nid, attrs in g.nodes(data=True):
        ntype = attrs.get('type', '')
        label = attrs.get('label', nid)
        short_label = label[:25] + '...' if len(label) > 25 else label
        fg.add_node(nid, label=short_label, title=label, type=ntype,
                     color=COLORS.get(ntype, '#999'))
    for src, dst, attrs in g.edges(data=True):
        rel = attrs.get('relation', '')
        fg.add_edge(src, dst, title=rel)
    return fg


def render_pyvis(g, out_path, title, height='750px', physics=True):
    """用pyvis渲染交互式HTML"""
    net = Network(height=height, width='100%', directed=True, notebook=False)
    net.set_options(f'''
    {{
      "nodes": {{"font": {{"size": 14, "face": "Microsoft YaHei, PingFang SC, sans-serif"}}}},
      "edges": {{"arrows": {{"to": {{"enabled": true, "scaleFactor": 0.5}}}}}},
      "physics": {{
        "enabled": {str(physics).lower()},
        "barnesHut": {{"gravitationalConstant": -3000, "centralGravity": 0.3, "springLength": 150}},
        "stabilization": {{"iterations": 200}}
      }},
      "interaction": {{"navigationButtons": true, "keyboard": true}}
    }}
    ''')

    for nid, attrs in g.nodes(data=True):
        net.add_node(nid, label=attrs.get('label', nid),
                     title=attrs.get('title', ''),
                     color=attrs.get('color', '#999'),
                     shape='dot' if attrs.get('type') != 'Business' else 'square',
                     size=20 if attrs.get('type') == 'Business' else 12)

    for src, dst, attrs in g.edges(data=True):
        net.add_edge(src, dst, title=attrs.get('title', ''),
                     width=min(attrs.get('weight', 1), 5))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    net.save_graph(out_path)
    print(f'  Saved: {out_path}')


def main():
    print('Loading graph...')
    g = load_graph()
    print(f'  {g.number_of_nodes()} nodes, {g.number_of_edges()} edges')

    # 1. 高层视图：业务→意图→法规
    print('\nBuilding high-level graph...')
    hlg = build_high_level_graph(g)
    print(f'  High-level: {hlg.number_of_nodes()} nodes, {hlg.number_of_edges()} edges')
    render_pyvis(hlg, os.path.join(OUT_DIR, 'graph_high_level.html'),
                 '知识图谱高层视图 — 业务域→意图→法规', height='700px')

    # 2. 完整视图（采样以控制规模）
    print('\nBuilding full graph (sampled)...')
    fg = build_full_graph(g)
    print(f'  Full: {fg.number_of_nodes()} nodes, {fg.number_of_edges()} edges')
    render_pyvis(fg, os.path.join(OUT_DIR, 'graph_full.html'),
                 '知识图谱全量视图 — 5类实体×4类关系', height='900px')

    # 3. 按业务域拆分
    print('\nBuilding per-business views...')
    os.makedirs(os.path.join(OUT_DIR, 'business'), exist_ok=True)
    for nid, attrs in g.nodes(data=True):
        if attrs.get('type') == 'Business':
            biz_name = attrs.get('label', nid)
            # 提取该业务域的子图
            sub_nodes = {nid}
            # 找下属 Intent
            for _, intent_nid, _ in g.out_edges(nid, data=True):
                if g.nodes[intent_nid].get('type') == 'Intent':
                    sub_nodes.add(intent_nid)
                    # 找下属 Answer
                    for _, ans_nid, _ in g.out_edges(intent_nid, data=True):
                        if g.nodes[ans_nid].get('type') == 'Answer':
                            sub_nodes.add(ans_nid)
                            # 找关联 Regulation
                            for _, reg_nid, _ in g.out_edges(ans_nid, data=True):
                                if g.nodes[reg_nid].get('type') == 'Regulation':
                                    sub_nodes.add(reg_nid)
            sub_g = g.subgraph(sub_nodes)
            sub_fg = build_full_graph(sub_g)
            fname = f'graph_{biz_name}.html'
            render_pyvis(sub_fg, os.path.join(OUT_DIR, 'business', fname),
                         f'知识图谱 — {biz_name}', height='600px')

    print(f'\nDone! Open {os.path.join(OUT_DIR, "graph_high_level.html")} to start.')


if __name__ == '__main__':
    main()
