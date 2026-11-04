"""
数据预处理与知识库构建
读取数据.xls，生成：
1. 标签体系表.xlsx
2. 应答依据表.xlsx
3. knowledge_base.json (结构化知识库)
"""
import xlrd
import json
import os
from collections import defaultdict

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
XLS_PATH = os.path.join(DATA_DIR, "数据.xls")

def load_data():
    """加载Excel数据"""
    wb = xlrd.open_workbook(XLS_PATH)
    ws = wb.sheet_by_name('Sheet1')

    headers = [ws.cell_value(0, i) for i in range(ws.ncols)]
    print(f"Headers: {headers}")
    print(f"Total rows: {ws.nrows - 1}")

    data = []
    for row in range(1, ws.nrows):
        record = {}
        for col in range(ws.ncols):
            record[headers[col]] = str(ws.cell_value(row, col)).strip()
        data.append(record)
    return data, headers

def build_tag_system(data):
    """构建标签体系（一级域 -> 二级标签 -> 标签定义）"""
    tag_tree = defaultdict(lambda: defaultdict(list))

    for r in data:
        cat1 = r['一级域']
        cat2 = r['二级标签']
        tag_tree[cat1][cat2].append({
            '编码': r['编码'],
            '标签定义': r['标签定义'],
            '典型问句': r['典型问句示例'],
            '标准化回复': r['标准化回复'][:100] + '...',
            '关联法规': r['关联法规FG编码'],
            '安全敏感': r['安全敏感'] == '是'
        })

    return tag_tree

def build_regulation_table(data):
    """构建应答依据表（FG编码 -> 法规条款）"""
    reg_map = {}

    # FG编码层级映射
    LEVEL_MAP = {
        'FL': '法律',
        'XZ': '行政法规',
        'BM': '部门规章',
        'DF': '地方规范',
        'HY': '行业规范',
        'CZ': '操作规范'
    }

    for r in data:
        fg_code = r['关联法规FG编码']
        if fg_code not in reg_map:
            # 从编码推断层级
            parts = fg_code.split('-')
            level_code = parts[1] if len(parts) >= 2 else '??'
            level_name = LEVEL_MAP.get(level_code, '其他')

            reg_map[fg_code] = {
                '依据编码': fg_code,
                '层级代码': level_code,
                '层级名称': level_name,
                '关联标签': set(),
                '相关条目数': 0
            }
        reg_map[fg_code]['关联标签'].add(r['二级标签'])
        reg_map[fg_code]['相关条目数'] += 1

    # 转换set为list
    for v in reg_map.values():
        v['关联标签'] = '、'.join(sorted(v['关联标签']))

    return sorted(reg_map.values(), key=lambda x: x['依据编码'])

def build_knowledge_base(data):
    """构建结构化知识库JSON"""
    kb = []
    for i, r in enumerate(data):
        entry = {
            'id': i,
            '编码': r['编码'],
            '一级域': r['一级域'],
            '二级标签': r['二级标签'],
            '标签定义': r['标签定义'],
            '标准问': r['典型问句示例'],
            '标准答': r['标准化回复'],
            '法规编码': r['关联法规FG编码'],
            '安全敏感': r['安全敏感'] == '是',
            '关键词': extract_keywords(r['典型问句示例'] + r['标签定义'])
        }
        kb.append(entry)
    return kb

def extract_keywords(text):
    """简单关键词提取"""
    try:
        import jieba
        words = jieba.lcut(text)
        # 过滤停用词和单字
        stopwords = {'的', '了', '是', '我', '有', '在', '不', '人', '都', '一', '一个',
                     '什么', '怎么', '怎么办', '为什么', '如何', '哪', '哪些', '请问', '吗'}
        keywords = [w for w in words if len(w) >= 2 and w not in stopwords]
        return ' '.join(keywords[:10])
    except:
        return text

def export_tag_system_excel(tag_tree):
    """导出标签体系Excel"""
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "标签体系"
        ws.append(['一级域', '二级标签', '标签编码示例', '标签定义', '典型问句', '标准化回复(摘要)', '关联法规编码', '安全敏感'])

        for cat1 in sorted(tag_tree.keys()):
            for cat2 in sorted(tag_tree[cat1].keys()):
                items = tag_tree[cat1][cat2]
                for item in items:
                    ws.append([cat1, cat2, item['编码'], item['标签定义'],
                              item['典型问句'], item['标准化回复'], item['关联法规'],
                              '是' if item['安全敏感'] else ''])

        path = os.path.join(DATA_DIR, '标签体系表.xlsx')
        wb.save(path)
        print(f"标签体系表已保存: {path}")
    except Exception as e:
        print(f"导出标签体系Excel失败: {e}")

def export_regulation_excel(reg_table):
    """导出应答依据Excel"""
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "应答依据"
        ws.append(['依据编码', '层级代码', '层级名称', '关联标签', '相关条目数', '生效状态'])

        for r in reg_table:
            ws.append([r['依据编码'], r['层级代码'], r['层级名称'],
                      r['关联标签'], r['相关条目数'], '现行有效'])

        path = os.path.join(DATA_DIR, '应答依据表.xlsx')
        wb.save(path)
        print(f"应答依据表已保存: {path}")
    except Exception as e:
        print(f"导出应答依据Excel失败: {e}")

def main():
    print("=" * 50)
    print("某市燃气AI客服 - 数据预处理")
    print("=" * 50)

    # 加载数据
    data, headers = load_data()
    print(f"\n加载 {len(data)} 条数据记录")

    # 构建标签体系
    tag_tree = build_tag_system(data)
    print(f"\n标签体系: {len(tag_tree)} 个一级域")
    for cat1, cat2s in sorted(tag_tree.items()):
        print(f"  [{cat1}] {len(cat2s)} 个二级标签: {', '.join(sorted(cat2s.keys()))}")

    # 构建应答依据表
    reg_table = build_regulation_table(data)
    print(f"\n应答依据: {len(reg_table)} 个唯一FG编码")
    for r in reg_table:
        print(f"  {r['依据编码']} [{r['层级名称']}] - {r['相关条目数']}条")

    # 构建知识库
    kb = build_knowledge_base(data)
    kb_path = os.path.join(DATA_DIR, 'knowledge_base.json')
    with open(kb_path, 'w', encoding='utf-8') as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    print(f"\n知识库已保存: {kb_path} ({len(kb)} 条)")

    # 导出Excel
    export_tag_system_excel(tag_tree)
    export_regulation_excel(reg_table)

    # 统计
    print(f"\n=== 数据统计 ===")
    print(f"总条目数: {len(data)}")
    print(f"一级域数: {len(tag_tree)}")
    print(f"二级标签数: {sum(len(v) for v in tag_tree.values())}")
    print(f"法规编码数: {len(reg_table)}")
    print(f"安全敏感条目: {sum(1 for r in data if r['安全敏感'] == '是')}")

    return data, tag_tree, reg_table, kb

if __name__ == '__main__':
    main()
