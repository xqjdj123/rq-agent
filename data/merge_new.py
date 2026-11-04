"""
合并新数据到知识库
将 某市燃气业务_补全表头_新增缴费途径.xlsx 合并入现有knowledge_base.json
"""
import json
import os
import openpyxl

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(DATA_DIR, 'knowledge_base.json')
NEW_XLSX = os.path.join(DATA_DIR, "某市燃气业务_补全表头_新增缴费途径.xlsx")

# 新数据字段映射: (新列名, 旧字段名)
# Col1:编号  Col2:一级分类  Col3:二级分类  Col4:内容类型  Col5:问题  Col6:回答
# Col7:关键词  Col8:来源单位  Col9:来源链接  Col10:更新时间
# 目标: 编码, 一级域, 二级标签, 标签定义, 典型问句示例, 标准化回复, 关联法规FG编码, 安全敏感

def load_new_data():
    """加载新Excel"""
    wb = openpyxl.load_workbook(NEW_XLSX)
    ws = wb.active
    new_entries = []

    for row in range(2, ws.max_row + 1):
        cat1 = str(ws.cell(row=row, column=2).value or '').strip()
        cat2 = str(ws.cell(row=row, column=3).value or '').strip()
        question = str(ws.cell(row=row, column=5).value or '').strip()
        answer = str(ws.cell(row=row, column=6).value or '').strip()
        keywords = str(ws.cell(row=row, column=7).value or '').strip()
        source = str(ws.cell(row=row, column=8).value or '').strip()
        source_url = str(ws.cell(row=row, column=9).value or '').strip()
        update_time = str(ws.cell(row=row, column=10).value or '').strip()
        orig_id = str(ws.cell(row=row, column=1).value or '').strip()

        if not question or not answer:
            continue

        new_entries.append({
            '一级域': cat1,
            '二级标签': cat2,
            '问题': question,
            '回答': answer,
            '关键词': keywords,
            '来源单位': source,
            '来源链接': source_url,
            '更新时间': update_time,
            '原始编号': orig_id
        })

    return new_entries

def merge_to_kb(new_entries):
    """合并到现有知识库"""
    # 加载现有知识库
    with open(KB_PATH, 'r', encoding='utf-8') as f:
        kb = json.load(f)

    print(f"现有知识库: {len(kb)} 条")
    print(f"新数据: {len(new_entries)} 条")

    # 一级域映射（新→标准）
    # 新文件的某些分类需要映射到已有分类
    cat1_map = {
        '业务申请': '业务申请',
        '安全用气': '安全用气',
        '上门服务': '上门服务',
        '政策咨询': '政策咨询',
        '系统操作': '系统操作',
        '气费管理': '气费管理',
        '缴费充值': '气费管理',  # 新分类→归入气费管理
    }

    next_id = max(item['id'] for item in kb) + 1

    # 生成新FG编码（新数据没有法规编码，用 FG-CZ-002 表示企业自有操作规程）
    fg_default = 'FG-CZ-002'

    added = 0
    skipped = 0

    for entry in new_entries:
        question = entry['问题']
        answer = entry['回答']
        cat1 = cat1_map.get(entry['一级域'], entry['一级域'])
        cat2 = entry['二级标签']

        # 去重：检查是否已有高度相似的问题
        is_dup = False
        for item in kb:
            # 简单判断：问题文本完全一致
            if item['标准问'] == question:
                print(f"  跳过重复: {question[:50]}...")
                skipped += 1
                is_dup = True
                break

        if is_dup:
            continue

        # 构建标签定义（从回答中提取摘要）
        tag_def = f"用户咨询{cat2}相关问题"

        # 判断安全敏感
        is_safety = '安全用气' in cat1 and any(
            kw in question + answer
            for kw in ['泄漏', '漏气', '中毒', '火灾', '爆炸', '事故', '紧急']
        )

        # 提取关键词
        try:
            import jieba
            words = jieba.lcut(question + ' ' + cat2)
            stopwords = {'的', '了', '是', '我', '有', '在', '不', '人', '都',
                         '什么', '怎么', '如何', '哪', '哪些', '请问', '吗', '？', '？'}
            kw = ' '.join([w for w in words if len(w) >= 2 and w not in stopwords][:10])
        except:
            kw = entry.get('关键词', '')

        new_item = {
            'id': next_id,
            '编码': f'FG-CZ-002-{next_id:03d}',
            '一级域': cat1,
            '二级标签': cat2,
            '标签定义': tag_def,
            '标准问': question,
            '标准答': answer,
            '法规编码': fg_default,
            '安全敏感': is_safety,
            '关键词': kw,
            '来源单位': entry.get('来源单位', ''),
            '来源链接': entry.get('来源链接', ''),
            '更新时间': entry.get('更新时间', ''),
            '原始编号': entry.get('原始编号', '')
        }

        kb.append(new_item)
        next_id += 1
        added += 1
        print(f"  添加 [{cat1}>{cat2}] {question[:50]}...")

    # 保存
    with open(KB_PATH, 'w', encoding='utf-8') as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

    print(f"\n合并完成: 新增 {added} 条, 跳过 {skipped} 条重复, 总计 {len(kb)} 条")
    return kb

def main():
    entries = load_new_data()
    print(f"加载新数据: {len(entries)} 条")
    print("\n分类分布:")
    cats = {}
    for e in entries:
        k = f"{e['一级域']}>{e['二级标签']}"
        cats[k] = cats.get(k, 0) + 1
    for k, v in sorted(cats.items()):
        print(f"  {k}: {v}")

    kb = merge_to_kb(entries)

    # 统计最终结果
    cat_count = {}
    for item in kb:
        cat_count[item['一级域']] = cat_count.get(item['一级域'], 0) + 1
    print(f"\n最终知识库 - 一级域分布:")
    for k, v in sorted(cat_count.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
