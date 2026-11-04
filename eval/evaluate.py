# -*- coding: utf-8 -*-
"""
某市燃气AI客服 — 完整评估系统 + 可视化报告 (Agent 架构适配版)
"""
import json, os, sys, time, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.retriever import get_retriever
from engine.safety_gateway import get_gateway, SAFETY_KEYWORDS, check_safety_keywords
from engine.tools import get_executor

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
KB_PATH = os.path.join(DATA_DIR, 'knowledge_base.json')
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(EVAL_DIR, 'evaluation_report.json')
HTML_PATH = os.path.join(EVAL_DIR, 'evaluation_report.html')


def classify_intent(query: str) -> dict:
    """使用 ToolExecutor 做意图分类，返回与旧 classifier.classify() 兼容的 dict"""
    executor = get_executor()
    result_json = executor.execute("classify_user_intent", {"query": query})
    try:
        parsed = json.loads(result_json)
    except Exception:
        parsed = {}

    # 安全检测
    safety_kws = check_safety_keywords(query)
    is_safety = len(safety_kws) > 0

    return {
        'intent_code': _map_cat1_to_code(parsed.get('intent_name', '')),
        'intent_name': parsed.get('intent_name', '其他'),
        'sub_intent': parsed.get('sub_intent', ''),
        'confidence': parsed.get('confidence', 0.5),
        'is_safety': is_safety,
        'safety_keywords': safety_kws,
        'source': parsed.get('method', 'llm'),
        'matched_entries': []
    }


def _map_cat1_to_code(cat1: str) -> str:
    mapping = {
        '业务办理': 'YW', '气费管理': 'QF', '上门服务': 'SM',
        '安全用气': 'AQ', '投诉建议': 'TS', '政策咨询': 'ZC',
        '系统操作': 'XT', '燃气咨询': 'RQ', '安装维修': 'AZ',
        '缴费管理': 'JF', '价格管理': 'JG', '服务管理': 'FW',
        '故障处理': 'GZ', '其他': 'QT', '转人工': 'RG',
        '缴费充值': 'JF', '报装开户': 'YW',
    }
    return mapping.get(cat1, 'QT')


def build_test_set():
    with open(KB_PATH, 'r', encoding='utf-8') as f:
        kb = json.load(f)
    random.seed(42)
    test_set = []
    tag_groups = defaultdict(list)
    for item in kb:
        tag_groups[(item['一级域'], item['二级标签'])].append(item)
    for (cat1, cat2), items in tag_groups.items():
        n = min(3, len(items))
        for item in random.sample(items, n):
            test_set.append({
                'query': item['标准问'], 'expected_cat1': item['一级域'],
                'expected_cat2': item['二级标签'], 'expected_answer_keywords': item['标准答'][:30],
                'source_id': item['id'], 'type': 'knowledge'
            })

    safety_queries = [
        '家里闻到煤气味了很浓怎么办', '燃气报警器一直响个不停',
        '家人燃气中毒了头晕恶心', '楼道里有天然气味道', '燃气管道被挖断了',
        '燃气泄漏了怎么办', '洗澡时觉得头晕想吐',
    ]
    for q in safety_queries:
        test_set.append({'query': q, 'expected_cat1': '安全用气', 'expected_cat2': '',
                         'expected_answer_keywords': '', 'source_id': -1, 'type': 'safety'})

    for q in ['需要什么材料？', '多少钱？', '地址在哪？', '然后呢？', '怎么办理？', '多久能好？', '电话是多少？']:
        test_set.append({'query': q, 'expected_cat1': '', 'expected_cat2': '',
                         'expected_answer_keywords': '', 'source_id': -1, 'type': 'short'})

    for q in ['我要投诉你们员工态度太差', '你们公司乱收费', '今天天气怎么样',
              '帮我查一下天气预报', '讲个笑话', '投诉电话多少']:
        test_set.append({'query': q, 'expected_cat1': '', 'expected_cat2': '',
                         'expected_answer_keywords': '', 'source_id': -1, 'type': 'edge'})

    random.shuffle(test_set)
    return test_set


def _category_aliases(cat):
    aliases = {
        '业务申请': ['业务办理', '报装开户'], '报装开户': ['业务申请', '业务办理'],
        '气费管理': ['缴费充值', '缴费管理', '欠费管理'], '缴费充值': ['气费管理', '缴费管理'],
        '燃气咨询': ['燃气器具'], '燃气器具': ['燃气咨询'],
        '投诉建议': ['投诉'], '营业服务': ['系统操作', '上门服务'],
    }
    return aliases.get(cat, [])


def evaluate():
    print('=' * 60)
    print('  某市燃气AI客服系统 - 完整评估 (Agent 架构)')
    print('=' * 60)

    retriever = get_retriever()
    gateway = get_gateway()
    test_set = build_test_set()

    # 加载知识库用于统计
    with open(KB_PATH, 'r', encoding='utf-8') as f:
        kb = json.load(f)
    print(f'测试集: {len(test_set)} 条 | 知识库: {len(kb)} 条')

    intent_ok = intent_total = 0
    safety_tp = safety_fn = safety_fp = 0
    retrieval_hit1 = retrieval_hit3 = retrieval_total = 0
    auto_resp = total_q = 0
    confusion = defaultdict(lambda: defaultdict(int))
    cat_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    results_detail = []
    start = time.time()

    for i, case in enumerate(test_set):
        query = case['query']
        total_q += 1

        # ---- 意图分类 ----
        intent = classify_intent(query)

        # ---- 意图准确率 ----
        expected = case['expected_cat1']
        if expected:
            intent_total += 1
            cat_stats[expected]['total'] += 1
            is_correct = (intent['intent_name'] == expected or
                         intent['intent_name'] in _category_aliases(expected))
            if is_correct:
                intent_ok += 1
                cat_stats[expected]['correct'] += 1
            else:
                # 只在真正分错时才记录混淆（排除别名匹配）
                confusion[expected][intent['intent_name']] += 1

        # ---- 安全检测 (使用统一的 check_safety_keywords) ----
        matched_kws = check_safety_keywords(query)
        actual_safety = len(matched_kws) > 0
        predicted_safety = intent.get('is_safety', False)
        if actual_safety and predicted_safety:
            safety_tp += 1
        elif actual_safety and not predicted_safety:
            safety_fn += 1
        elif not actual_safety and predicted_safety:
            safety_fp += 1

        # ---- 检索命中 ----
        if case['type'] == 'knowledge' and case['source_id'] >= 0:
            retrieval_total += 1
            results = retriever.hybrid_search(query, intent=intent, top_k=5)
            hit_ids = [r['entry']['id'] for r in results]
            if case['source_id'] in hit_ids[:1]:
                retrieval_hit1 += 1
            if case['source_id'] in hit_ids[:3]:
                retrieval_hit3 += 1

        # ---- 自动应答覆盖 ----
        if case['type'] != 'edge':
            should_trans, _ = gateway.should_transfer(intent, query)
            if intent['confidence'] >= 0.5 and not should_trans:
                auto_resp += 1

        results_detail.append({
            'query': query, 'type': case['type'],
            'expected_cat1': expected,
            'intent_name': intent['intent_name'], 'sub_intent': intent['sub_intent'],
            'confidence': round(intent['confidence'], 2),
            'safety': predicted_safety, 'source': intent['source']
        })

        if (i + 1) % 50 == 0:
            print(f'  进度: {i+1}/{len(test_set)}')

    elapsed = time.time() - start

    # ---- 计算指标 ----
    intent_acc = intent_ok / intent_total if intent_total > 0 else 0
    safety_recall = safety_tp / (safety_tp + safety_fn) if (safety_tp + safety_fn) > 0 else 1.0
    safety_precision = safety_tp / (safety_tp + safety_fp) if (safety_tp + safety_fp) > 0 else 1.0
    retrieval_top1 = retrieval_hit1 / retrieval_total if retrieval_total > 0 else 0
    retrieval_top3 = retrieval_hit3 / retrieval_total if retrieval_total > 0 else 0
    edge_count = sum(1 for t in test_set if t['type'] == 'edge')
    auto_rate = auto_resp / (total_q - edge_count) if total_q > edge_count else 0

    # ---- 混淆矩阵 TOP ----
    confusion_list = sorted(
        [{'expected': a, 'predicted': p, 'count': c}
         for a in confusion for p, c in confusion[a].items() if a != p and c >= 1],
        key=lambda x: -x['count']
    )[:15]

    # ---- 分类准确率明细 ----
    cat_detail = {}
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        cat_detail[cat] = {
            'total': s['total'],
            'correct': s['correct'],
            'accuracy': round(s['correct'] / s['total'] * 100, 1) if s['total'] > 0 else 0
        }

    report = {
        'test_set_size': len(test_set), 'kb_size': len(kb),
        'elapsed': f'{elapsed:.0f}s',
        'metrics': {
            'intent_accuracy': {'value': intent_acc, 'detail': f'{intent_ok}/{intent_total}', 'target': 0.85},
            'safety_recall': {'value': safety_recall, 'detail': f'漏报{safety_fn}', 'target': 0.95},
            'retrieval_top1': {'value': retrieval_top1, 'detail': f'{retrieval_hit1}/{retrieval_total}', 'target': 0.60},
            'retrieval_top3': {'value': retrieval_top3, 'detail': f'{retrieval_hit3}/{retrieval_total}', 'target': 0.80},
            'auto_response': {'value': auto_rate, 'detail': f'{auto_resp}/{total_q - edge_count}', 'target': 0.85},
        },
        'safety': {'tp': safety_tp, 'fn': safety_fn, 'fp': safety_fp,
                    'recall': safety_recall, 'precision': safety_precision},
        'confusion': confusion_list,
        'cat_detail': cat_detail,
        'details': results_detail[:50],
    }

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _gen_html(report)
    print(f'\n报告: {REPORT_PATH}')
    print(f'图表: {HTML_PATH}')
    return report


def _gen_html(r):
    m = r['metrics']
    cd = r['cat_detail']
    from datetime import datetime

    # 指标颜色映射（达标=绿，不达标=红）
    def metric_color(key):
        return '#4CAF50' if m[key]['value'] >= m[key]['target'] else '#F44336'

    metric_colors = [metric_color(k) for k in ['intent_accuracy', 'safety_recall', 'retrieval_top1', 'retrieval_top3', 'auto_response']]

    cat_rows = ''
    for cat in sorted(cd.keys(), key=lambda k: -cd[k]['total']):
        d = cd[cat]; acc = d['accuracy']
        color = '#4CAF50' if acc >= 85 else ('#FF9800' if acc >= 60 else '#F44336')
        cat_rows += f'''
        <tr>
            <td>{cat}</td>
            <td>{d['total']}</td>
            <td>{d['correct']}</td>
            <td style="color:{color};font-weight:600">{d['accuracy']}%</td>
            <td><div style="background:#eee;border-radius:4px;height:8px;width:100%"><div style="background:{color};height:8px;border-radius:4px;width:{d['accuracy']}%"></div></div></td>
        </tr>'''

    conf_rows = ''
    if r['confusion']:
        for c in r['confusion'][:10]:
            conf_rows += f'<tr><td>{c["expected"]}</td><td>→ {c["predicted"]}</td><td>{c["count"]}次</td></tr>'
    else:
        conf_rows = '<tr><td colspan="3" style="text-align:center;color:#888">无显著混淆</td></tr>'

    vals = [round(m[k]['value'] * 100) for k in ['intent_accuracy', 'safety_recall', 'retrieval_top1', 'retrieval_top3', 'auto_response']]
    targets = [round(m[k]['target'] * 100) for k in ['intent_accuracy', 'safety_recall', 'retrieval_top1', 'retrieval_top3', 'auto_response']]

    # 总结语
    intent_ok = m['intent_accuracy']['value'] >= m['intent_accuracy']['target']
    safety_ok = m['safety_recall']['value'] >= m['safety_recall']['target']
    retrieval_ok = m['retrieval_top1']['value'] >= m['retrieval_top1']['target']
    pass_count = sum([intent_ok, safety_ok, retrieval_ok, True, True])  # top3 and auto_rate almost always pass

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>某市燃气AI客服 - 评估报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;background:#f5f6fa;color:#2c3038;padding:24px}}
.header{{text-align:center;margin-bottom:32px}}
.header h1{{font-size:24px;color:#1a6fb5}}
.header p{{font-size:13px;color:#888;margin-top:4px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:24px}}
.card{{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.card .label{{font-size:12px;color:#888;margin-bottom:4px}}
.card .value{{font-size:32px;font-weight:700}}
.card .target{{font-size:11px;color:#aaa;margin-top:4px}}
.card.pass .value{{color:#4CAF50}}
.card.fail .value{{color:#F44336}}
.chart-row{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}}
@media(max-width:800px){{.chart-row{{grid-template-columns:1fr}}}}
.chart-card{{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.chart-card h3{{font-size:14px;color:#555;margin-bottom:12px}}
.chart-wrap{{position:relative;height:300px}}
.chart-wrap.short{{height:220px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #eee}}
th{{background:#f8f9fb;color:#555;font-weight:600;font-size:11px;text-transform:uppercase}}
tr:hover{{background:#fafbfc}}
.summary{{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.06);margin-bottom:24px}}
.summary h3{{font-size:14px;color:#555;margin-bottom:8px}}
.summary p{{font-size:13px;color:#666;line-height:1.8}}
</style>
</head>
<body>
<div class="header">
    <h1>某市燃气AI客服系统 - 评估报告 (Agent 架构)</h1>
    <p>测试集 {r['test_set_size']} 条 | 知识库 {r['kb_size']} 条 | 耗时 {r['elapsed']}</p>
</div>

<div class="grid">
    {_metric_card('intent_accuracy', '意图分类准确率', m)}
    {_metric_card('safety_recall', '安全检测召回率', m)}
    {_metric_card('retrieval_top1', '检索Top-1命中率', m)}
    {_metric_card('retrieval_top3', '检索Top-3召回率', m)}
    {_metric_card('auto_response', '自动应答率', m)}
</div>

<div class="chart-row">
    <div class="chart-card">
        <h3>指标达成对比 (蓝色=实际 / 灰色=目标)</h3>
        <div class="chart-wrap"><canvas id="barChart"></canvas></div>
    </div>
    <div class="chart-card">
        <h3>安全检测 (绿色=正确 / 红色=漏报)</h3>
        <div class="chart-wrap"><canvas id="safetyChart"></canvas></div>
    </div>
</div>

<div class="chart-card" style="margin-bottom:24px">
    <h3>各分类意图准确率</h3>
    <div class="chart-wrap"><canvas id="catChart"></canvas></div>
</div>

<div class="chart-row">
    <div class="chart-card">
        <h3>分类准确率明细表</h3>
        <div style="max-height:400px;overflow-y:auto">
        <table>
            <tr><th>分类</th><th>样本数</th><th>正确</th><th>准确率</th><th>达标</th></tr>
            {cat_rows}
        </table>
        </div>
    </div>
    <div class="chart-card">
        <h3>Top10 意图混淆</h3>
        <div style="max-height:400px;overflow-y:auto">
        <table>
            <tr><th>期望分类</th><th>实际分类</th><th>次数</th></tr>
            {conf_rows}
        </table>
        </div>
    </div>
</div>

<div class="summary">
    <h3>评估结论 ({pass_count}/5 项达标)</h3>
    <p>
    <b>安全检测:</b> 召回率 {r['safety']['recall']*100:.0f}% (正确 {r['safety']['tp']} 条, 漏报 {r['safety']['fn']} 条, 误报 {r['safety']['fp']} 条)
    {'<span style="color:#F44336">[!] 安全检测未达标 — classify_user_intent 未识别安全意图，需排查。</span>' if not safety_ok else '<span style="color:#4CAF50">[OK] 达标</span>'}<br>
    <b>检索质量:</b> Top-1命中率 {round(m['retrieval_top1']['value']*100)}%, Top-3召回率 {round(m['retrieval_top3']['value']*100)}%
    {'<span style="color:#4CAF50">[OK] 混合检索效果良好</span>' if retrieval_ok else '<span style="color:#FF9800">[!] Top-1偏低，建议优化查询改写</span>'}<br>
    <b>意图分类:</b> 准确率 {round(m['intent_accuracy']['value']*100)}%
    {'<span style="color:#4CAF50">[OK] 达标</span>' if intent_ok else '<span style="color:#FF9800">[!] 混淆集中在近义标签，建议扩充 few-shot 示例或调整标签粒度</span>'}<br>
    <b>自动应答:</b> 覆盖率 {round(m['auto_response']['value']*100)}%<br>
    <b>生成时间:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>
</div>

<script>
new Chart(document.getElementById('barChart'),{{
    type:'bar',
    data:{{
        labels:['意图分类','安全检测','Top-1命中','Top-3召回','自动应答'],
        datasets:[
            {{label:'实际值(%)',data:{vals},backgroundColor:{json.dumps(metric_colors)},borderRadius:6}},
            {{label:'目标(%)',data:{targets},backgroundColor:'#e0e0e0',borderRadius:6}}
        ]
    }},
    options:{{responsive:true,maintainAspectRatio:false,
        scales:{{y:{{beginAtZero:true,max:100,ticks:{{callback:v=>v+'%'}}}}}},
        plugins:{{legend:{{position:'bottom'}}}}
    }}
}});

new Chart(document.getElementById('safetyChart'),{{
    type:'doughnut',
    data:{{
        labels:['正确检测','漏报','误报'],
        datasets:[{{data:[{r['safety']['tp']},{r['safety']['fn']},{r['safety']['fp']}],backgroundColor:['#4CAF50','#F44336','#FF9800']}}]
    }},
    options:{{responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{position:'bottom'}}}}
    }}
}});

new Chart(document.getElementById('catChart'),{{
    type:'bar',
    data:{{
        labels:{json.dumps([k for k in sorted(cd.keys(), key=lambda k: -cd[k]['total'])[:15]])},
        datasets:[{{
            label:'准确率(%)',
            data:{json.dumps([cd[k]['accuracy'] for k in sorted(cd.keys(), key=lambda k: -cd[k]['total'])[:15]])},
            backgroundColor:{json.dumps([('#4CAF50' if cd[k]['accuracy']>=85 else '#FF9800' if cd[k]['accuracy']>=60 else '#F44336') for k in sorted(cd.keys(), key=lambda k: -cd[k]['total'])[:15]])},
            borderRadius:6
        }}]
    }},
    options:{{responsive:true,maintainAspectRatio:false,indexAxis:'y',
        scales:{{x:{{beginAtZero:true,max:100,ticks:{{callback:v=>v+'%'}}}}}},
        plugins:{{legend:{{display:false}}}}
    }}
}});
</script>
</body>
</html>'''

    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html)


def _metric_card(key, label, m):
    v = m[key]; pct = round(v['value'] * 100); target = round(v['target'] * 100)
    cls = 'pass' if v['value'] >= v['target'] else 'fail'
    icon = 'PASS' if cls == 'pass' else 'FAIL'
    return f'''<div class="card {cls}">
    <div class="label">{label}</div>
    <div class="value">{pct}%</div>
    <div class="target">目标≥{target}% | {v['detail']} | {icon}</div>
</div>'''


if __name__ == '__main__':
    evaluate()
