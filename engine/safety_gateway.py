"""
安全网关模块
安全类意图强制模板 + 转人工路由
"""
import json
import os

# 安全关键词 — 命中直接走强制安全模板，不经过 Agent
# 方案文档要求：安全类问题严禁 LLM 自由发挥
SAFETY_KEYWORDS = [
    '漏气', '煤气泄漏', '燃气泄漏', '天然气泄漏', '闻到煤气味', '煤气味', '臭鸡蛋味',
    '燃气异味', '报警器响', '报警器报警', '燃气报警', '着火', '起火', '爆炸',
    '燃气中毒', '一氧化碳中毒', '中毒', '抢修', '紧急',
    '管破了', '管道破裂', '燃气管道破损', '挖断管道'
]

# 否定语境 — 当这些词出现在问句中时，"着火"等词不触发安全
SAFETY_NEGATION_PATTERNS = [
    '打不着', '点不着', '不着火', '打不燃', '点不燃',
    '熄火', '打不开', '开不了', '打不着火', '点不着火',
]


def check_safety_keywords(query: str) -> list:
    """检查安全关键词，排除否定语境（如"打不着火"不触发"着火"）"""
    # 检查否定语境
    has_negation = any(p in query for p in SAFETY_NEGATION_PATTERNS)

    matched = []
    for kw in SAFETY_KEYWORDS:
        if kw in query:
            # "着火"在否定语境中跳过（如"打不着火""点不着火"）
            if kw == '着火' and has_negation:
                continue
            matched.append(kw)
    return matched

# 安全应答强制模板（不可被LLM改写）
SAFETY_TEMPLATES = {
    '疑似漏气处置': {
        'title': '⚠️ 燃气安全紧急处置',
        'steps': [
            '① 立即关闭燃气总阀门',
            '② 打开所有门窗，保持通风',
            '③ 切勿开关任何电器（包括电灯、排风扇、抽油烟机）',
            '④ 不要使用明火、打火机、手机在现场拨打',
            '⑤ 迅速撤离到室外安全区域',
            '⑥ 在室外拨打24小时抢修专线：400-000-0000',
            '⑦ 在专业人员到达前，不要返回室内'
        ],
        'warnings': [
            '🚫 禁止开关电器',
            '🚫 禁止使用明火',
            '🚫 禁止在现场拨打电话',
            '🚫 禁止启动车辆',
        ],
        'transfer': True,
        'transfer_msg': '已为您优先接入人工客服和抢修调度，请保持电话畅通。'
    },
    '报警器报警': {
        'title': '⚠️ 燃气报警器报警处置',
        'steps': [
            '① 不要慌张，立即关闭燃气总阀',
            '② 打开门窗通风',
            '③ 不要开关任何电器',
            '④ 撤离到室外安全区域',
            '⑤ 拨打24小时抢修专线：400-000-0000',
            '⑥ 等待专业人员检查，确定原因后方可恢复用气'
        ],
        'warnings': [
            '🚫 不要自行拆卸报警器',
            '🚫 不要在未查明原因前恢复用气',
        ],
        'transfer': True,
        'transfer_msg': '已为您转接抢修专线，请保持电话畅通。'
    },
    '燃气中毒': {
        'title': '⚠️ 燃气中毒紧急处置',
        'steps': [
            '① 立即打开门窗通风',
            '② 将中毒者移至空气新鲜处',
            '③ 解开衣领、裤带，保持呼吸道通畅',
            '④ 注意保暖，防止受凉',
            '⑤ 立即拨打120急救电话',
            '⑥ 同时拨打抢修专线：400-000-0000',
            '⑦ 如有呼吸心跳停止，立即进行心肺复苏'
        ],
        'warnings': [
            '🚨 生命危险！立即行动！',
            '🚫 不要在没有防护的情况下进入高浓度泄漏区域',
        ],
        'transfer': True,
        'transfer_msg': '已同时通知120急救和燃气抢修，请保持电话畅通。'
    },
    '火灾爆炸': {
        'title': '⚠️ 燃气火灾/爆炸紧急处置',
        'steps': [
            '① 立即关闭燃气总阀（如可安全接近）',
            '② 立即拨打119火警',
            '③ 拨打抢修专线：400-000-0000',
            '④ 迅速疏散，远离事故现场',
            '⑤ 不要乘坐电梯',
            '⑥ 不要返回现场取财物'
        ],
        'warnings': [
            '🚨 生命第一！立即疏散！',
            '🚫 不要试图自行灭火（除非火势很小且你有把握）',
            '🚫 不要乘坐电梯',
        ],
        'transfer': True,
        'transfer_msg': '已同时通知119和燃气抢修，请立即撤离到安全区域。'
    }
}

# 转人工条件
TRANSFER_CONDITIONS = {
    'low_confidence': 0.5,     # 置信度低于0.5转人工
    'clarify_range': (0.5, 0.8),  # 0.5-0.8需要澄清追问
    'emotion_keywords': ['投诉', '举报', '赔偿', '起诉', '媒体', '曝光', '12345',
                         '态度差', '乱收费', '不作为', '敷衍', '推诿'],
    'complex_keywords': ['法律', '律师', '法院', '诉讼', '判决', '强制执行'],
}


class SafetyGateway:
    """安全网关：安全模板 + 转人工判断"""

    def is_safety_intent(self, intent_result: dict) -> bool:
        """判断是否为安全类意图"""
        return intent_result.get('is_safety', False)

    def get_safety_template(self, matched_keywords: list) -> dict:
        """根据关键词匹配最佳安全模板"""
        kw_str = ' '.join(matched_keywords)

        if any(k in kw_str for k in ['中毒', '一氧化碳', '窒息']):
            return SAFETY_TEMPLATES['燃气中毒']
        if any(k in kw_str for k in ['着火', '起火', '爆炸', '火灾']):
            return SAFETY_TEMPLATES['火灾爆炸']
        if any(k in kw_str for k in ['报警器', '报警']):
            return SAFETY_TEMPLATES['报警器报警']
        # 默认漏气处置
        return SAFETY_TEMPLATES['疑似漏气处置']

    def format_safety_response(self, template: dict) -> str:
        """格式化安全应答"""
        lines = [template['title'], '']
        for step in template['steps']:
            lines.append(step)
        lines.append('')
        lines.append('【重要提醒】')
        for warning in template['warnings']:
            lines.append(warning)
        if template.get('transfer_msg'):
            lines.append('')
            lines.append(f"📞 {template['transfer_msg']}")
        return '\n'.join(lines)

    def should_transfer(self, intent_result: dict, query: str) -> tuple:
        """
        判断是否需要转人工
        返回 (should_transfer, reason)
        """
        # 安全类必须转人工
        if self.is_safety_intent(intent_result):
            return True, '安全紧急事件，强制转人工'

        # 低置信度转人工
        conf = intent_result.get('confidence', 0)
        if conf < TRANSFER_CONDITIONS['low_confidence']:
            return True, f'意图置信度过低({conf:.0%})，建议转人工'

        # 检查情绪/投诉关键词
        for kw in TRANSFER_CONDITIONS['emotion_keywords']:
            if kw in query:
                return True, f'检测到投诉/情绪关键词"{kw}"，建议转人工'

        # 检查法律相关关键词
        for kw in TRANSFER_CONDITIONS['complex_keywords']:
            if kw in query:
                return True, f'涉及法律事务，建议转人工'

        return False, ''

    def get_clarification_question(self, intent_result: dict) -> str:
        """生成澄清追问（置信度0.5-0.8时）"""
        cat1 = intent_result.get('intent_name', '')
        cat2 = intent_result.get('sub_intent', '')

        if cat2:
            return f"您是想咨询【{cat1} - {cat2}】相关的问题吗？请再详细描述一下您的需求，我可以给您更准确的解答。"
        elif cat1:
            return f"您的问题可能和【{cat1}】相关。请问具体是关于{cat1}的哪个方面呢？比如报装、缴费、安检还是其他？"
        else:
            return "抱歉，我没有完全理解您的问题。能否再详细描述一下？或者您可以直接拨打人工客服 400-000-0001 咨询。"


# 单例
_gateway_instance = None

def get_gateway() -> SafetyGateway:
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = SafetyGateway()
    return _gateway_instance
