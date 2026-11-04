"""
Agent 工具定义与执行器
为 AI Agent 提供 5 个可调用工具，每个工具有 OpenAI Function Calling 兼容的 JSON Schema
"""
import json
import os
import sys
import requests

# 项目根目录
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

# API 配置 — 从环境变量读取，或使用默认值
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# 服务信息库（营业网点、电话、时间等）
SERVICE_INFO = {
    "抢修电话": "400-000-0000（24小时）",
    "客服电话": "400-000-0001",
    "服务热线": "400-000-0000",
    "监督投诉": "400-000-0002",
    "报装咨询": "400-000-0003",
    "营业网点": "示例区营业厅（示例路xxx号）、示例区营业厅（示例路xxx号）、示例区营业厅（示例路xxx号）、示例区营业厅（示例路xxx号）",
    "营业时间": "工作日 8:30-17:30，周六 9:00-16:00，周日休息",
    "微信公众号": "某市燃气",
    "官方网站": "example.com",
}

# 安全关键词 + 否定语境（与 safety_gateway 保持同步）
from engine.safety_gateway import SAFETY_KEYWORDS, check_safety_keywords


# ============================================================
# 工具 JSON Schema 定义（OpenAI Function Calling 格式）
# ============================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "check_safety",
            "description": "★★★ 最高优先级 ★★★ 检测用户问题是否涉及燃气安全紧急情况（泄漏、中毒、火灾、爆炸、报警器等）。如果用户提到煤气味、报警器响、头晕恶心、着火爆炸等词，必须在回答任何问题前先调用此工具。如果此工具返回 is_safety=true，必须直接使用返回的安全模板回答，禁止调用其他工具，禁止改写安全话术。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户原始输入文本"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "在某市燃气客服知识库中检索与用户问题相关的内容。知识库包含报装、缴费、安检、抢修、安全用气、政策咨询等全业务场景的标准答案。返回最匹配的知识条目（含标准答案、标签分类、法规编码）。当需要回答用户关于燃气业务的具体问题时，优先使用此工具检索。如果首次检索结果不理想，可以调整查询词重新检索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询词。应提炼用户问题的核心意图和关键词，而非直接使用原始口语化表达。例如用户说'我想问下那个气费怎么在手机上搞'，应提炼为'手机缴费 线上缴费方式'"
                    },
                    "intent_filter": {
                        "type": "string",
                        "description": "可选，按一级业务域过滤结果。可选值包括：业务办理、气费管理、上门服务、安全用气、投诉建议、政策咨询、系统操作、燃气咨询、安装维修、缴费管理、价格管理、服务管理、故障处理。不传则不过滤。"
                    },
                    "intent_hint": {
                        "type": "object",
                        "description": "可选，当前对话的意图上下文，格式 {'intent_name':'气费管理','sub_intent':'欠费停气'}。传入后可提升同类问题的检索精度。"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "classify_user_intent",
            "description": "识别用户问题的意图分类，返回对应的一级业务域和二级标签。用于确认用户意图所属的业务类别，辅助后续的精准检索和回答。当用户问题模糊、可能需要澄清，或需要确认意图分类准确性时使用。支持传入上轮意图作为上下文参考。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户原始问题文本"
                    },
                    "context_intent": {
                        "type": "object",
                        "description": "可选，上一轮对话的意图分类结果，格式 {'intent_name':'气费管理','sub_intent':'欠费停气'}。用于理解追问型短句（如'多少钱''地址在哪'）的上下文。"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_regulation",
            "description": "根据法规编码(FG-XX-XXX格式)查询具体的法律法规条款详情，包括法规名称、层级（法律/行政法规/部门规章/地方规范/行业规范）等信息。在回答中需要引用法规依据时调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "fg_code": {
                        "type": "string",
                        "description": "法规编码，格式如 FG-FL-001(法律)、FG-XZ-002(行政法规)、FG-BM-001(部门规章)、FG-DF-001(地方规范)、FG-HY-001(行业规范)"
                    }
                },
                "required": ["fg_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_info",
            "description": "查询某市燃气公司的服务信息。用于回答用户关于客服电话、抢修电话、营业网点地址、营业时间、微信公众号等基础服务信息的问题。用户问电话/地址/营业时间时直接调用此工具，不要调用 search_knowledge_base。",
            "parameters": {
                "type": "object",
                "properties": {
                    "info_type": {
                        "type": "string",
                        "description": "信息类型。可选值：'电话'（客服+抢修+服务热线）、'地址'（营业网点）、'营业时间'、'公众号'、'网站'、'全部'"
                    }
                },
                "required": ["info_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_related_questions",
            "description": "获取某个业务意图下用户常问的相关问题列表。用于在回答完用户当前问题后，向用户推荐可能感兴趣的后续问题，提升服务体验。",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent_name": {
                        "type": "string",
                        "description": "意图名称，格式为'一级域-二级标签'，如'安全用气-疑似漏气处置'、'业务办理-新装报装'等"
                    }
                },
                "required": ["intent_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_human_transfer",
            "description": "当以下情况发生时，将对话转接至人工客服：1)多次检索仍无法找到匹配答案；2)用户明确要求转人工；3)问题涉及投诉、法律纠纷等复杂事务；4)系统无法准确理解用户意图且澄清失败。调用此工具意味着本轮对话以转人工结束。",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "转人工的具体原因，如'知识库未覆盖该问题'、'用户明确要求转人工'、'意图置信度过低'、'涉及法律投诉'等"
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["normal", "urgent"],
                        "description": "紧急程度：normal=普通转接（如未覆盖问题），urgent=紧急转接（安全事件已通过安全模板处理后的后续跟进）"
                    }
                },
                "required": ["reason"]
            }
        }
    }
]


# ============================================================
# 工具执行函数
# ============================================================

class ToolExecutor:
    """工具执行器：接收 Agent 的工具调用请求，执行并返回结果"""

    def __init__(self):
        self._retriever = None
        self._graph_query = None
        self._kb = None
        self._tag_index = None
        self._keyword_index = None

    # ---- 延迟初始化 ----

    @property
    def retriever(self):
        if self._retriever is None:
            from engine.retriever import get_retriever
            self._retriever = get_retriever()
        return self._retriever

    @property
    def graph_query(self):
        if self._graph_query is None:
            try:
                from kg.query import get_graph_query
                self._graph_query = get_graph_query()
            except Exception:
                self._graph_query = None
        return self._graph_query

    @property
    def kb(self):
        if self._kb is None:
            kb_path = os.path.join(PROJECT_DIR, 'data', 'knowledge_base.json')
            with open(kb_path, 'r', encoding='utf-8') as f:
                self._kb = json.load(f)
        return self._kb

    @property
    def tag_index(self):
        """构建标签体系索引：{一级域: {二级标签: [条目列表]}}"""
        if self._tag_index is None:
            self._tag_index = {}
            for item in self.kb:
                cat1 = item['一级域']
                cat2 = item['二级标签']
                if cat1 not in self._tag_index:
                    self._tag_index[cat1] = {}
                if cat2 not in self._tag_index[cat1]:
                    self._tag_index[cat1][cat2] = []
                self._tag_index[cat1][cat2].append(item)
        return self._tag_index

    # ---- 工具执行入口 ----

    def execute(self, tool_name: str, arguments: dict) -> str:
        """执行工具调用，返回结果字符串（JSON格式）"""
        try:
            if tool_name == "check_safety":
                return self._check_safety(**arguments)
            elif tool_name == "search_knowledge_base":
                return self._search_knowledge_base(**arguments)
            elif tool_name == "classify_user_intent":
                return self._classify_user_intent(**arguments)
            elif tool_name == "lookup_regulation":
                return self._lookup_regulation(**arguments)
            elif tool_name == "get_service_info":
                return self._get_service_info(**arguments)
            elif tool_name == "get_related_questions":
                return self._get_related_questions(**arguments)
            elif tool_name == "request_human_transfer":
                return self._request_human_transfer(**arguments)
            else:
                return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"工具执行异常: {str(e)[:200]}"}, ensure_ascii=False)

    # ---- 各工具实现 ----

    def _search_knowledge_base(self, query: str, intent_filter: str = None, intent_hint: dict = None) -> str:
        """
        混合检索知识库：FAISS向量检索 + BM25关键词检索 + RRF融合
        """
        intent = None
        if intent_hint:
            intent = intent_hint
        elif intent_filter:
            intent = {
                'intent_name': intent_filter,
                'sub_intent': '',
                'confidence': 0.7
            }

        results = self.retriever.hybrid_search(query, intent=intent, top_k=5)

        if not results:
            return json.dumps({
                "status": "no_results",
                "message": "未找到与查询匹配的知识条目。建议调整查询词或告知用户该问题暂未收录。",
                "results": []
            }, ensure_ascii=False)

        formatted = []
        for r in results:
            entry = r['entry']
            formatted.append({
                "score": round(r['score'], 3),
                "vector_score": r.get('vector_score', 0.0),
                "category": f"{entry['一级域']} > {entry['二级标签']}",
                "question": entry.get('标准问', '')[:100],
                "answer": entry.get('标准答', '')[:500],
                "regulation_code": entry.get('法规编码', ''),
                "keywords": entry.get('关键词', '')[:100]
            })

        return json.dumps({
            "status": "ok",
            "total_found": len(formatted),
            "results": formatted
        }, ensure_ascii=False)

    def _classify_user_intent(self, query: str, context_intent: dict = None) -> str:
        """
        识别用户意图：关键词匹配 + DeepSeek few-shot 双通道
        支持上下文意图继承（短追问自动沿用上一轮意图）
        """
        # Step 0: 上下文意图继承 — 短追问自动沿用上一轮意图
        if context_intent and len(query) <= 10:
            followup_patterns = ['多少钱', '收费', '价格', '费用', '地址', '在哪', '电话',
                                '材料', '流程', '怎么办', '需要什么', '然后', '接着',
                                '之后', '第二种', '别的', '其他', '还有', '步骤', '微信',
                                '公众号', '网上', '线上', '手机', '怎么交', '怎么查']
            if any(p in query for p in followup_patterns):
                return json.dumps({
                    "method": "context_inherit",
                    "intent_name": context_intent.get('intent_name', ''),
                    "sub_intent": context_intent.get('sub_intent', ''),
                    "confidence": 0.85,
                    "note": f"短追问'{query}'自动继承上一轮意图"
                }, ensure_ascii=False)

        # Step 1: 关键词快速匹配
        try:
            import jieba
            words = jieba.lcut(query)
        except Exception:
            words = list(query)

        # 构建关键词倒排索引
        if self._keyword_index is None:
            self._keyword_index = {}
            for item in self.kb:
                kw_text = item.get('关键词', '')
                for kw in kw_text.split():
                    if kw not in self._keyword_index:
                        self._keyword_index[kw] = []
                    self._keyword_index[kw].append(item['id'])

        keyword_scores = {}
        for word in words:
            if word in self._keyword_index:
                for item_id in self._keyword_index[word]:
                    keyword_scores[item_id] = keyword_scores.get(item_id, 0) + 1

        if keyword_scores:
            cat1_votes = {}
            cat2_votes = {}
            top_items = sorted(keyword_scores.items(), key=lambda x: -x[1])[:5]
            top_ids = [i for i, s in top_items]
            for item in self.kb:
                if item['id'] in top_ids:
                    cat1_votes[item['一级域']] = cat1_votes.get(item['一级域'], 0) + 1
                    cat2_votes[item['二级标签']] = cat2_votes.get(item['二级标签'], 0) + 1

            if cat1_votes and max(cat1_votes.values()) >= 2:
                best_cat1 = max(cat1_votes, key=cat1_votes.get)
                best_cat2 = max(cat2_votes, key=cat2_votes.get) if cat2_votes else ''
                return json.dumps({
                    "method": "keyword_match",
                    "intent_name": best_cat1,
                    "sub_intent": best_cat2,
                    "confidence": min(0.9, 0.5 + len(top_items) * 0.1),
                    "note": f"基于{len(top_items)}个关键词匹配结果投票得出"
                }, ensure_ascii=False)

        # Step 2: DeepSeek few-shot 分类
        if not DEEPSEEK_API_KEY:
            return json.dumps({
                "method": "fallback",
                "intent_name": "其他",
                "sub_intent": "",
                "confidence": 0.3,
                "note": "API密钥未配置，使用兜底分类"
            }, ensure_ascii=False)

        tag_list = []
        for cat1, cat2s in self.tag_index.items():
            for cat2 in cat2s:
                tag_list.append(f"{cat1} > {cat2}")
        tags_text = "\n".join(tag_list)

        prompt = f"""你是一个燃气客服意图分类系统。根据用户输入归类到以下标签体系中。

## 标签体系（一级域 > 二级标签）：
{tags_text}

## 用户输入：
{query}

## 任务：输出JSON格式，包含 intent_name(一级域)、sub_intent(二级标签)、confidence(0-1)
只输出JSON，不要其他内容。"""

        try:
            resp = requests.post(
                DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-v4-pro",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 200
                },
                timeout=15
            )
            if resp.status_code == 200:
                content = resp.json()['choices'][0]['message']['content'].strip()
                if content.startswith('```'):
                    content = content.split('\n', 1)[1]
                    if content.endswith('```'):
                        content = content[:-3]
                llm_result = json.loads(content)
                llm_result["method"] = "llm_few_shot"
                return json.dumps(llm_result, ensure_ascii=False)
            else:
                return json.dumps({
                    "method": "fallback",
                    "intent_name": "其他", "sub_intent": "",
                    "confidence": 0.3, "note": f"LLM API异常({resp.status_code})"
                }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "method": "fallback",
                "intent_name": "其他", "sub_intent": "",
                "confidence": 0.3, "note": f"分类异常: {str(e)[:50]}"
            }, ensure_ascii=False)

    def _lookup_regulation(self, fg_code: str) -> str:
        """查询法规详情（从知识图谱）"""
        if self.graph_query is None:
            return json.dumps({
                "status": "unavailable",
                "message": "知识图谱查询服务暂不可用",
                "code": fg_code
            }, ensure_ascii=False)

        try:
            reg = self.graph_query.lookup_regulation(fg_code)
            if reg and reg.get('name'):
                return json.dumps({
                    "status": "ok",
                    "code": fg_code,
                    "name": reg['name'],
                    "level": reg.get('level', ''),
                    "full_ref": f"{reg['name']}({fg_code}) [{reg.get('level', '')}]"
                }, ensure_ascii=False)
            else:
                return json.dumps({
                    "status": "not_found",
                    "code": fg_code,
                    "message": f"未找到编码 {fg_code} 对应的法规详情"
                }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "code": fg_code,
                "message": f"查询异常: {str(e)[:100]}"
            }, ensure_ascii=False)

    def _get_related_questions(self, intent_name: str) -> str:
        """获取关联问题（从知识图谱）"""
        if self.graph_query is None:
            return json.dumps({"status": "unavailable", "questions": []}, ensure_ascii=False)

        try:
            questions = self.graph_query.get_related_questions(intent_name, limit=5)
            return json.dumps({
                "status": "ok",
                "intent": intent_name,
                "questions": questions
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)[:100], "questions": []}, ensure_ascii=False)

    def _request_human_transfer(self, reason: str, severity: str = "normal") -> str:
        """转人工工具：返回转人工的确认信息和联系方式"""
        contact_info = {
            "normal": {
                "phone": "400-000-0001",
                "hours": "工作日 8:30-17:30",
                "wechat": "「某市燃气」微信公众号 - 在线客服"
            },
            "urgent": {
                "phone": "400-000-0000",
                "hours": "24小时抢修专线",
                "wechat": ""
            }
        }

        info = contact_info.get(severity, contact_info["normal"])

        return json.dumps({
            "status": "transferred",
            "severity": severity,
            "reason": reason,
            "contact": {
                "phone": info["phone"],
                "hours": info["hours"],
                "wechat": info["wechat"]
            },
            "note": "已将用户请求记录为待转人工。请生成包含联系方式和人转理由的礼貌结束语。"
        }, ensure_ascii=False)

    def _check_safety(self, query: str) -> str:
        """
        安全检测工具：检测用户问题是否涉及燃气安全紧急情况
        命中后返回安全模板，Agent 必须直接使用，禁止改写
        """
        # 信息查询类句式不触发安全模式
        INFO_PATTERNS = ["电话", "多少", "是什么", "怎么查", "在哪", "号码", "热线"]
        is_info_query = any(p in query for p in INFO_PATTERNS)

        matched = []
        if not is_info_query:
            # 使用统一的检测函数（自动排除"打不着火"等否定语境）
            matched = check_safety_keywords(query)

        if not matched:
            return json.dumps({
                "is_safety": False,
                "keywords": [],
                "template_name": None,
                "action_required": "无安全风险，可以继续正常处理"
            }, ensure_ascii=False)

        # 命中安全关键词 — 获取对应模板
        try:
            from engine.safety_gateway import get_gateway
            gateway = get_gateway()
            template = gateway.get_safety_template(matched)
        except Exception:
            template = {"title": "⚠️ 燃气安全紧急处置", "steps": []}

        return json.dumps({
            "is_safety": True,
            "keywords": matched,
            "template_name": template.get("title", "安全处置"),
            "template_content": template,
            "action_required": "★★★ 强制走安全模板 + 转人工/抢修 ★★★ 禁止调用其他工具，禁止LLM改写安全话术。请直接用安全模板内容回复用户。"
        }, ensure_ascii=False)

    def _get_service_info(self, info_type: str) -> str:
        """
        查询某市燃气公司的服务信息（电话、地址、时间等）
        """
        if info_type == "全部":
            content = "\n".join(f"{k}: {v}" for k, v in SERVICE_INFO.items())
            return json.dumps({
                "found": True,
                "info_type": "全部",
                "content": content
            }, ensure_ascii=False)

        # 模糊匹配
        for key, value in SERVICE_INFO.items():
            if info_type in key or key in info_type:
                return json.dumps({
                    "found": True,
                    "info_type": key,
                    "content": value
                }, ensure_ascii=False)

        return json.dumps({
            "found": False,
            "info_type": info_type,
            "content": "未找到该类信息，建议转人工咨询"
        }, ensure_ascii=False)


# ============================================================
# 工具调度（供 agent.py 调用）
# ============================================================

_executor_instance = None


def get_executor() -> ToolExecutor:
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = ToolExecutor()
    return _executor_instance


def execute_tool(tool_name: str, arguments: dict) -> str:
    """Agent 调用工具的入口"""
    return get_executor().execute(tool_name, arguments)
