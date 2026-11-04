"""
AI Agent 核心 — ReAct (Reasoning + Acting) 循环

与旧 Pipeline 的关键区别：
- 旧：代码硬编码调用顺序（安全→意图→检索→生成）
- 新：LLM 作为"大脑"，自主决策何时调用哪个工具、何时给出最终回答
- LLM 可以：多次检索、调整查询、组合多工具结果、自主判断信息是否充足
"""
import json
import os
import sys
import requests
import time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from engine.tools import TOOL_DEFINITIONS, execute_tool

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# ============================================================
# Agent 系统提示词
# ============================================================

SYSTEM_PROMPT = """你是某市燃气公司（某市燃气）的AI智能客服助手。你的名字是"某市燃气AI客服"。

## 你的能力
你拥有一套工具，可以在需要时调用它们来获取信息：
1. **check_safety** ★★★ 最高优先级！检测用户是否提到燃气泄漏、中毒、着火、爆炸等紧急情况。如果用户提及相关词汇，必须第一个调用此工具。
2. **search_knowledge_base** — 检索燃气业务知识库，获取标准答案
3. **classify_user_intent** — 识别用户意图分类（支持传入上轮意图理解追问）
4. **lookup_regulation** — 查询法律法规条款详情
5. **get_service_info** — 查询某市燃气的客服电话、营业网点、营业时间等基础服务信息
6. **get_related_questions** — 获取用户可能关心的相关问题
7. **request_human_transfer** — 将对话转接至人工客服

## 核心原则

### 1. 安全优先（最重要！）
- 用户提到燃气泄漏、中毒、火灾、爆炸、报警器响、煤气味 → 第一步必须调用 check_safety
- check_safety 返回 is_safety=true → 直接用返回的安全模板回复，不调其他工具，一字不改

### 2. 事实来源唯一性
- 燃气业务知识只能来自 search_knowledge_base 的检索结果
- 不在检索结果外添加知识，不确定就转人工，不猜测

### 3. 效率原则
- 基础服务信息（电话/地址/营业时间）→ 调 get_service_info，不要调 search_knowledge_base
- search_knowledge_base 首次结果够用就直接回答，不重复检索
- 只有首次搜索完全无结果时，才换查询词重试1次

### 4. 法规追溯
- 检索结果中有 regulation_code → 必须调用 lookup_regulation 获取法规全名
- 回答末尾附上法规依据，格式: "依据：《法规全名》(编码)"

### 5. 诚实边界
- 搜不到答案 → request_human_transfer
- 用户要求转人工 → request_human_transfer
- 投诉 / 法律纠纷 → request_human_transfer

### 6. 多轮对话理解
- 用户的简短追问（"多少钱""地址在哪""然后呢""第二种呢""需要什么材料"）是对上一轮回答的追问
- 结合对话历史理解用户意图，不要当成全新的独立问题
- 如果历史中有上一轮对话的意图分类结果，可以传给 classify_user_intent 或 search_knowledge_base 的 context_intent / intent_hint 参数

### 7. 元问题处理（用户引用历史对话）
- "回答我的上一个问题""刚才那个问题""之前的问题再回答一下" → 用户要求回顾之前的问答
- 做法：查看对话历史中最近一轮的用户消息（role=user），直接重新回答那个问题
- 不要检索知识库，不要调用任何工具，直接从对话历史中提取上一轮的问题并重新回答
- "第一个问题""第二个问题""第N个问题""重新回答第X个" → 用户引用编号的问题
  * 从对话历史中按顺序数用户消息（role=user）
  * 第1条用户消息 = 本轮对话的第一个问题
  * 第2条用户消息 = 第二个问题，以此类推
  * 找到对应轮次的用户问题后，重新组织回答那个问题
- 简单直接地重新输出对应问题的回答即可，不需要重新检索知识库

### 8. 回答风格
- 先给结论，再给步骤，最后附"依据：《法规名》(编码)"
- 不用套话开头（如"您好，很高兴为您服务"），直接给信息
- 简洁专业，具体可操作

## 某市燃气关键信息
- 24小时抢修专线：400-000-0000
- 人工客服电话：400-000-0001
- 服务热线/报修：400-000-0000
- 监督投诉：400-000-0002
- 微信公众号：某市燃气

## 禁止行为
- 禁止编造业务流程、费用标准、政策法规
- 禁止给出与检索结果矛盾的回答
- 禁止在未检索的情况下仅凭自己的知识回答业务问题
- check_safety 命中后禁止调用其他工具，直接输出安全模板内容"""

# ============================================================
# Agent 核心类
# ============================================================

class GasServiceAgent:
    """某市燃气 AI Agent — ReAct 模式"""

    def __init__(self):
        self.max_iterations = 5  # 最多5轮工具调用
        self.api_timeout = 30    # API 超时（秒）

    def run(self, user_query: str, history: list = None, session_id: str = None,
            last_intent: dict = None) -> dict:
        """
        运行 Agent，处理用户问题

        参数:
            user_query: 用户当前问题
            history: 对话历史 [{'role':'user'/'assistant', 'content':'...'}]
            session_id: 会话ID（用于日志）
            last_intent: 上一轮对话的意图分类结果，用于上下文继承

        返回:
            {
                'answer': str,           # 最终回答
                'tool_calls_log': list,  # 工具调用链（用于调试和透明度）
                'iterations': int,       # 实际迭代次数
                'source': str,           # 'agent' | 'fallback'
                'context_inherited': bool # 是否触发了上下文继承
            }
        """
        context_inherited = False

        # === 上下文意图继承检测 ===
        # 短追问（≤10字 + 特定追问模式）自动继承上一轮意图
        followup_patterns = ['多少钱', '收费', '价格', '费用', '地址', '在哪', '电话',
                            '材料', '流程', '怎么办', '需要什么', '然后', '接着',
                            '之后', '第二种', '别的', '其他', '还有', '步骤', '微信',
                            '公众号', '网上', '线上', '手机', '怎么交', '怎么查']
        # 元问题模式：用户引用对话历史本身，不应被意图提示误导
        meta_patterns = ['上一个', '刚才', '之前', '前面', '刚刚', '上次', '前面那个',
                        '上上个', '之前的', '刚说的', '刚才的', '前一个',
                        '第一个', '第二个', '第三个', '第四个', '第五个',
                        '第1个', '第2个', '第3个', '第4个', '第5个',
                        '重新回答', '再回答', '回答第']

        is_meta_question = any(p in user_query for p in meta_patterns)

        if last_intent and len(user_query) <= 10 and not is_meta_question:
            if any(p in user_query for p in followup_patterns):
                context_inherited = True
                print(f"[Agent] Context inherited: '{user_query}' → {last_intent.get('intent_name','')}>{last_intent.get('sub_intent','')}")

        # 构建消息列表
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 注入对话历史（最近3轮，最多6条）
        if history:
            recent = history[-6:]
            for msg in recent:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")[:500]
                })

        # 添加上下文意图提示（帮助 Agent 理解多轮对话）
        if context_inherited:
            # 业务追问 → 注入完整意图上下文，提升检索精度
            intent_hint = f"{user_query}\n\n[系统提示：这是一个对上一轮回答的追问。上一轮意图为「{last_intent.get('intent_name', '')}」—「{last_intent.get('sub_intent', '')}」。请在检索时使用上轮意图域过滤，并用 classify_user_intent 传入上轮意图作为 context_intent 确认。]"
            messages.append({"role": "user", "content": intent_hint})
        elif is_meta_question:
            # 元问题（"回答上一个问题""刚才说的"）→ 不注入意图提示，让 LLM 直接从对话历史理解
            messages.append({"role": "user", "content": user_query})
        elif last_intent and len(user_query) >= 4:
            # 非短追问但有上一轮意图，做轻量提示（仅对足够长的正常问题）
            hint = f"{user_query}\n\n[上轮对话意图参考：{last_intent.get('intent_name', '')} > {last_intent.get('sub_intent', '')}]"
            messages.append({"role": "user", "content": hint})
        else:
            messages.append({"role": "user", "content": user_query})

        tool_calls_log = []

        # === ReAct 循环 ===
        for iteration in range(1, self.max_iterations + 1):
            try:
                response = self._call_llm(messages)
            except Exception as e:
                print(f"[Agent] LLM call failed at iteration {iteration}: {e}")
                return self._fallback_response(
                    f"系统繁忙，请稍后重试或拨打人工客服 400-000-0001。",
                    tool_calls_log, iteration
                )

            assistant_msg = response.get("choices", [{}])[0].get("message", {})
            finish_reason = response.get("choices", [{}])[0].get("finish_reason", "")

            # 情况1：LLM 要求调用工具
            if finish_reason == "tool_calls" or assistant_msg.get("tool_calls"):
                tool_calls = assistant_msg.get("tool_calls", [])

                # 将 assistant 消息（含 tool_calls）追加到对话
                messages.append({
                    "role": "assistant",
                    "content": assistant_msg.get("content") or "",
                    "tool_calls": tool_calls
                })

                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    fn_args = json.loads(tc["function"]["arguments"])

                    print(f"[Agent] Iter {iteration}: calling {fn_name}({json.dumps(fn_args, ensure_ascii=False)[:100]})")

                    # 执行工具
                    result = execute_tool(fn_name, fn_args)

                    # 解析工具返回的 JSON，提取结构化信息
                    result_parsed = None
                    try:
                        result_parsed = json.loads(result)
                    except Exception:
                        pass

                    tool_calls_log.append({
                        "iteration": iteration,
                        "tool": fn_name,
                        "arguments": fn_args,
                        "result_preview": result[:150] + ("..." if len(result) > 150 else ""),
                        "result_parsed": result_parsed  # 完整结构化数据，供 API 层解析
                    })

                    # 将工具结果追加到对话
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{iteration}"),
                        "content": result
                    })

                # 继续循环，让 LLM 看到工具结果后决定下一步
                continue

            # 情况2：LLM 返回最终回答
            content = assistant_msg.get("content", "")
            if content:
                print(f"[Agent] Final answer at iteration {iteration} ({len(content)} chars)")
                return {
                    "answer": content,
                    "tool_calls_log": tool_calls_log,
                    "iterations": iteration,
                    "source": "agent",
                    "context_inherited": context_inherited
                }

            # 情况3：异常空响应
            print(f"[Agent] Empty response at iteration {iteration}")
            return self._fallback_response(
                "抱歉，我暂时无法处理您的问题。请拨打人工客服 400-000-0001 咨询。",
                tool_calls_log, iteration
            )

        # 超出最大迭代次数
        print(f"[Agent] Max iterations ({self.max_iterations}) reached")
        return {
            "answer": "抱歉，处理您的问题时遇到了复杂情况。为了更准确地帮助您，建议拨打人工客服 400-000-0001 咨询。",
            "tool_calls_log": tool_calls_log,
            "iterations": self.max_iterations,
            "source": "max_iterations",
            "context_inherited": context_inherited
        }

    def _call_llm(self, messages: list) -> dict:
        """调用 DeepSeek API（带 function calling）"""
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置，请设置环境变量后重试")

        payload = {
            "model": "deepseek-v4-pro",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 800,
            "tools": TOOL_DEFINITIONS,
            "tool_choice": "auto"  # 让模型自主决定是否调用工具
        }

        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=self.api_timeout
        )

        if resp.status_code != 200:
            raise RuntimeError(f"DeepSeek API error {resp.status_code}: {resp.text[:200]}")

        return resp.json()

    def _fallback_response(self, message: str, log: list, iterations: int) -> dict:
        return {
            "answer": message,
            "tool_calls_log": log,
            "iterations": iterations,
            "source": "fallback",
            "context_inherited": False
        }


# ============================================================
# 单例
# ============================================================

_agent_instance = None


def get_agent() -> GasServiceAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = GasServiceAgent()
    return _agent_instance
