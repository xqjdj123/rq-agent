"""
FastAPI 后端服务 — Agent 架构版
某市燃气AI客服系统 API

架构变化（v2.0 Agent）：
- 旧：Pipeline 硬编码流程（安全→意图→检索→生成）
- 新：安全预检(硬网关) → Agent 自主决策 → 响应
- LLM 从"文本转换器"升级为"对话大脑"，自主调用工具、组合信息、生成回答
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from engine.safety_gateway import get_gateway, check_safety_keywords
from engine.agent import get_agent
from engine.session import get_session_manager
from engine.tools import get_executor
from engine.retriever import get_embedder

app = FastAPI(
    title="某市燃气AI客服系统 (Agent架构)",
    description="基于ReAct Agent模式的智能燃气客服系统API — LLM自主决策工具调用",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web')
app.mount("/web", StaticFiles(directory=WEB_DIR, html=True), name="web")


@app.get("/web/index.html", include_in_schema=False)
async def index_html_nocache():
    """禁止浏览器缓存 index.html，确保用户始终获取最新版本"""
    return FileResponse(
        os.path.join(WEB_DIR, 'index.html'),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.get("/")
async def root():
    return RedirectResponse(url="/web/")


# 全局引擎（延迟初始化）
gateway = None
agent = None
session_mgr = None
_warmed_up = False


def init_engines():
    """初始化所有引擎组件（首次调用时加载，后续调用为 no-op）"""
    global gateway, agent, session_mgr
    if gateway is None:
        print("=" * 60)
        print("  某市燃气AI客服系统 v2.0 (Agent 架构)")
        print("  Architecture: Safety Gateway + ReAct Agent")
        print("=" * 60)
        gateway = get_gateway()
        session_mgr = get_session_manager()
        agent = get_agent()
        print("All engines initialized. Agent ready.")


def warmup():
    """服务启动时全面预热：加载所有模型和索引，避免首个请求的冷启动延迟"""
    global _warmed_up
    if _warmed_up:
        return
    _warmed_up = True

    init_engines()

    # 1. 触发工具执行器及其所有懒加载属性
    executor = get_executor()

    print("Preloading retriever (FAISS + BM25)...")
    _ = executor.retriever

    print("Preloading knowledge graph...")
    _ = executor.graph_query

    print("Preloading knowledge base & tag index...")
    _ = executor.kb
    _ = executor.tag_index

    # 2. 加载本地 BGE 嵌入模型（★ 最大的冷启动瓶颈：~400MB 模型加载）
    print("Loading BGE embedding model (this may take 10-20 seconds on first run)...")
    embedder = get_embedder()

    # 3. 预热 PyTorch 计算图（避免首次推理时的额外延迟）
    print("Warming up embedding model with a dummy input...")
    embedder.encode(["预热测试"])
    print("Warmup complete — all engines ready for requests.\n")


@app.on_event("startup")
async def startup():
    """FastAPI 启动事件：在服务开始接受请求前完成所有预热"""
    warmup()


# ====== 数据模型 ======

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent_code: str
    intent_name: str
    sub_intent: str
    confidence: float
    regulation_ref: str
    source: str
    is_safety: bool
    transfer_to_human: bool
    transfer_reason: str
    # Agent 特有字段
    tool_calls_count: int = 0
    tool_calls_detail: list = []
    agent_iterations: int = 0


class IntentRequest(BaseModel):
    query: str


class IntentResponse(BaseModel):
    intent_code: str
    intent_name: str
    sub_intent: str
    confidence: float
    is_safety: bool
    safety_keywords: List[str]
    source: str


class HealthResponse(BaseModel):
    status: str
    version: str
    architecture: str
    engines_ready: bool


class StatsResponse(BaseModel):
    active_sessions: int
    knowledge_base_size: int


# ====== API 路由 ======

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        version="2.0.0",
        architecture="Safety Gateway + ReAct Agent (DeepSeek Function Calling)",
        engines_ready=agent is not None
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    主对话接口 — Agent 架构版

    流程：
    1. 安全预检（硬网关，不经过 Agent）
    2. Agent 自主处理（ReAct 循环）
    3. 记录会话并返回
    """
    import traceback
    try:
        return await _do_chat(req)
    except Exception as e:
        traceback.print_exc()
        return ChatResponse(
            session_id=req.session_id or 'error',
            answer=f'系统内部错误：{str(e)[:200]}\n\n请检查服务器终端日志获取详情。',
            intent_code='QT', intent_name='', sub_intent='',
            confidence=0, regulation_ref='', source='error',
            is_safety=False, transfer_to_human=False, transfer_reason=''
        )


async def _do_chat(req: ChatRequest):
    if session_mgr is None:
        return ChatResponse(
            session_id='', answer='系统正在初始化，请稍后重试。',
            intent_code='QT', intent_name='', sub_intent='',
            confidence=0, regulation_ref='', source='init',
            is_safety=False, transfer_to_human=False, transfer_reason=''
        )

    # 会话管理
    if req.session_id:
        session = session_mgr.get_session(req.session_id)
        if session is None:
            session_id = session_mgr.create_session()
            session = session_mgr.get_session(session_id)
        else:
            session_id = req.session_id
    else:
        session_id = session_mgr.create_session()
        session = session_mgr.get_session(session_id)

    query = req.query.strip()

    # ============================================================
    # Step 1: 安全预检（硬网关 — 不经过 Agent）
    # 方案文档要求：安全类问题严禁 LLM 自由发挥
    # 使用 check_safety_keywords() 排除否定语境（如"打不着火"不触发"着火"）
    # ============================================================
    matched_safety_kws = check_safety_keywords(query)
    if matched_safety_kws:
        template = gateway.get_safety_template(matched_safety_kws)
        answer = gateway.format_safety_response(template)

        intent_result = {
            'intent_code': 'AQ', 'intent_name': '安全用气',
            'sub_intent': '疑似漏气处置', 'confidence': 1.0,
            'is_safety': True, 'safety_keywords': matched_safety_kws,
            'source': 'safety_keyword'
        }
        session.add_message('user', query, intent=intent_result)
        session.add_message('assistant', answer, type='safety', transfer=True)
        session_mgr.save_session(session_id)

        return ChatResponse(
            session_id=session_id, answer=answer,
            intent_code='AQ', intent_name='安全用气',
            sub_intent='疑似漏气处置', confidence=1.0,
            regulation_ref='FG-XZ-002 (城镇燃气管理条例)',
            source='safety_template', is_safety=True,
            transfer_to_human=True, transfer_reason='安全紧急事件，强制转人工/抢修'
        )

    # ============================================================
    # Step 2: Agent 自主处理（ReAct 循环）
    # LLM 作为"大脑"，自主决定：
    #   - 是否需要检索知识库？检索什么关键词？
    #   - 是否需要识别意图？
    #   - 是否需要查法规？
    #   - 检索结果够不够？要不要换关键词再搜？
    #   - 是否应该转人工？
    # ============================================================
    conversation_history = session.get_history(last_n=3) if session else []

    # 提取上一轮意图（用于上下文继承）
    last_intent = session.get_last_intent() if session else None

    # 将历史转为 Agent 可用的格式
    history_for_agent = []
    for msg in conversation_history:
        history_for_agent.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")[:500]
        })

    agent_result = agent.run(
        user_query=query,
        history=history_for_agent,
        session_id=session_id,
        last_intent=last_intent
    )

    # 解析 Agent 的工具调用日志，提取意图信息
    intent_info = _extract_intent_from_log(agent_result.get('tool_calls_log', []))
    regulation_ref = _extract_regulation_from_log(agent_result.get('tool_calls_log', []))

    # 记录用户消息（含意图信息，用于下一轮的上下文继承）
    session.add_message('user', query, intent=intent_info)

    # 判断是否为安全场景（Agent 调用了 check_safety 且命中）
    is_safety = False
    for call in agent_result.get('tool_calls_log', []):
        if call.get('tool') == 'check_safety':
            parsed = call.get('result_parsed')
            if parsed and parsed.get('is_safety'):
                is_safety = True
                break

    # 判断是否需要转人工
    is_transfer = _is_transfer_result(agent_result) or is_safety
    transfer_reason = ''
    if is_transfer:
        if is_safety:
            transfer_reason = '安全紧急事件，强制转人工/抢修'
        else:
            transfer_reason = _extract_transfer_reason(agent_result)

    # 记录助手消息
    session.add_message(
        'assistant',
        agent_result['answer'],
        type='agent_response',
        intent=intent_info,
        regulation_ref=regulation_ref,
        source=agent_result['source'],
        tool_calls=agent_result.get('tool_calls_log', []),
        is_safety=is_safety,
        transfer=is_transfer
    )
    session_mgr.save_session(session_id)

    # 记录未覆盖问题
    search_results = [c for c in agent_result.get('tool_calls_log', [])
                      if c.get('tool') == 'search_knowledge_base']
    if search_results:
        all_no_results = all(
            c.get('result_parsed', {}).get('status') == 'no_results'
            for c in search_results
        )
        if all_no_results:
            session_mgr.add_uncovered_query(query, intent_info)

    # 转人工也记录
    if is_transfer:
        session_mgr.add_uncovered_query(query, intent_info)

    return ChatResponse(
        session_id=session_id,
        answer=agent_result['answer'],
        intent_code=intent_info.get('intent_code', 'QT'),
        intent_name=intent_info.get('intent_name', ''),
        sub_intent=intent_info.get('sub_intent', ''),
        confidence=intent_info.get('confidence', 0.0),
        regulation_ref=regulation_ref,
        source=agent_result['source'],
        is_safety=is_safety,
        transfer_to_human=is_transfer,
        transfer_reason=transfer_reason,
        tool_calls_count=len(agent_result.get('tool_calls_log', [])),
        tool_calls_detail=agent_result.get('tool_calls_log', []),
        agent_iterations=agent_result.get('iterations', 0)
    )


def _extract_intent_from_log(tool_calls_log: list) -> dict:
    """从 Agent 工具调用日志中提取意图信息
    优先级：classify_user_intent > search_knowledge_base（使用向量相似度） > 默认
    """
    # 优先找 classify_user_intent 的调用结果
    for call in tool_calls_log:
        if call.get('tool') == 'classify_user_intent':
            parsed = call.get('result_parsed')
            if parsed:
                return {
                    'intent_code': _map_cat1_to_code(parsed.get('intent_name', '')),
                    'intent_name': parsed.get('intent_name', ''),
                    'sub_intent': parsed.get('sub_intent', ''),
                    'confidence': parsed.get('confidence', 0.7),
                    'is_safety': False,
                    'safety_keywords': [],
                    'source': 'agent_classify'
                }

    # 其次从 search_knowledge_base 的 top result 提取分类和置信度
    for call in tool_calls_log:
        if call.get('tool') == 'search_knowledge_base':
            parsed = call.get('result_parsed')
            if parsed and parsed.get('results'):
                top = parsed['results'][0]
                category = top.get('category', '')
                if ' > ' in category:
                    cat1, cat2 = category.split(' > ', 1)
                else:
                    cat1, cat2 = category, ''

                # 优先使用向量余弦相似度（天然 0~1，语义相关性的真实度量）
                vec_score = top.get('vector_score', 0.0)
                confidence = vec_score if vec_score > 0 else 0.5

                return {
                    'intent_code': _map_cat1_to_code(cat1),
                    'intent_name': cat1,
                    'sub_intent': cat2,
                    'confidence': round(confidence, 2),
                    'is_safety': False,
                    'safety_keywords': [],
                    'source': 'agent_retrieval'
                }

    return {
        'intent_code': 'QT', 'intent_name': '', 'sub_intent': '',
        'confidence': 0.5, 'is_safety': False, 'safety_keywords': [],
        'source': 'agent'
    }


def _map_cat1_to_code(cat1: str) -> str:
    """一级域名称 → 编码映射"""
    mapping = {
        '业务办理': 'YW', '气费管理': 'QF', '上门服务': 'SM',
        '安全用气': 'AQ', '投诉建议': 'TS', '政策咨询': 'ZC',
        '系统操作': 'XT', '燃气咨询': 'RQ', '安装维修': 'AZ',
        '缴费管理': 'JF', '价格管理': 'JG', '服务管理': 'FW',
        '故障处理': 'GZ', '其他': 'QT', '转人工': 'RG'
    }
    return mapping.get(cat1, 'QT')


def _extract_regulation_from_log(tool_calls_log: list) -> str:
    """从工具调用日志中提取法规引用"""
    # 1. 从 lookup_regulation 工具提取
    for call in tool_calls_log:
        if call.get('tool') == 'lookup_regulation':
            parsed = call.get('result_parsed')
            if parsed:
                if parsed.get('full_ref'):
                    return parsed['full_ref']
                code = parsed.get('code', '')
                name = parsed.get('name', '')
                if name:
                    return f"{name}({code})"
                return code
    # 2. 从 search_knowledge_base 结果的 regulation_code 提取
    for call in tool_calls_log:
        if call.get('tool') == 'search_knowledge_base':
            parsed = call.get('result_parsed')
            if parsed and parsed.get('results'):
                for r in parsed['results']:
                    reg_code = r.get('regulation_code', '')
                    if reg_code:
                        return reg_code
    return ''


def _is_transfer_result(agent_result: dict) -> bool:
    """判断 Agent 结果是否为转人工"""
    log = agent_result.get('tool_calls_log', [])
    for call in log:
        if call.get('tool') == 'request_human_transfer':
            return True
    return False


def _extract_transfer_reason(agent_result: dict) -> str:
    """提取转人工原因"""
    log = agent_result.get('tool_calls_log', [])
    for call in log:
        if call.get('tool') == 'request_human_transfer':
            return call.get('arguments', {}).get('reason', 'Agent判定需转人工')
    return ''


# ====== 意图识别接口 ======

@app.post("/api/intent", response_model=IntentResponse)
async def detect_intent(req: IntentRequest):
    """意图识别接口 — 独立于 Agent，用于调试和评估"""
    executor = get_executor()
    result_json = executor.execute("classify_user_intent", {"query": req.query})
    result = json.loads(result_json)

    return IntentResponse(
        intent_code=_map_cat1_to_code(result.get('intent_name', '')),
        intent_name=result.get('intent_name', ''),
        sub_intent=result.get('sub_intent', ''),
        confidence=result.get('confidence', 0.0),
        is_safety=False,
        safety_keywords=[],
        source=result.get('method', 'llm_few_shot')
    )


# ====== 辅助路由 ======

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    if session_mgr is None:
        return StatsResponse(active_sessions=0, knowledge_base_size=0)

    kb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'knowledge_base.json')
    kb_size = 0
    try:
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb_size = len(json.load(f))
    except Exception:
        pass

    return StatsResponse(
        active_sessions=len(session_mgr.sessions),
        knowledge_base_size=kb_size
    )


@app.get("/api/sessions")
async def list_sessions():
    return {
        'sessions': session_mgr.list_archived_sessions(limit=50) if session_mgr else []
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_mgr is None:
        raise HTTPException(status_code=503, detail="Session manager not ready")
    ok = session_mgr.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    session = session_mgr.get_session(session_id) if session_mgr else None
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@app.get("/api/tags")
async def get_tags():
    kb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'knowledge_base.json')
    with open(kb_path, 'r', encoding='utf-8') as f:
        kb = json.load(f)

    tag_tree = {}
    for item in kb:
        cat1 = item['一级域']
        cat2 = item['二级标签']
        if cat1 not in tag_tree:
            tag_tree[cat1] = []
        if cat2 not in tag_tree[cat1]:
            tag_tree[cat1].append(cat2)
    return tag_tree


@app.get("/api/debug")
async def debug_check():
    """诊断各组件状态"""
    results = {}

    # 知识库
    try:
        kb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'data', 'knowledge_base.json')
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb = json.load(f)
        results['knowledge_base'] = {'status': 'ok', 'entries': len(kb)}
    except Exception as e:
        results['knowledge_base'] = {'status': 'error', 'msg': str(e)[:100]}

    # Agent
    try:
        if agent:
            results['agent'] = {
                'status': 'ok',
                'architecture': 'ReAct (Reasoning + Acting)',
                'max_iterations': agent.max_iterations,
                'tools': [t['function']['name'] for t in TOOL_DEFINITIONS]
            }
        else:
            results['agent'] = {'status': 'not_initialized'}
    except Exception as e:
        results['agent'] = {'status': 'error', 'msg': str(e)[:100]}

    # Session
    try:
        active = len(session_mgr.sessions) if session_mgr else 0
        results['sessions'] = {'status': 'ok', 'active': active}
    except Exception as e:
        results['sessions'] = {'status': 'error', 'msg': str(e)[:100]}

    # DeepSeek API
    try:
        import requests as req
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            results['deepseek_api'] = {'status': 'error', 'msg': 'DEEPSEEK_API_KEY 未配置'}
        else:
            resp = req.post(
                'https://api.deepseek.com/chat/completions',
                headers={'Authorization': f'Bearer {api_key}',
                         'Content-Type': 'application/json'},
                json={'model': 'deepseek-v4-pro', 'messages': [{'role': 'user', 'content': 'OK'}],
                      'max_tokens': 5, 'temperature': 0},
                timeout=10
            )
            if resp.status_code == 200:
                results['deepseek_api'] = {'status': 'ok', 'model': 'deepseek-v4-pro'}
            else:
                results['deepseek_api'] = {'status': 'error', 'http_code': resp.status_code}
    except Exception as e:
        results['deepseek_api'] = {'status': 'error', 'msg': str(e)[:100]}

    return results


@app.get("/api/graph/summary")
async def get_graph_summary():
    try:
        from kg.query import get_graph_query
        return get_graph_query().get_graph_summary()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/graph/regulation/{fg_code}")
async def get_regulation(fg_code: str):
    try:
        from kg.query import get_graph_query
        return get_graph_query().lookup_regulation(fg_code)
    except Exception as e:
        return {"error": str(e)}


# 需要导入 TOOL_DEFINITIONS 用于 debug
from engine.tools import TOOL_DEFINITIONS


# ====== 公司概况 API（docx 自动转 HTML） ======

def docx_to_html(file_path):
    """将 docx 文档转换为带格式的 HTML（优先 python-docx，回退 zipfile）"""
    try:
        from docx import Document
        return _docx_to_html_via_python_docx(file_path, Document)
    except ImportError:
        return _docx_to_html_via_zipfile(file_path)


def _docx_to_html_via_python_docx(file_path, Document):
    """使用 python-docx 库解析"""
    import re

    doc = Document(file_path)
    html_parts = []

    for para in doc.paragraphs:
        style_name = para.style.name if para.style else 'Normal'
        if style_name.startswith('Heading'):
            level = re.sub(r'\D', '', style_name) or '1'
            tag = f'h{level}'
        else:
            tag = 'p'

        align = ''
        if para.alignment:
            align_map = {0: 'left', 1: 'center', 2: 'right', 3: 'justify'}
            align = f' style="text-align:{align_map.get(para.alignment, "left")}"'

        text_parts = []
        for run in para.runs:
            text = run.text
            if not text:
                continue
            if run.bold:
                text = f'<strong>{text}</strong>'
            if run.italic:
                text = f'<em>{text}</em>'
            if run.underline:
                text = f'<u>{text}</u>'
            text_parts.append(text)

        text_content = ''.join(text_parts) if text_parts else para.text
        if not text_content.strip():
            text_content = '&nbsp;'

        html_parts.append(f'<{tag}{align}>{text_content}</{tag}>')

    for table in doc.tables:
        html_parts.append('<table style="border-collapse:collapse;width:100%;margin:10px 0;">')
        for row in table.rows:
            html_parts.append('<tr>')
            for cell in row.cells:
                cell_text = []
                for p in cell.paragraphs:
                    text = p.text
                    if text.strip():
                        cell_text.append(text)
                html_parts.append(f'<td style="border:1px solid #ccc;padding:8px;">{"<br>".join(cell_text)}</td>')
            html_parts.append('</tr>')
        html_parts.append('</table>')

    return _wrap_docx_html(''.join(html_parts))


def _docx_to_html_via_zipfile(file_path):
    """使用 stdlib zipfile + xml 解析 docx（无需 python-docx）"""
    import zipfile
    from xml.etree import ElementTree as ET

    NS = {
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    }

    html_parts = []
    with zipfile.ZipFile(file_path, 'r') as z:
        if 'word/document.xml' not in z.namelist():
            return _wrap_docx_html('<p>无法解析文档内容</p>')
        xml_content = z.read('word/document.xml')
        root = ET.fromstring(xml_content)

        for para in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
            # Check paragraph style
            style = ''
            pPr = para.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr')
            if pPr is not None:
                pStyle = pPr.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pStyle')
                if pStyle is not None:
                    style = pStyle.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', '')

            # Determine tag
            if style and style.startswith('Heading'):
                level = ''.join(c for c in style if c.isdigit()) or '1'
                tag = f'h{level}'
            else:
                tag = 'p'

            # Extract text with bold markers
            text_parts = []
            for run in para.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r'):
                run_text = []
                for t in run.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                    run_text.append(t.text or '')
                text = ''.join(run_text)
                if not text:
                    continue
                # Check if bold
                bold = run.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr/{http://schemas.openxmlformats.org/wordprocessingml/2006/main}b') is not None
                if bold:
                    text = f'<strong>{text}</strong>'
                text_parts.append(text)

            text_content = ''.join(text_parts)
            if not text_content.strip():
                text_content = '&nbsp;'

            html_parts.append(f'<{tag}>{text_content}</{tag}>')

    return _wrap_docx_html(''.join(html_parts))


def _wrap_docx_html(body):
    """包裹统一样式"""
    style = '''
    <style>
        .docx-content { font-size:14px; line-height:1.8; color:#2c3038; }
        .docx-content h1 { font-size:22px; font-weight:700; margin:16px 0 8px 0; color:#1a6fb5; }
        .docx-content h2 { font-size:18px; font-weight:600; margin:14px 0 6px 0; color:#1a6fb5; }
        .docx-content h3 { font-size:16px; font-weight:600; margin:12px 0 6px 0; color:#1a5f9e; }
        .docx-content p { margin:6px 0; }
        .docx-content strong { color:#1a5f9e; }
        .docx-content table { margin:10px 0; }
        .docx-content td { border:1px solid #ddd; padding:8px; }
    </style>
    <div class="docx-content">
    '''
    return style + body + '</div>'


@app.get("/api/company/profile")
async def get_company_profile():
    """获取企业概况（docx 自动转 HTML）"""
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', '公司概述1.docx')
    try:
        html_content = docx_to_html(file_path)
        return {"status": "ok", "content": html_content, "format": "html"}
    except FileNotFoundError:
        return {"status": "error", "content": "企业概况文件不存在"}
    except Exception as e:
        return {"status": "error", "content": f"读取文件失败: {str(e)}"}


@app.get("/api/company/culture")
async def get_company_culture():
    """获取企业文化（docx 自动转 HTML）"""
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', '企业文化1.docx')
    try:
        html_content = docx_to_html(file_path)
        return {"status": "ok", "content": html_content, "format": "html"}
    except FileNotFoundError:
        return {"status": "error", "content": "企业文化文件不存在"}
    except Exception as e:
        return {"status": "error", "content": f"读取文件失败: {str(e)}"}


@app.get("/api/company/qrcode")
async def get_company_qrcode():
    """获取企业二维码"""
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', '企业二维码.png')
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    return {"status": "error", "message": "二维码文件不存在"}


@app.get("/api/company/background")
async def get_company_background():
    """获取企业背景图"""
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', '背景图.png')
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    return {"status": "error", "message": "背景图文件不存在"}


# ====== 启动 ======

if __name__ == '__main__':
    print("=" * 60)
    print("  某市燃气AI客服系统 v2.0 (Agent 架构)")
    print("  Safety Gateway + ReAct Agent")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
