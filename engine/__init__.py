"""
某市燃气AI客服引擎 — Agent 架构版

模块说明：
- agent.py   : ReAct Agent 核心循环（LLM 自主决策工具调用）
- tools.py   : 5个Agent工具定义与执行器
- retriever.py : 混合检索引擎（FAISS + BM25 + RRF）
- safety_gateway.py : 安全网关（硬预检，不经过Agent）
- session.py : 会话管理与日志
"""
