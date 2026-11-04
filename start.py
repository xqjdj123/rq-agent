"""
某市燃气AI客服系统 v2.0 — Agent 架构 — 启动脚本

架构升级说明（v1.0 → v2.0）：
  v1.0 (Pipeline): 代码硬编码控制流，LLM 仅做文本转换
  v2.0 (Agent):    LLM 作为"大脑"，ReAct 循环自主决策工具调用

启动方式：
  python start.py

首次启动：
  1. 复制 .env.example 为 .env，填入 DeepSeek API Key
  2. 确保 BAAI/bge-base-zh-v1.5 模型已下载到本地
  3. python start.py
"""
import sys
import os
import subprocess
import threading
import time
import urllib.request
import json

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# 加载 .env 文件（如果存在）
def load_dotenv():
    env_file = os.path.join(PROJECT_DIR, '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())
        print("[OK] .env loaded")
    else:
        print("[!] .env not found — 请复制 .env.example 为 .env 并设置 API Key")
        # 检查环境变量
        if not os.environ.get("DEEPSEEK_API_KEY"):
            print("[!] DEEPSEEK_API_KEY 未设置，Agent 将无法工作")
            print("    设置方式：")
            print("    1. 复制 .env.example → .env 并填入 Key")
            print("    2. 或 set DEEPSEEK_API_KEY=sk-xxx")

load_dotenv()


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║       某市燃气AI客服智能体系 v2.0 (Agent 架构)           ║
║       Hengyang Gas AI Customer Service System            ║
║                                                          ║
║       架构: Safety Gateway + ReAct Agent                  ║
║       引擎: DeepSeek Function Calling                    ║
║       检索: FAISS + BM25 (RRF融合)                       ║
║       图谱: NetworkX | 后端: FastAPI                     ║
║                                                          ║
║       Pipeline → Agent: LLM 自主决策工具调用              ║
╚══════════════════════════════════════════════════════════╝
""")


def run_step(name, script_path, args=None):
    """运行预处理步骤"""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=False)
    if result.returncode != 0:
        print(f"\n[WARNING] {name} completed with code {result.returncode}")
    return result.returncode == 0


def find_available_port(start=8000, end=8010):
    import socket
    for port in range(start, end + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', port))
            sock.close()
            return port
        except OSError:
            sock.close()
    return None


def main():
    print_banner()

    # Step 1: 数据预处理
    if not os.path.exists(os.path.join(PROJECT_DIR, 'data', 'knowledge_base.json')):
        print("\n[1/3] Running data preprocessing...")
        run_step("Data Preprocessing", os.path.join(PROJECT_DIR, 'data', 'preprocess.py'))
    else:
        print("\n[1/3] Knowledge base exists, skipping preprocessing.")

    # Step 2: 知识图谱
    if not os.path.exists(os.path.join(PROJECT_DIR, 'kg', 'graph_export.json')):
        print("\n[2/3] Building knowledge graph...")
        run_step("Knowledge Graph", os.path.join(PROJECT_DIR, 'kg', 'build_graph.py'))
    else:
        print("\n[2/3] Knowledge graph exists, skipping.")

    # Step 3: 端口检测
    PORT = find_available_port(8000, 8009)
    if PORT is None:
        print("\n[!] 端口 8000-8009 全部被占用，请手动释放后重试")
        sys.exit(1)
    if PORT != 8000:
        print(f"\n[!] 端口 8000 被占用，已自动切换到 {PORT}")

    # Step 4: 启动 Agent API 服务（后台线程 + 预热请求）
    print(f"\n[3/3] Starting Agent API server on http://localhost:{PORT}")
    print(f"  - Chat Demo:    http://localhost:{PORT}")
    print(f"  - API Docs:     http://localhost:{PORT}/docs")
    print(f"  - Health Check: http://localhost:{PORT}/api/health")
    print(f"  - Debug:        http://localhost:{PORT}/api/debug")
    print()
    print("  Agent 特性：LLM 自主决策工具调用（检索/分类/法规查证/转人工）")
    print("  观察 Agent 行为：检查 API 返回中的 tool_calls_detail 字段")
    print()
    print("  Loading models and warming up... (may take 10-20s on first run)")
    print()

    import uvicorn

    def run_server():
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=PORT,
            reload=False,
            log_level="info"
        )

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # 等待服务器就绪 + 预热请求（带重试）
    warmup_url = f"http://localhost:{PORT}/api/chat"
    warmup_data = json.dumps({"query": "预热测试", "session_id": "_warmup_"}).encode("utf-8")
    warmup_success = False

    for attempt in range(20):
        time.sleep(2)
        try:
            req = urllib.request.Request(warmup_url, data=warmup_data,
                                         headers={"Content-Type": "application/json"},
                                         method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 200:
                    print(f"\n[Warmup] Chat API warmed up successfully!")
                    warmup_success = True
                    break
        except Exception:
            if attempt == 0:
                pass  # 第一次通常连接被拒绝，静默等待
            elif attempt % 3 == 0:
                print(f"  Waiting for server... ({attempt * 2}s)")

    if not warmup_success:
        print("\n[Warmup] Warning: warmup request timed out, but server may still work.")

    print(f"\n{'='*50}")
    print(f"  Server ready! Visit http://localhost:{PORT}")
    print(f"  Press Ctrl+C to stop the server.")
    print(f"{'='*50}")

    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()
