"""
会话管理模块
多轮对话上下文 + 日志记录
"""
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
LOG_DIR = os.path.join(DATA_DIR, 'logs')


class Session:
    """单个会话"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now().isoformat()
        self.messages: List[Dict] = []  # [{role, content, intent, ...}]
        self.context: Dict = {}          # 上下文状态
        self.closed = False

    def add_message(self, role: str, content: str, **meta):
        msg = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            **meta
        }
        self.messages.append(msg)
        return msg

    def get_history(self, last_n: int = 5) -> List[Dict]:
        """获取最近N轮对话"""
        return self.messages[-last_n * 2:]  # user + assistant pairs

    def get_last_intent(self) -> Optional[Dict]:
        """获取最近一次意图"""
        for msg in reversed(self.messages):
            if 'intent' in msg:
                return msg['intent']
        return None

    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'created_at': self.created_at,
            'messages': self.messages,
            'context': self.context,
            'closed': self.closed
        }


def _get_first_user_message(messages: list) -> str:
    """从消息列表中提取第一条用户消息"""
    for msg in messages:
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            return content[:40] if len(content) > 40 else content
    return '（空对话）'


class SessionManager:
    """会话管理器"""

    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        os.makedirs(LOG_DIR, exist_ok=True)

    def create_session(self) -> str:
        """创建新会话，返回session_id"""
        sid = f"sess_{int(time.time() * 1000)}"
        self.sessions[sid] = Session(sid)
        return sid

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话，先查内存再查磁盘"""
        if session_id in self.sessions:
            return self.sessions[session_id]
        # 从磁盘恢复
        log_file = os.path.join(LOG_DIR, f"{session_id}.json")
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                session = Session(session_id)
                session.created_at = data.get('created_at', '')
                session.messages = data.get('messages', [])
                session.context = data.get('context', {})
                session.closed = False
                self.sessions[session_id] = session
                return session
            except Exception:
                pass
        return None

    def save_session(self, session_id: str):
        """实时保存会话到磁盘（每轮对话后调用）"""
        if session_id in self.sessions:
            self._archive_log(self.sessions[session_id])

    def close_session(self, session_id: str):
        """关闭会话并存档日志"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.closed = True
            self._archive_log(session)
            del self.sessions[session_id]

    def delete_session(self, session_id: str) -> bool:
        """删除会话（内存+磁盘）"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        log_file = os.path.join(LOG_DIR, f"{session_id}.json")
        if os.path.exists(log_file):
            os.remove(log_file)
            return True
        return False

    def _archive_log(self, session: Session):
        """将会话日志存档到磁盘"""
        log_file = os.path.join(LOG_DIR, f"{session.session_id}.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

    def list_archived_sessions(self, limit: int = 20) -> list:
        """列出所有存档的会话"""
        sessions = []
        if os.path.exists(LOG_DIR):
            for fname in os.listdir(LOG_DIR):
                if fname.startswith('sess_') and fname.endswith('.json'):
                    fpath = os.path.join(LOG_DIR, fname)
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        sessions.append({
                            'session_id': data.get('session_id', fname[:-5]),
                            'created_at': data.get('created_at', ''),
                            'message_count': len(data.get('messages', [])),
                            'first_message': _get_first_user_message(data.get('messages', [])),
                            'closed': data.get('closed', False)
                        })
                    except Exception:
                        pass
        sessions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return sessions[:limit]

    def add_uncovered_query(self, query: str, intent: dict):
        """记录未覆盖问题，用于知识治理迭代"""
        uncovered_file = os.path.join(LOG_DIR, 'uncovered_queries.jsonl')
        record = {
            'query': query,
            'intent': intent,
            'timestamp': datetime.now().isoformat()
        }
        with open(uncovered_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def get_stats(self) -> Dict:
        """获取运行统计"""
        return {
            'active_sessions': len(self.sessions),
            'total_sessions_tracked': len(self.sessions)
        }


# 单例
_manager_instance = None

def get_session_manager() -> SessionManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = SessionManager()
    return _manager_instance
