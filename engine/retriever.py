"""
混合检索引擎
FAISS向量检索 + BM25关键词检索
"""
import json
import os
import numpy as np
import pickle

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
KB_PATH = os.path.join(DATA_DIR, 'knowledge_base.json')
INDEX_DIR = os.path.join(DATA_DIR, 'indexes')
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'BAAI', 'bge-base-zh-v1.5')

# 延迟加载
_faiss_index = None
_bm25 = None
_embedder = None
_kb = None


class LocalEmbedder:
    """本地BGE嵌入模型封装（使用transformers直接加载）"""

    def __init__(self, model_path=MODEL_PATH):
        from transformers import AutoModel, AutoTokenizer
        import torch

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading BGE model from {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        self.dim = 768
        print(f"BGE model loaded (dim={self.dim})")

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True,
               batch_size=32):
        """编码文本为向量"""
        import torch

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling with attention mask
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(
                    mask_expanded.sum(1), min=1e-9
                )

                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

            if show_progress_bar and i % (batch_size * 10) == 0:
                print(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)}")

        return np.vstack(all_embeddings)


def get_embedder():
    """获取嵌入模型单例"""
    global _embedder
    if _embedder is None:
        _embedder = LocalEmbedder()
    return _embedder


def load_kb():
    global _kb
    if _kb is None:
        with open(KB_PATH, 'r', encoding='utf-8') as f:
            _kb = json.load(f)
    return _kb


class HybridRetriever:
    """混合检索器：FAISS向量相似度 + BM25关键词匹配"""

    def __init__(self):
        self.kb = load_kb()
        self._build_or_load_indexes()

    def _build_or_load_indexes(self):
        """构建或加载索引"""
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss_path = os.path.join(INDEX_DIR, 'faiss.index')
        bm25_path = os.path.join(INDEX_DIR, 'bm25.pkl')

        # 准备文档
        self.doc_texts = []
        for item in self.kb:
            text = f"{item['一级域']} {item['二级标签']} {item['标准问']} {item['标签定义']}"
            self.doc_texts.append(text)

        # FAISS索引（加载失败自动重建）
        faiss_loaded = False
        if os.path.exists(faiss_path):
            try:
                import faiss
                self.faiss_index = faiss.read_index(faiss_path)
                print("Loaded existing FAISS index")
                faiss_loaded = True
            except Exception as e:
                print(f"FAISS index load failed, will rebuild: {e}")
        if not faiss_loaded:
            self._build_faiss_index()

        # BM25索引
        bm25_loaded = False
        if os.path.exists(bm25_path):
            try:
                with open(bm25_path, 'rb') as f:
                    self.bm25 = pickle.load(f)
                print("Loaded existing BM25 index")
                bm25_loaded = True
            except Exception as e:
                print(f"BM25 index load failed, will rebuild: {e}")
        if not bm25_loaded:
            self._build_bm25_index()

    def _build_faiss_index(self):
        """构建FAISS向量索引"""
        import faiss
        embedder = get_embedder()
        print("Building FAISS index...")
        embeddings = embedder.encode(self.doc_texts, show_progress_bar=True, normalize_embeddings=True)

        self.faiss_index = faiss.IndexFlatIP(embedder.dim)
        self.faiss_index.add(embeddings.astype(np.float32))

        faiss.write_index(self.faiss_index, os.path.join(INDEX_DIR, 'faiss.index'))
        print(f"FAISS index built: {self.faiss_index.ntotal} vectors, dim={embeddings.shape[1]}")

    def _build_bm25_index(self):
        """构建BM25索引"""
        from rank_bm25 import BM25Okapi
        print("Building BM25 index...")

        try:
            import jieba
            tokenized = [list(jieba.cut(text)) for text in self.doc_texts]
        except:
            tokenized = [text.split() for text in self.doc_texts]

        self.bm25 = BM25Okapi(tokenized)

        pickle_path = os.path.join(INDEX_DIR, 'bm25.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.bm25, f)
        print(f"BM25 index built: {len(self.doc_texts)} documents")

    def vector_search(self, query: str, top_k: int = 10) -> list:
        """向量语义检索"""
        embedder = get_embedder()
        q_embedding = embedder.encode([query], normalize_embeddings=True).astype(np.float32)

        scores, indices = self.faiss_index.search(q_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.kb):
                results.append({
                    'entry': self.kb[idx],
                    'score': float(score),
                    'source': 'vector'
                })
        return results

    def keyword_search(self, query: str, top_k: int = 10) -> list:
        """BM25关键词检索"""
        try:
            import jieba
            tokenized = list(jieba.cut(query))
        except:
            tokenized = query.split()

        scores = self.bm25.get_scores(tokenized)

        max_score = max(scores) if max(scores) > 0 else 1
        norm_scores = scores / max_score

        top_indices = np.argsort(norm_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if norm_scores[idx] > 0:
                results.append({
                    'entry': self.kb[idx],
                    'score': float(norm_scores[idx]),
                    'source': 'bm25'
                })
        return results

    def hybrid_search(self, query: str, intent: dict = None, top_k: int = 5) -> list:
        """混合检索：向量 + BM25 + 意图过滤（RRF融合）"""
        vector_results = self.vector_search(query, top_k=top_k * 2)
        keyword_results = self.keyword_search(query, top_k=top_k * 2)

        # RRF融合 + 保留原始向量分数（用于置信度展示）
        scores = {}
        vector_scores = {}  # eid → 原始余弦相似度 (0~1)
        for rank, r in enumerate(vector_results):
            eid = r['entry']['id']
            scores[eid] = scores.get(eid, 0) + 1.0 / (rank + 60)
            scores[f"{eid}_entry"] = r
            # 保留最高向量分数
            if eid not in vector_scores or r['score'] > vector_scores[eid]:
                vector_scores[eid] = r['score']

        for rank, r in enumerate(keyword_results):
            eid = r['entry']['id']
            scores[eid] = scores.get(eid, 0) + 1.0 / (rank + 60)
            scores[f"{eid}_entry"] = r

        # 意图加权
        if intent and intent.get('confidence', 0) >= 0.5:
            intent_cat1 = intent.get('intent_name', '')
            intent_cat2 = intent.get('sub_intent', '')
            for key in list(scores.keys()):
                if isinstance(key, str) and key.endswith('_entry'):
                    eid = int(key.replace('_entry', ''))
                    entry = scores[key]['entry']
                    if entry['一级域'] == intent_cat1:
                        scores[eid] = scores.get(eid, 0) * 1.3
                    if intent_cat2 and entry['二级标签'] == intent_cat2:
                        scores[eid] = scores.get(eid, 0) * 1.5

        # 排序取top-k
        entry_ids = [k for k in scores if isinstance(k, int)]
        entry_ids.sort(key=lambda x: scores[x], reverse=True)

        results = []
        for eid in entry_ids[:top_k]:
            entry_key = f"{eid}_entry"
            if entry_key in scores:
                results.append({
                    'entry': scores[entry_key]['entry'],
                    'score': scores[eid],
                    'vector_score': round(vector_scores.get(eid, 0.0), 4),
                    'source': 'hybrid'
                })

        return results


# 单例
_retriever_instance = None

def get_retriever() -> HybridRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance
