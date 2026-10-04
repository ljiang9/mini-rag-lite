"""纯 Python 实现的 TF-IDF 向量化 + cosine 相似度检索。

不依赖 numpy / sklearn：词袋用 dict 表示，向量归一化与点积均手算。
分词同时兼顾英文单词与中文单字，并为中文补充相邻二字（bigram）以提升召回。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    """把文本切成 token 列表：英文单词 + 中文单字 + 中文相邻二字组。"""
    text = text.lower()
    tokens = _TOKEN_RE.findall(text)
    # 为连续中文串补充 bigram，缓解单字歧义
    bigrams: List[str] = []
    for run in _CJK_RUN_RE.findall(text):
        for i in range(len(run) - 1):
            bigrams.append(run[i:i + 2])
    return tokens + bigrams


def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    dot = sum(w * v2.get(t, 0.0) for t, w in v1.items())
    return dot  # 入库时向量已归一化为单位向量，点积即 cosine


class TfidfIndex:
    """对一组文档片段建立 TF-IDF 索引并提供 top-k 检索。"""

    def __init__(self, documents: List[str]) -> None:
        if not documents:
            raise ValueError("documents 不能为空")
        self.documents = documents
        self.N = len(documents)
        self._doc_tokens = [tokenize(d) for d in documents]

        # 文档频率 df
        df: Dict[str, int] = defaultdict(int)
        for toks in self._doc_tokens:
            for term in set(toks):
                df[term] += 1
        # 平滑 idf
        self.idf: Dict[str, float] = {
            t: math.log((self.N + 1) / (d + 1)) + 1.0 for t, d in df.items()
        }
        # 预先把每个文档归一化为单位向量
        self.doc_vecs = [self._to_unit_vector(toks) for toks in self._doc_tokens]

    def _to_unit_vector(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        length = len(tokens)
        raw: Dict[str, float] = {}
        for term, count in tf.items():
            idf = self.idf.get(term)
            if idf is not None:
                raw[term] = (count / length) * idf
        norm = math.sqrt(sum(w * w for w in raw.values()))
        if norm == 0:
            return {}
        return {t: w / norm for t, w in raw.items()}

    def query_vector(self, query: str) -> Dict[str, float]:
        return self._to_unit_vector(tokenize(query))

    def search(self, query: str, k: int = 3) -> List[Tuple[float, int]]:
        """返回 top-k (相似度, 文档下标)，按相似度降序。"""
        qv = self.query_vector(query)
        if not qv:
            return []
        scored = [(_cosine(qv, dv), i) for i, dv in enumerate(self.doc_vecs)]
        scored = [(s, i) for s, i in scored if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]
