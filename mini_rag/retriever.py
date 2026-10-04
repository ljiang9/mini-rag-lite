"""把 docs/ 目录下的 txt 文档加载、切分、建索引，并对外提供检索接口。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from .chunking import chunk_text
from .tfidf import TfidfIndex


@dataclass
class Chunk:
    """一个检索到的文本片段。"""
    text: str
    source: str      # 来源文件名
    score: float     # 相似度


class Retriever:
    """加载目录文档 -> 切分 -> TF-IDF 索引 -> top-k 检索。"""

    def __init__(self, docs_dir: str, chunk_size: int = 200, overlap: int = 50) -> None:
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[Chunk] = []
        self._index: Optional[TfidfIndex] = None
        self.build()

    def build(self) -> None:
        """读取 docs_dir 下所有 *.txt，切分并建立索引。"""
        texts: List[str] = []
        sources: List[str] = []
        if os.path.isdir(self.docs_dir):
            for name in sorted(os.listdir(self.docs_dir)):
                if not name.endswith(".txt"):
                    continue
                path = os.path.join(self.docs_dir, name)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                for piece in chunk_text(content, self.chunk_size, self.overlap):
                    texts.append(piece)
                    sources.append(name)

        if not texts:
            raise FileNotFoundError(f"目录 {self.docs_dir} 下没有可用的 .txt 文档")

        self._index = TfidfIndex(texts)
        self.chunks = [Chunk(text=t, source=s, score=0.0) for t, s in zip(texts, sources)]

    def search(self, query: str, k: int = 3) -> List[Chunk]:
        """返回 top-k 相关片段（已填入相似度分数）。"""
        assert self._index is not None, "索引尚未构建"
        results: List[Chunk] = []
        for score, idx in self._index.search(query, k=k):
            c = self.chunks[idx]
            results.append(Chunk(text=c.text, source=c.source, score=score))
        return results
