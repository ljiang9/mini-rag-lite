"""mini-rag-lite：纯 Python 实现的轻量检索增强原型。"""

from .chunking import chunk_text
from .tfidf import TfidfIndex, tokenize
from .retriever import Retriever

__all__ = ["chunk_text", "TfidfIndex", "tokenize", "Retriever"]
__version__ = "0.1.0"
