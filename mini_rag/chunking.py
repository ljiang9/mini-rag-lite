"""带重叠窗口的文本切分。

以字符为单位做滑动窗口切分：窗口大小 chunk_size，相邻窗口重叠 overlap 个字符。
实现简单、确定性强，便于单测；对中文文档同样适用。
"""

from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """把长文本切分成若干带重叠的片段。

    Args:
        text: 原文。
        chunk_size: 每个片段的最大字符数。
        overlap: 相邻片段重叠的字符数，必须小于 chunk_size。

    Returns:
        片段列表；空文本返回空列表。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if overlap < 0:
        raise ValueError("overlap 不能为负")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    text = text.strip()
    if not text:
        return []

    step = chunk_size - overlap
    chunks: List[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start += step
    return chunks
