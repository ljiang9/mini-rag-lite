"""可选的 LLM 答案生成（OpenAI 兼容接口，标准库 urllib 实现）。

未设置 OPENAI_API_KEY 时不可用，调用方应直接输出检索片段。
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import List

from .retriever import Chunk

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


def llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def generate_answer(query: str, chunks: List[Chunk]) -> str:
    """结合检索到的片段，让 LLM 生成一段答案。无 key 时抛错。"""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY，无法生成答案")
    base_url = (os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).strip()
    model = os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL

    context = "\n\n".join(f"[来自 {c.source}] {c.text}" for c in chunks)
    system = "你是一个助手，请只根据下面提供的上下文回答用户问题；若上下文不足，请如实说明。"
    user = f"上下文：\n{context}\n\n问题：{query}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]
