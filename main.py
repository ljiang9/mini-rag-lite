"""命令行入口：python3 main.py "你的问题" [--k 3]

无 API key 时直接输出检索到的片段；有 OPENAI_API_KEY 时额外生成摘要答案。
"""

from __future__ import annotations

import argparse
import os

from mini_rag.retriever import Retriever
from mini_rag.llm import llm_available, generate_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-rag-lite：零依赖 TF-IDF 检索增强原型")
    parser.add_argument("query", help="检索问题")
    parser.add_argument("--k", type=int, default=3, help="返回的片段数量，默认 3")
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--docs", default=os.path.join(here, "docs"), help="文档目录")
    args = parser.parse_args()

    retriever = Retriever(args.docs)
    results = retriever.search(args.query, k=args.k)

    print(f"检索到 {len(results)} 个相关片段：")
    for i, c in enumerate(results, 1):
        print(f"\n[{i}] (来源: {c.source}, 相似度: {c.score:.4f})")
        print(c.text)

    if llm_available():
        print("\n=== LLM 生成答案 ===")
        print(generate_answer(args.query, results))
    else:
        print("\n（未检测到 OPENAI_API_KEY：本地模式，以上为检索到的原始片段）")


if __name__ == "__main__":
    main()
