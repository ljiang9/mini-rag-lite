# mini-rag-lite

一个**零第三方依赖**的轻量 RAG（检索增强）原型：读取 `docs/` 下的 `.txt` 文档，用带重叠窗口的方式切分，纯 Python 实现 **TF-IDF + cosine 相似度**检索 top-k 片段；有 API Key 时可调用 LLM 生成答案，无 Key 时直接输出检索到的原文片段。

## 功能特性

- **文档切分**：字符级滑动窗口，相邻片段带重叠（`chunk_size` / `overlap` 可调）。
- **TF-IDF**：不依赖 numpy/sklearn，词袋用 dict 手算，含平滑 IDF。
- **中文友好**：分词兼顾英文单词与中文单字，并补充中文相邻二字（bigram）提升召回。
- **cosine 相似度**：入库前向量归一化，点积即相似度。
- **可选 LLM 生成**：配置 `OPENAI_API_KEY` 后，结合检索片段生成自然语言答案；否则直接打印片段。

## 快速开始

环境要求：Python 3.10+（开发于 3.12）。无需安装任何依赖。

```bash
cd mini-rag-lite

# 无 Key：本地模式，直接输出检索到的片段
python3 main.py "Python 有哪些应用领域"

# 返回更多片段
python3 main.py "远程办公一周可以几天" --k 2
```

### 接入 LLM 生成答案（可选）

```bash
export OPENAI_API_KEY="sk-xxxx"
export OPENAI_BASE_URL="https://api.openai.com/v1"   # 可选
export OPENAI_MODEL="gpt-4o-mini"                    # 可选
python3 main.py "什么是 Python"
```

## 使用示例

无 Key 时的实际运行效果：

```
$ python3 main.py "Python 为什么适合机器学习"
检索到 3 个相关片段：

[1] (来源: about_python.txt, 相似度: 0.1234)
Python 在数据科学、机器学习、Web 开发、自动化脚本等领域应用广泛。
NumPy、pandas、PyTorch 等第三方生态使其成为人工智能领域最流行的语言之一。
...
（未检测到 OPENAI_API_KEY：本地模式，以上为检索到的原始片段）
```

## 无 API Key 如何运行？

直接 `python3 main.py "问题"` 即可。检索（切分 → TF-IDF → cosine top-k）全程本地完成，不调用任何网络；程序会在末尾提示「本地模式」。配置 `OPENAI_API_KEY` 后才会额外请求 LLM 生成答案。

## 运行测试

```bash
python3 -m unittest discover -s tests
```

覆盖：切分正确性与重叠窗口、TF-IDF 向量归一化与排序、检索召回（含临时目录建索引）。

## 目录结构

```
mini-rag-lite/
├── main.py               # 命令行入口
├── mini_rag/
│   ├── __init__.py
│   ├── chunking.py       # 带重叠窗口的切分
│   ├── tfidf.py          # 分词 + TF-IDF + cosine 检索
│   ├── retriever.py      # 加载 docs/、切分、建索引、top-k
│   └── llm.py            # 可选：OpenAI 兼容接口生成答案
├── docs/
│   ├── about_python.txt  # 示例文档 1
│   └── company_policy.txt# 示例文档 2
├── tests/
│   ├── test_chunking.py
│   ├── test_tfidf.py
│   └── test_retriever.py
├── README.md
├── LICENSE               # MIT
└── .gitignore
```

## 许可证

[MIT](./LICENSE) © ljiang9
