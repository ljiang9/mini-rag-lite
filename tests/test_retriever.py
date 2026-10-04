import os
import tempfile
import unittest

from mini_rag.retriever import Retriever


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        with open(os.path.join(self.dir, "a.txt"), "w", encoding="utf-8") as f:
            f.write("Python 是一门解释型编程语言，语法简洁，内置电池齐全。" * 5)
        with open(os.path.join(self.dir, "b.txt"), "w", encoding="utf-8") as f:
            f.write("公司远程办公政策规定每周最多两天居家办公，需提前申请。" * 5)

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_and_chunks(self):
        r = Retriever(self.dir, chunk_size=60, overlap=10)
        self.assertTrue(len(r.chunks) > 2)
        sources = {c.source for c in r.chunks}
        self.assertEqual(sources, {"a.txt", "b.txt"})

    def test_search_python_query(self):
        r = Retriever(self.dir, chunk_size=60, overlap=10)
        results = r.search("Python 编程语言", k=2)
        self.assertEqual(len(results), 2)
        # 最相关片段应来自 a.txt
        self.assertEqual(results[0].source, "a.txt")
        self.assertGreater(results[0].score, 0)

    def test_search_policy_query(self):
        r = Retriever(self.dir, chunk_size=60, overlap=10)
        results = r.search("远程办公 居家", k=1)
        self.assertEqual(results[0].source, "b.txt")

    def test_empty_dir_raises(self):
        empty = tempfile.TemporaryDirectory()
        try:
            with self.assertRaises(FileNotFoundError):
                Retriever(empty.name)
        finally:
            empty.cleanup()


if __name__ == "__main__":
    unittest.main()
