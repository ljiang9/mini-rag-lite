import math
import unittest

from mini_rag.tfidf import TfidfIndex, tokenize


class TestTokenize(unittest.TestCase):
    def test_ascii_words(self):
        toks = tokenize("Hello World hello")
        self.assertIn("hello", toks)
        self.assertIn("world", toks)

    def test_cjk_chars_and_bigrams(self):
        toks = tokenize("我爱编程")
        self.assertIn("我", toks)
        self.assertIn("爱编", toks)  # bigram


class TestTfidf(unittest.TestCase):
    def setUp(self):
        self.docs = [
            "Python 是一门广泛使用的编程语言",
            "公司远程办公政策每周最多两天居家办公",
            "Python 在机器学习和数据科学中很流行",
        ]
        self.index = TfidfIndex(self.docs)

    def test_unit_vectors_norm(self):
        for dv in self.index.doc_vecs:
            norm = math.sqrt(sum(w * w for w in dv.values()))
            self.assertAlmostEqual(norm, 1.0, places=6)

    def test_search_finds_relevant(self):
        results = self.index.search("Python 机器学习", k=2)
        self.assertTrue(results)
        # 第一名应为第 3 篇（含 Python 与机器学习）
        self.assertEqual(results[0][1], 2)
        self.assertGreater(results[0][0], 0)

    def test_search_unknown_returns_empty(self):
        self.assertEqual(self.index.search("zzzzqqqq", k=2), [])

    def test_search_sorted_desc(self):
        results = self.index.search("Python", k=3)
        scores = [s for s, _ in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
