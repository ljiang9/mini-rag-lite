import unittest

from mini_rag.chunking import chunk_text


class TestChunking(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   "), [])

    def test_short_text_single_chunk(self):
        out = chunk_text("短文本", chunk_size=20, overlap=5)
        self.assertEqual(out, ["短文本"])

    def test_chunk_size_bound(self):
        text = "甲" * 250
        out = chunk_text(text, chunk_size=100, overlap=20)
        self.assertTrue(len(out) > 1)
        for c in out:
            self.assertLessEqual(len(c), 100)

    def test_overlap(self):
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        out = chunk_text(text, chunk_size=10, overlap=3)
        # step = 7；第一个片段 ABCDEFGHIJ，第二个从索引 7 开始 HIJKLMNOPQ
        self.assertEqual(out[0], "ABCDEFGHIJ")
        self.assertEqual(out[1], "HIJKLMNOPQ")

    def test_overlap_too_large(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", chunk_size=10, overlap=10)

    def test_last_chunk_reaches_end(self):
        text = "测试内容" * 100
        out = chunk_text(text, chunk_size=50, overlap=10)
        # 最后一个片段必须以原文最后一个字符结尾
        self.assertEqual(out[-1][-1], text[-1])


if __name__ == "__main__":
    unittest.main()
