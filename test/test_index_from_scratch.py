import unittest

from index.index_from_scratch import VectorIndexFromScratchImplementation


class TestVectorIndexFromScratchImplementation(unittest.TestCase):

    def setUp(self):
        self.index = VectorIndexFromScratchImplementation(dimension=3)
        self.index.add(
            embeddings=[[2, 3, 4], [1, 1, 1], [7, 8, 9]],
            chunks=["Chunk A", "Chunk B", "Chunk C"]
        )

    # Basic functionality test
    def test_basic_search(self):
        result = self.index.search([2, 3, 5], top_k=1)
        self.assertEqual(result, ["Chunk A"])

    # 2Multiple results test
    def test_multiple_results(self):
        result = self.index.search([2, 3, 5], top_k=2)
        self.assertEqual(result[0], "Chunk A")
        self.assertEqual(len(result), 2)

    # Exact match test
    def test_exact_match(self):
        result = self.index.search([1, 1, 1], top_k=1)
        self.assertEqual(result, ["Chunk B"])

    # Dimension mismatch on search
    def test_query_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            self.index.search([1, 2], top_k=1)

    # Empty index test
    def test_empty_index(self):
        empty_index = VectorIndexFromScratchImplementation(dimension=3)
        result = empty_index.search([1, 2, 3], top_k=1)
        self.assertEqual(result, [])

    # top_k greater than number of vectors
    def test_top_k_greater_than_available(self):
        result = self.index.search([2, 3, 5], top_k=10)
        self.assertEqual(len(result), 3)

    # Single vector test
    def test_single_vector(self):
        single_index = VectorIndexFromScratchImplementation(dimension=2)
        single_index.add(
            embeddings=[[5, 5]],
            chunks=["Only Chunk"]
        )
        result = single_index.search([5, 6], top_k=1)
        self.assertEqual(result, ["Only Chunk"])

    # Negative numbers test
    def test_negative_numbers(self):
        index2 = VectorIndexFromScratchImplementation(dimension=2)
        index2.add(
            embeddings=[[-1, -1], [1, 1]],
            chunks=["Negative", "Positive"]
        )
        result = index2.search([0, 0], top_k=2)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()

