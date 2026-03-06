from collections import Counter
import math

class SimpleEmbedding:
    def __init__(self):
        self.vocab = {}

    def build_vocab(self, texts):
        """Builds vocabulary from all texts"""
        word_set = set()
        for text in texts:
            words = text.lower().split()
            word_set.update(words)
        self.vocab = {word: i for i, word in enumerate(sorted(word_set))}

    def embed_text(self, text):
        """Returns a vector representation of a single text"""
        vector = [0] * len(self.vocab)
        words = text.lower().split()
        counts = Counter(words)
        for word, count in counts.items():
            if word in self.vocab:
                vector[self.vocab[word]] = count
        return vector

    def embed_texts(self, texts):
        """Embed multiple texts at once"""
        return [self.embed_text(text) for text in texts]
