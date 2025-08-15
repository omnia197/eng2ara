import pickle
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.word_counts = Counter()

    def build_vocab(self, sentences, min_freq=1):
        """Build vocabulary from sentences"""
        for sentence in sentences:
            self.word_counts.update(sentence.split())

        # Add words meeting frequency threshold
        for word, count in self.word_counts.items():
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def sentence_to_indices(self, sentence):
        """Convert sentence string to list of indices"""
        return [self.word2idx.get(word, self.word2idx['<UNK>'])
                for word in sentence.split()]

    def save(self, filepath):
        """Save vocabulary to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """Load vocabulary from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.word2idx)