import pyarrow.parquet as pq
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter
import re
import json

def basic_tokenize(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return text.split()

class TextVocab:
    def __init__(self, max_size=50000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.vocab_built = False

    def build_vocab(self, texts):
        print("Building vocabulary...")
        counter = Counter()
        for text in texts:
            counter.update(basic_tokenize(text))
                
        common_words = [w for w, c in counter.most_common(self.max_size) if c >= self.min_freq]
        for w in common_words:
            if w not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[w] = idx
                self.idx2word[idx] = w
        self.vocab_built = True
        print(f"Vocab size: {len(self.word2idx)}")
    
    def encode(self, text, max_len=128):
        tokens = basic_tokenize(text)
        token_ids = [self.word2idx.get(w, self.word2idx['<unk>']) for w in tokens][:max_len]
        if len(token_ids) < max_len:
            token_ids += [self.word2idx['<pad>']] * (max_len - len(token_ids))
        return token_ids

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({'word2idx': self.word2idx}, f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = {int(v): k for k, v in self.word2idx.items()}
            self.vocab_built = True

class ParquetDataset(Dataset):
    def __init__(self, df, vocab, topic2idx, max_len=128):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.topic2idx = topic2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_encoded = self.vocab.encode(row['DATA'], self.max_len)
        label = self.topic2idx[row['TOPIC']]
        return torch.tensor(text_encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)
