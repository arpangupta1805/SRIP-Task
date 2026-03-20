import torch
import torch.nn as nn

class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.fc = nn.Linear(d_model, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        mask = (text != 0).float().unsqueeze(-1)
        embedded = self.embedding(text)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.fc(pooled)
