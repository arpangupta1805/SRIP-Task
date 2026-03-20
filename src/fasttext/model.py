import torch.nn as nn

class FastTextModel(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=128):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, d_model, sparse=False, mode='mean', padding_idx=0)
        self.fc = nn.Linear(d_model, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        return self.fc(embedded)
