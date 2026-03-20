import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=128, hidden_size=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Using bidirectional LSTM to capture context from both sides
        self.lstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hn, cn) = self.lstm(embedded)
        hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)
