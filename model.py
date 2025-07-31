#encoder, decoder , seq2seq , attention in one class to be used by one line 
import torch.nn as nn
import torch

import pickle
from collections import Counter

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        hidden = hidden.permute(1, 0, 2)
        hidden_repeated = hidden.repeat(1, encoder_outputs.shape[1], 1)
        energy = torch.tanh(self.attn(torch.cat((hidden_repeated, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden, cell, encoder_outputs):
        embedded = self.embedding(x)
        attn_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell, attn_weights
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(source.device)
        encoder_outputs, hidden, cell = self.encoder(source)
        
        x = target[:, 0].unsqueeze(1)
        
        for t in range(1, target_len):
            output, hidden, cell, _ = self.decoder(x, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1)
            x = target[:, t].unsqueeze(1) if torch.rand(1) < teacher_forcing_ratio else top1.unsqueeze(1)
        
        return outputs
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
    
    @classmethod
    def load(cls, encoder, decoder, filepath, device='cpu'):
        model = cls(encoder, decoder)
        model.load_state_dict(torch.load(filepath, map_location=device))
        return model