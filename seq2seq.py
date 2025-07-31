import torch
import torch.nn as nn

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