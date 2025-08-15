import re
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, file_path, num_samples=None):
        self.english = []
        self.arabic = []
        self._load_data(file_path, num_samples)
    
    def _load_data(self, file_path, num_samples):
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if num_samples and i >= num_samples:
                    break
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    self.english.append(self._clean_text(parts[0]))
                    self.arabic.append('\t' + self._clean_text(parts[1]) + '\n')  # Add start/end tokens
    
    def _clean_text(self, text):
        text = re.sub(r"([.!?,])", r" \1", text.lower())
        return re.sub(r"\s+", " ", text).strip()
    
    def __len__(self):
        return len(self.english)
    
    def __getitem__(self, idx):
        return self.english[idx], self.arabic[idx]