#here in this class it is the source from building the model weights file .pth and all training loops
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq
from Vocabulary import Vocabulary
from dataset import TranslationDataset

DATA_PATH = "ara.txt"
OUTPUT_DIR = "translation_pt"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BATCH_SIZE = 32
EMBED_SIZE = 256
HIDDEN_SIZE = 512
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn(batch, eng_vocab, ara_vocab):
    src, trg = zip(*batch)
    
    #entences to indices
    src_indices = [
        [eng_vocab.word2idx['<SOS>']] + eng_vocab.sentence_to_indices(s) + [eng_vocab.word2idx['<EOS>']]
        for s in src
    ]
    trg_indices = [
        [ara_vocab.word2idx['<SOS>']] + ara_vocab.sentence_to_indices(t) + [ara_vocab.word2idx['<EOS>']]
        for t in trg
    ]

    src_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor(s) for s in src_indices],
        padding_value=eng_vocab.word2idx['<PAD>'],
        batch_first=True
    )
    trg_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor(t) for t in trg_indices],
        padding_value=ara_vocab.word2idx['<PAD>'],
        batch_first=True
    )
    
    return src_padded.to(DEVICE), trg_padded.to(DEVICE)

def main():

    dataset = TranslationDataset(DATA_PATH)
    eng_vocab = Vocabulary()
    eng_vocab.build_vocab([eng for eng, _ in dataset])
    eng_vocab.save(f"{OUTPUT_DIR}/vocab_eng.pkl")
    
    ara_vocab = Vocabulary()
    ara_vocab.build_vocab([ara for _, ara in dataset])
    ara_vocab.save(f"{OUTPUT_DIR}/vocab_ara.pkl")

    encoder = Encoder(len(eng_vocab.word2idx), EMBED_SIZE, HIDDEN_SIZE)
    decoder = Decoder(len(ara_vocab.word2idx), EMBED_SIZE, HIDDEN_SIZE)
    model = Seq2Seq(encoder, decoder).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=ara_vocab.word2idx['<PAD>'])
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, eng_vocab, ara_vocab)
    )
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for src, trg in dataloader:
            optimizer.zero_grad()
            output = model(src, trg)
            
            output = output[:, 1:].reshape(-1, output.shape[2])
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(dataloader):.4f}")

    model.save(f"{OUTPUT_DIR}/model_weights.pth")
    print(f"Assets saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()