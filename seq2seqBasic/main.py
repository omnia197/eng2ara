import torch
from fastapi import FastAPI
from pydantic import BaseModel
from pyngrok import ngrok
import nest_asyncio
import uvicorn

#running uvicorn
nest_asyncio.apply()

#Define your Vocabulary and Model classes if needed
#Make sure you have Vocabulary, Encoder, Decoder, Seq2Seq defined/imported

from Vocabulary import Vocabulary
from model import Encoder, Decoder, Seq2Seq  #all the pipeline model structure in one .py file

# Load vocabularies
eng_vocab = Vocabulary.load("translation_pt/vocab_eng.pkl")
ara_vocab = Vocabulary.load("translation_pt/vocab_ara.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
encoder = Encoder(len(eng_vocab), 256, 512)
decoder = Decoder(len(ara_vocab), 256, 512)
model = Seq2Seq(encoder, decoder).to(DEVICE)
model.load_state_dict(torch.load("translation_pt/model_weights.pth", map_location=DEVICE))
model.eval()

# FastAPI app
app = FastAPI()

class InputText(BaseModel):
    sentence: str

def translate(sentence: str, max_len: int = 50):
    with torch.no_grad():
        tokens = sentence.lower().split()
        indices = [eng_vocab.word2idx.get(word, eng_vocab.word2idx["<UNK>"]) for word in tokens]
        src_tensor = torch.LongTensor([[eng_vocab.word2idx["<SOS>"]] + indices + [eng_vocab.word2idx["<EOS>"]]]).to(DEVICE)

        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        x = torch.tensor([[ara_vocab.word2idx["<SOS>"]]]).to(DEVICE)

        translated = []
        for _ in range(max_len):
            output, hidden, cell, _ = model.decoder(x, hidden, cell, encoder_outputs)
            pred_token = output.argmax(1).item()
            if pred_token == ara_vocab.word2idx["<EOS>"]:
                break
            translated.append(ara_vocab.idx2word.get(pred_token, "<UNK>"))
            x = torch.tensor([[pred_token]]).to(DEVICE)

        return " ".join(translated)

@app.post("/translate/")
async def translate_text(input_text: InputText):
    translation = translate(input_text.sentence)
    return {"translation": translation}


#====================================================================

nest_asyncio.apply()

#ngrok tunnel
public_url = ngrok.connect(8000)
print("The website is available at:", public_url.public_url + "/docs")

#running the application
uvicorn.run(app, host="0.0.0.0", port=8000)
