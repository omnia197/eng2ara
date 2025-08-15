from model import TranslationModel
from translation_data import TranslationDataset
from transformers import MarianTokenizer


dataset = TranslationDataset(file_path="hf://datasets/salehalmansour/english-to-arabic-translate/en_ar_final.tsv")
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
tokenized = dataset.preprocess(tokenizer)


translator = TranslationModel()

test_sentences = [
    "Hello, how are you?",
    "I am learning data science and AI.",
    "The weather today is sunny and warm.",
    "Can you help me with my homework?",
    "NASA is exploring Mars and other planets.",
    "I love reading books about history and science.",
    "We should save water and protect the environment.",
    "Artificial intelligence is transforming the world.",
    "The cat is sleeping on the sofa.",
    "Tomorrow, I will go to the market to buy fruits."
]

for sentence in test_sentences:
    arabic_translation = translator.translate(sentence)
    print(f"EN: {sentence}")
    print(f"AR: {arabic_translation}\n")
