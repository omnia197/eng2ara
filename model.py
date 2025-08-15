import pandas as pd
from datasets import Dataset
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

class TranslationModel:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-ar"):
        self.model_name = model_name
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def train(self, tokenized_dataset, output_dir="./eng2ar_model", epochs=3, batch_size=16, lr=5e-5):
        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=2,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=True
        )
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def translate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        outputs = self.model.generate(**inputs, max_length=128)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
