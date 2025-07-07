from pathlib import Path

model_path = Path("models/distilbert_sentiment_model")

if not model_path.exists():
    print("Training model from scratch...")

    import os 
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, f1_score
    from datasets import Dataset
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        Trainer,
        TrainingArguments
    )

    df = pd.read_csv("data/cleaned_reviews.csv")
    df = df[['clean_text', 'label']].rename(columns={'clean_text': 'text'})
    df['label'] = df['label'].astype(int)
    df = df.sample(200, random_state=42)

    hf_dataset = Dataset.from_pandas(df)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)

    tokenized_dataset = hf_dataset.map(tokenize, batched=True)
    train_test = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test['train']
    eval_dataset = train_test['test']

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds)
        }

    training_args = TrainingArguments(
        output_dir=str(model_path),
        evaluation_strategy="epoch",
        logging_dir='./logs',
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))

else:
    print(f"Model already exists at {model_path}. Skipping training.")

