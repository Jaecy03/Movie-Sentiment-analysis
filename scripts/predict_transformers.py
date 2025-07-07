import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

model_path = "models/distilbert_sentiment_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

texts = [
    "I was absolutely abysmal.",
    "It was absolutely fantastic!",
    "Loved the actors and the plot.",
    "The movie was a total disaster.",
    "Very slow and boring."
]

label_map = {0: "negative", 1: "positive"}

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred = outputs.logits.argmax().item()
    conf = torch.softmax(outputs.logits, dim=1).max().item()
    print(f"'{text}' -> {label_map[pred]} (confidence: {conf:.2f})")
