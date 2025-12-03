# script to create model for CI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

# Save it locally
model.save_pretrained("local_model")
tokenizer.save_pretrained("local_model")