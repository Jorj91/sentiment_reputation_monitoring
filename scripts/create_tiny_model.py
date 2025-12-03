
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

os.makedirs("local_model", exist_ok=True)

model_id = "hf-internal-testing/tiny-random-distilbert"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

tok.save_pretrained("local_model")
model.save_pretrained("local_model")

print("Tiny model created successfully!")
