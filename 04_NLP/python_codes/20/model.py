import torch
from transformers import AutoModelForSequenceClassification

def load_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1 )

    return model