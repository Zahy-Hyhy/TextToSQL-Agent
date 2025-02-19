# model.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import MODEL_NAME

def get_chat_pipeline():
    """
    Loads the Deepseek model and returns a text-generation pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)
