import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

def load_model(model_name, hf_token=None, quantization="8bit"):
    bnb_config = None
    if quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, 
                                                device_map="auto", quantization_config=bnb_config)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)
