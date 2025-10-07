import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def load_model(model_name, hf_token=None, quantization="8bit"):
    print(f"Loading model: {model_name}")
    print(f"Quantization: {quantization}")
    
    bnb_config = None
    if quantization == "4bit":
        print("Configuring 4-bit quantization (NF4)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # CRITICAL: compute dtype
            bnb_4bit_quant_type="nf4",              # CRITICAL: use NF4 quantization
            bnb_4bit_use_double_quant=True,        # CRITICAL: nested quantization
        )
    elif quantization == "8bit":
        print("Configuring 8-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        token=hf_token, 
        device_map="auto", 
        quantization_config=bnb_config,
        torch_dtype="auto"
    )
    
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3    # GB
        print(f"✅ Model loaded successfully!")
        print(f"   GPU Memory Allocated: {memory_allocated:.2f} GB")
        print(f"   GPU Memory Reserved: {memory_reserved:.2f} GB")
    
    # Return model and tokenizer separately (not pipeline)
    return model, tokenizer