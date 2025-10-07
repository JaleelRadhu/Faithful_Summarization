import torch

class ChatGenerator:
    """Wrapper for generating text with chat-based models using proper chat templates"""
    
    def __init__(self, model, tokenizer, system_prompt=None):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt or "You are a helpful assistant."
    
    def __call__(self, prompt, max_new_tokens=512, temperature=0.7, return_full_text=False):
        """
        Generate text using the model with chat template formatting.
        Returns in pipeline-compatible format: [{"generated_text": "..."}]
        """
        # Format messages with chat template
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract only new tokens (remove input)
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Return in pipeline-compatible format
        return [{"generated_text": response}]
    
    def set_system_prompt(self, system_prompt):
        """Update the system prompt"""
        self.system_prompt = system_prompt