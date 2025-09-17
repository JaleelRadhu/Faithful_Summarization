import os
# import yaml
from typing import Dict, Any



class LLMClient:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.model_name = cfg['model']['name']
        self.backend = cfg['model'].get('backend', 'hf')
        self.device = cfg['model'].get('device', '0')
        self.max_new_tokens = cfg['model'].get('max_new_tokens', 512)
        self.temperature = cfg['model'].get('temperature', 0.7)
        
        if self.backend == 'hf':
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            hf_token = os.getenv("HF_TOKEN")
            
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token = hf_token,
                device_map="auto" if self.device not in ["cup", -1] else None,
            )
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer)      
        elif self.backend == 'openai':
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.client = openai
            
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        
        def generate(self, prompt: str, **kwargs) -> str:
            params = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            }
            params.update(kwargs)   
            if self.backend == 'hf':
                outputs = self.pipe(
                    prompt,
                    **params
                )
                return outputs[0]['generated_text']
            elif self.backend == 'openai':
                response = self.client.Completion.create(
                    model=self.model_name,
                    message=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature
                )
                return response.choices[0].text.strip()
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")    
    