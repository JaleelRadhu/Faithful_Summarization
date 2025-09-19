
from typing import Any, Dict, Tuple
from faith_improver.llm_client import LLMClient

class Evaluator:
    
    def __init__(self, cfg: Dict[str, Any], llm_client: LLMClient):
        self.cfg = cfg
        self.llm_client = llm_client
        self.prompt_template = cfg.get('prompt_template', "Evaluate the following text for faithfulness:\n\n{text}\n\nFaithfulness score (0-100):")
        self.threshold = cfg.get('threshold', 70)
        
    def evaluate(self, text: str) -> Tuple[int, bool]:
        prompt = self.prompt_template.format(text=text)
        response = self.llm_client.generate(prompt)
        
        try:
            score = int(''.join(filter(str.isdigit, response)))
        except ValueError:
            score = 0
        
        is_faithful = score >= self.threshold
        return score, is_faithful
    