from typing import Any, Dict

from faith_improver.llm_client import LLMClient

class Improver:
    
    def __init__(self, cfg: Dict[str, Any], llm_client: LLMClient):
        self.cfg = cfg
        self.llm_client = llm_client
        self.prompt_template = cfg.get('prompt_template', "Improve the following text for faithfulness:\n\n{text}\n\nImproved text:")
        
    def improve(self, summary_data: Dict[Any]) -> str:
        raise NotImplementedError("Function improve is not implemented yet.")
        
        return refined_summary
