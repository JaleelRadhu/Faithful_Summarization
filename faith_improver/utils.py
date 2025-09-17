

class Evaluator():
    
    def __init__(self, llm_client, eval_prompt):
        self.llm_client = llm_client
        self.eval_prompt = eval_prompt