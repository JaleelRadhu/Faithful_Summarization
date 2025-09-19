
from typing import Dict, Any

from rouge import Rouge
rouge = Rouge()

import yaml

cfg = yaml.safe_load(open("/home/abdullahm/jaleel/Faithfullness_Improver/configs/default.yaml", 'r'))

def build_evaluation_prompt(prompt_path: str, summary_data=None) -> str:
    
    raise NotImplementedError("Function build_evaluation_prompt is not implemented yet.")
    return prompt     

def build_improvement_prompt(prompt_path: str, original_text: str, evaluation_feedback: str) -> str:
    
    raise NotImplementedError("Function build_improvement_prompt is not implemented yet.")
    return prompt

def stopping_criterion_eval(original_text: str, improved_text: str) -> float:
    
    raise NotImplementedError("Function stopping_criterion_eval is not implemented yet, fully.")
    
    stopping_criterion = cfg['iteration']['stopping_criterion']
    if stopping_criterion == 'rouge-l-f':
        scores = rouge.get_scores(improved_text, original_text)
        return scores[0]['rouge-l']['f']
    elif stopping_criterion == "faithfulness_eval_1":
        return faithfulness_eval_1(original_text, improved_text)

def faithfulness_eval_1(original_text: str, improved_text: str) -> float:
    # Dummy implementation for faithfulness evaluation
    raise NotImplementedError("Function faithfulness_eval_1 is not implemented yet.")
    return 1.0 if original_text == improved_text else 0.0   

def calculate_rouge_scores(hypothesis: str, reference: str) -> Dict[str, Any]: 
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]