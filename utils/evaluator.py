from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from bert_score import score
import re
import torch
import yaml
# from pipeline.feedback_generator import generate_feedback
# from bart_score import BARTScorer




def get_rouge_l_score(reference: str, candidate: str) -> float:
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)[0]
    # print(scores)
    return scores['rouge-l']['f']

def get_rouge_1_score(reference: str, candidate: str) -> float:
    """returns the f1 of Rouge 1"""
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)[0]
    return scores['rouge-1']['f']

def get_rouge_2_score(reference: str, candidate: str) -> float:
    """returns the f1 of Rouge 2"""
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)[0]
    return scores['rouge-2']['f']

def get_meteor_score(reference: str, candidate: str) -> float:
    """returns the meteor score"""
    # Meteor score expects tokenized sentences
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    return meteor_score([reference_tokens], candidate_tokens)

def get_BERTScore(reference: str, candidate: str) -> float:
    """returns the BERTScore"""
    P, R, F1 = score([candidate], [reference], lang="en", verbose=False)
    return F1.item()

# Global instance for BARTScorer to avoid re-loading the model for each call
_bart_scorer_instance = None

def get_BARTScore(reference: str, candidate: str) -> float:
    """returns the BARTScore"""
    global _bart_scorer_instance
    if _bart_scorer_instance is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _bart_scorer_instance = BARTScorer(device=device, max_length=1024)
        # Optionally, you can load a specific BART model checkpoint if needed:
        # _bart_scorer_instance.load(path='path/to/bart.pth')
    scores = _bart_scorer_instance.score(candidates=[candidate], references=[reference], batch_size=4)
    return scores[0] # The score method returns a list of scores, we need the first (and only) one
    
def get_llm_metrics(llm_feedback: str) -> dict[str, dict]:
    """Given the llm feedback, it extracts the reasoning and score for each metric.

    This function uses a regular expression to find all occurrences of
    "Reasoning for [Criterion]: [Text]... Score for [Criterion]: [Score]"
    and returns a nested dictionary.

    Args:
        llm_feedback: A string containing the feedback from the LLM.

    Returns:
        A dictionary where keys are normalized criteria names (e.g.,
        "perspective_misalignment") and values are dictionaries containing
        the 'reasoning' (str) and 'score' (int).
    """
    # This pattern captures the criterion name, the reasoning text, and the score.
    # re.DOTALL is crucial for the 'reasoning' part to match across newlines.
    pattern = re.compile(
        r"\*\*Reasoning for (.+?):\*\* (.*?)\n\*\*Score for \1:\*\* (\d+)",
        re.DOTALL
    )
    matches = pattern.findall(llm_feedback)
    
    metrics = {}
    for criterion, reasoning, score in matches:
        normalized_key = criterion.strip().lower().replace(' ', '_')
        metrics[normalized_key] = {
            'reasoning': reasoning.strip(),
            'score': int(score)
        }
    return metrics

def get_llm_metrics_from_a_model(summary_data: dict, llm_pipe): 
    with open("configs/default.yaml", 'r') as f:
        config = yaml.safe_load(f)
    with open(config['evaluator_prompt_path'], 'r') as f:
        evaluator_prompt = f.read()
    
    feedback = generate_feedback(summary_data, evaluator_prompt, llm_pipe)
    return get_llm_metrics(feedback)






def get_evaluator(evaluator_name: str):
    if evaluator_name == "rouge-l-f":
        return get_rouge_l_score
    else:
        raise ValueError(f"Unsupported evaluator: {evaluator_name}")
    
if __name__ == "__main__":
    # Example usage
    ref = "The cat sat on the mat."
    cand = "The cat is sitting on the mat."
    score = get_rouge_l_score(ref, cand)
    print(f"ROUGE-L F1 Score: {score}")
    
    ex_feedback = "**Reasoning for Extraneous Information:** The summary does not include any information that is not present in the source text or answers. It accurately reflects the personal experience shared about the Atkins diet and its impact on the individual's health.\n**Score for Extraneous Information:** 5\n\n**Reasoning for Contradiction:** The summary does not contradict any of the provided answers or the source text. It mentions the same points about the Atkins diet helping some people but being unsuitable for the individual due to kidney problems.\n**Score for Contradiction:** 5\n\n**Reasoning for Perspective Misalignment:** The summary maintains the perspective of personal experience by focusing on the individual's own account of trying the Atkins diet and its effects on their health. This aligns well with the given perspective definition.\n**Score for Perspective Misalignment:** 5\n\n**Reasoning for Redundancy:** The summary is concise and does not repeat any information unnecessarily. It clearly conveys the key points without any redundancy.\n**Score for Redundancy:** 5"
    metrics = get_llm_metrics(ex_feedback)
    print(metrics)
    