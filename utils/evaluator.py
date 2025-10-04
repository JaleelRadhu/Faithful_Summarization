from evaluation.rouge_eval import calculate_rouge_scores

def get_rouge_l_score(reference: str, candidate: str) -> float:
    from evaluation.rouge_eval import calculate_rouge_scores
    scores = calculate_rouge_scores(reference, candidate)
    return scores['rouge-l']['f']


def get_evaluator(evaluator_name: str):
    if evaluator_name == "rouge-l":
        return get_rouge_l_score
    else:
        raise ValueError(f"Unsupported evaluator: {evaluator_name}")