from rouge import Rouge

def get_rouge_l_score(reference: str, candidate: str) -> float:
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)[0]
    # print(scores)
    return scores['rouge-l']['f']


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