from evaluation.rouge_eval import calculate_rouge_scores
from pipeline.feedback_generator import generate_feedback
from pipeline.summary_improver import revise_summary

def run_trial(summary_data, score_prompt, corrector_prompt, pipe, max_iterations=5, tolerance=0.01):
    previous_summary = summary_data["predicted"]
    previous_rouge_score = None

    for iteration in range(max_iterations):
        feedback = generate_feedback(summary_data, score_prompt, pipe)
        cot, revised_summary = revise_summary(summary_data, feedback, corrector_prompt, pipe)

        scores = calculate_rouge_scores(summary_data["Actual"], revised_summary)
        current_rouge = scores['rouge-l']['f']

        if previous_rouge_score is not None and current_rouge <= previous_rouge_score:
            break
        previous_rouge_score = current_rouge
        summary_data["predicted"] = revised_summary
    
    return feedback, revised_summary, cot, scores
