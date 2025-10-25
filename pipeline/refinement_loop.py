# from evaluation.rouge_eval import calculate_rouge_scores
from pipeline.feedback_generator import generate_feedback
from pipeline.summary_improver import revise_summary
from utils.evaluator import get_evaluator

import yaml


def refinement_loop(summary_data, evaluator_prompt, improver_prompt, pipe, max_iterations, stopping_criteria, tolerance=0.01):
    previous_summary = summary_data["Predicted"]
    previous_score = None

    evaluator = get_evaluator(stopping_criteria)
    
    total_iterations = 0
    for iteration in range(max_iterations):
        
        summary_data["Given Summary"] = previous_summary
        
        #generate feedback using evaluator prompt
        feedback = generate_feedback(summary_data, evaluator_prompt, pipe)
        if iteration == 0:
            first_feedback = feedback
        # write the above in a single line
        
        # cot, revised_summary = revise_summary(summary_data, feedback, improver_prompt, pipe)
        # print("--feedback_start--"*5)
        # print(f"Iteration {iteration+1} Feedback:\n {feedback}")
        # print("--feedback_end--"*5)
        
        #generate revised summary using improver prompt
        improved_summary_data = revise_summary(summary_data, feedback, improver_prompt, pipe)
        
        cot = improved_summary_data["part1_improvements"]
        revised_summary = improved_summary_data["revised_summary"]
        full_output = improved_summary_data["full_output"]
        # print("--full output_start--"*5)
        # print(f"Iteration {iteration+1} Full Improvement Output:\n {full_output}")
        # print("--full output_end--"*5)
        # print("--revised_summary_start--"*5)
        # print(f"Iteration {iteration+1} Revised Summary:\n {revised_summary}")
        # print("--revised_summary_end--"*5)
        
        if not revised_summary or revised_summary.strip() == "":
            print(f"⚠️ Warning: Iteration {iteration+1} produced empty summary. Using previous summary.")
            revised_summary = previous_summary
            print("--full output_start--"*5)
            print(f"Iteration {iteration+1} Full Improvement Output:\n {full_output}")
            print("--full output_end--"*5)
            break
        
        
        reference = summary_data["Actual"] #

        # scores = calculate_rouge_scores(summary_data["Actual"], revised_summary)
        current_score = evaluator(reference, revised_summary)
        # print(f"Iteration {iteration+1} {stopping_criteria} Score: {current_score}")

        if previous_score is not None and current_score <= previous_score:
            # print(f"Iteration {iteration+1}: No improvement in {stopping_criteria}. Stopping.")
            # current_score = previous_score
            revised_summary = previous_summary
            break
        previous_score = current_score
        previous_summary = revised_summary
        # summary_data["predicted"] = revised_summary
        total_iterations += 1
        
        # print all important info  
        

    
    return first_feedback, feedback, revised_summary, cot, total_iterations
