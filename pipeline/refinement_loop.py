# from evaluation.rouge_eval import calculate_rouge_scores
from pipeline.feedback_generator import generate_feedback
from pipeline.summary_improver import revise_summary
from utils.evaluator import get_evaluator
import time
import yaml


def refinement_loop(summary_data, evaluator_prompt, improver_prompt, evaluator_pipe, improver_pipe, max_iterations, stopping_criteria, tolerance=0.01):
    previous_summary = summary_data["Predicted"]
    previous_score = None
    # print("++"*100)
    # print(improver_prompt)
    # print("++"*100)
    evaluator = get_evaluator(stopping_criteria)
    
    total_iterations = 0
    start_time = time.time()
    for iteration in range(max_iterations):
        
        summary_data["Given Summary"] = previous_summary
        if evaluator_prompt == "no evaluation":
            feedback = ""
        else:
            feedback = generate_feedback(summary_data, evaluator_prompt, evaluator_pipe)
        if iteration == 0:
            first_feedback = feedback
        # write the above in a single line
        
        # cot, revised_summary = revise_summary(summary_data, feedback, improver_prompt, pipe)
        # print("="*100)
        # print("--feedback_start--"*5)
        # print("="*100)
        # print(f"Iteration {iteration+1} Feedback:\n {feedback}")
        # print("="*100)
        # print("--feedback_end--"*5)
        # print("="*100)
        #generate revised summary using improver prompt
        improved_summary_data = revise_summary(summary_data, feedback, improver_prompt, improver_pipe)
        
        cot = improved_summary_data["part1_improvements"]
        revised_summary = improved_summary_data["revised_summary"]
        full_output = improved_summary_data["full_output"]
        # print("="*100)
        # print("--full output_start--"*5)
        # print("="*100)
        # print(f"Iteration {iteration+1} Full Improvement Output:\n {full_output}")
        # print("="*100)
        # print("--full output_end--"*5)
        # print("="*100)
        # print("="*100)
        # print("--revised_summary_start--"*5)
        # print("="*100)
        # print(f"Iteration {iteration+1} Revised Summary:\n {revised_summary}")
        # print("="*100)
        # print("--revised_summary_end--"*5)
        # print("="*100)

        
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
    end_time = time.time()
    elapsed_time = end_time - start_time
    # print in seconds)
    print(f"Time taken for refinement loop: {elapsed_time:.2f} seconds")
    print(f"Total iterations: {total_iterations}")

    
    return first_feedback, feedback, revised_summary, cot, total_iterations
