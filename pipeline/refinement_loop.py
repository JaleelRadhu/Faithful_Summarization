# from evaluation.rouge_eval import calculate_rouge_scores
from pipeline.feedback_generator import generate_feedback
from pipeline.summary_improver import revise_summary
from utils.evaluator import get_evaluator

import yaml


def refinement_loop(summary_data, evaluator_prompt, improver_prompt, pipe, max_iterations=5, tolerance=0.01):
    previous_summary = summary_data["Predicted"]
    previous_score = None

    # load yaml 
    config = yaml.safe_load(open("/home/abdullahm/jaleel/Faithfullness_Improver/configs/default.yaml"))
    max_iterations = config["iteration"]["max_iterations"]
    # get stopping criteria from config file
    stopping_criteria = config["iteration"]["stopping_criterion"]
    evaluator = get_evaluator(stopping_criteria)
    
    
    for iteration in range(max_iterations):
        
        summary_data["Given Summary"] = previous_summary
        
        #generate feedback using evaluator prompt
        feedback = generate_feedback(summary_data, evaluator_prompt, pipe)
        # cot, revised_summary = revise_summary(summary_data, feedback, improver_prompt, pipe)
        print("--feedback_start--"*5)
        print(f"Iteration {iteration+1} Feedback:\n {feedback}")
        print("--feedback_end--"*5)
        
        #generate revised summary using improver prompt
        improved_summary_data = revise_summary(summary_data, feedback, improver_prompt, pipe)
        
        cot = improved_summary_data["part1_improvements"]
        revised_summary = improved_summary_data["revised_summary"]
        full_output = improved_summary_data["full_output"]
        print("--full output_start--"*5)
        print(f"Iteration {iteration+1} Full Improvement Output:\n {full_output}")
        print("--full output_end--"*5)
        print("--revised_summary_start--"*5)
        print(f"Iteration {iteration+1} Revised Summary:\n {revised_summary}")
        print("--revised_summary_end--"*5)
        
        if not revised_summary or revised_summary.strip() == "":
            print(f"⚠️ Warning: Iteration {iteration+1} produced empty summary. Using previous summary.")
            revised_summary = previous_summary
            break
        
        
        reference = summary_data["Actual"] #THIS IS PROBLEMATIC, YOU ARE USING THE | GOLD SUMMARY | TO EVALUATE THE PREDICTION !!!

        # scores = calculate_rouge_scores(summary_data["Actual"], revised_summary)
        current_score = evaluator(reference, revised_summary)
        print(f"Iteration {iteration+1} {stopping_criteria} Score: {current_score}")

        if previous_score is not None and current_score <= previous_score:
            print(f"Iteration {iteration+1}: No improvement in {stopping_criteria}. Stopping.")
            break
        previous_score = current_score
        previous_summary = revised_summary
        # summary_data["predicted"] = revised_summary
        
        # print all important info  
        

    
    return feedback, revised_summary, cot
