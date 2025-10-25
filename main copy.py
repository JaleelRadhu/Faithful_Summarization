import yaml
import pandas as pd
from models.model_loader import load_model
from pipeline.refinement_loop import refinement_loop
from utils.generation import ChatGenerator
from utils.evaluator import get_rouge_l_score
# from utils.io_utils import save_results
import argparse
import os
import json
import pandas as pd
import torch
from tqdm import tqdm

from dotenv import load_dotenv
#load hf token from environment variable
load_dotenv()


def load_prompts(evaluator_path, improver_path):
    """Load evaluator and improver prompts from files"""
    with open(evaluator_path, 'r') as f:
        evaluator_prompt = f.read()
    with open(improver_path, 'r') as f:
        improver_prompt = f.read()
    return evaluator_prompt, improver_prompt

def load_perspective_definitions(path):
    """Load perspective definitions from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)
def format_answers(answers_list):
    """Converts a list of answers into a single formatted string."""
    formatted = []
    for i, ans in enumerate(answers_list):
        formatted.append(f"Answer {i+1}: {ans}")
    return "\n".join(formatted)

def format_spans(spans_list):
    """Converts a list of spans into a single formatted string."""
    formatted = []      
    for i, span in enumerate(spans_list):
        formatted.append(f"Span {i+1}: {span}")
    return "\n".join(formatted)
def save_results(results, output_path): # helper
    
    """Helper function to save results to a CSV file."""
    
    # Define the column names to match the order in results
    columns = ["Perspective", "Feedback", "Revised Summary", "Chain of Thought Improvements", "Actual"]
    
    # Convert results into a DataFrame
    df = pd.DataFrame(results, columns=columns)
    
    # Save as CSV
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Results saved to {output_path}")


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', required=True, help='Path to input CSV file')
    parser.add_argument('--config', default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--without_cot', default=False, help="to not include cot")
    parser.add_argument('--without_evaluator', default=False, help="to not include evaluator")
    parser.add_argument('--is_evaluator_and_improver_same', default=True, help="to keep the evaluator and improver model same or not")
    parser.add_argument('--stopping_criteria', default='rouge-l-f', help='the stopping criterial for the refinement loop')
    parser.add_argument('--max_iterations', default=5, help="max number iterations for the refinement loop")

    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model and tokenizer
    model, tokenizer = load_model(
        config['model']['name'],
        quantization=config['model'].get('quantization', 'None')
    )
    
    system_prompt = "You are an expert summary faithfulness evaluator and  improver. Follow the instructions exactly and provide structured outputs."
    pipe = ChatGenerator(model, tokenizer, system_prompt=system_prompt)

    print(f"Device set to use {model.device}")
    
    # Load prompts and perspective definitions
    evaluator_prompt, improver_prompt = load_prompts(
        config['evaluator_prompt_path'],
        config['improver_prompt_path']
    )
    perspective_defs = load_perspective_definitions(config['perspective_definitions_path'])
    
    
    # Load input data
    # df = pd.read_csv(args.input_data)
    df = pd.read_json(args.input_data)
    print("you have changed code here, make sure it's running correctly. Now you are loading from json")
    results = []
    
    # Determine output path
    input_filename = os.path.splitext(os.path.basename(args.input_data))[0]
    output_dir = config["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Build a descriptive suffix for the output file based on arguments
    suffix_parts = ["improved"]
    if args.without_cot:
        suffix_parts.append("no-cot")
    if args.without_evaluator:
        suffix_parts.append("no-eval")
    suffix_parts.append(f"stop-{args.stopping_criteria}")

    # Add model names to the suffix
    if args.is_evaluator_and_improver_same:
        # Use the main model name for both if they are the same
        model_name_cleaned = config['model']['name'].replace('/', '_').replace('-', '_')
        suffix_parts.append(f"imp_eval_model-{model_name_cleaned}")
    else:
        # Use separate names for improver and evaluator models
        improver_model_name_cleaned = config['improver_model']['name'].replace('/', '_').replace('-', '_')
        evaluator_model_name_cleaned = config['evaluator_model']['name'].replace('/', '_').replace('-', '_')
        suffix_parts.append(f"impr-{improver_model_name_cleaned}")
        suffix_parts.append(f"eval-{evaluator_model_name_cleaned}")


    
    output_filename = f"{input_filename}_{'_'.join(suffix_parts)}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    # use tqdm for progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        print(f"\n{'='*80}")
        print(f"Processing sample {idx + 1}/{len(df)}")
        print(f"{'='*80}\n")
        
        answer_str = format_answers(row['answers'])
        perspective_spans_str = format_spans(row['input_spans'])
        # Prepare summary data
        summary_data = {
            "Perspective": row['perspective'],          
            "Perspective_Def": perspective_defs.get(row['perspective'], ""),
            "Question": row['question'],
            "Answers": answer_str,
            "Input_spans": perspective_spans_str,
            "Predicted": row['Predicted'],
            "Actual": row['Actual'],
            "Given Summary": row['Predicted']  # Initial summary
        }
        # starting rouge l f score
        starting_score = get_rouge_l_score(row['Actual'], row['Predicted'])
        print(f"Initial ROUGE Scores: {starting_score}")
        
        if args.without_cot and (not args.without_evaluator):
            # TO DO
            raise NotImplementedError
        elif args.without_evaluator and (not args.without_cot):
            # TO DO
            raise NotImplementedError
        elif (not args.without_cot) and (not args.without_evaluator): #default state
            if args.is_evaluator_and_improver_same:
                first_feedback, feedback, revised_summary, cot, total_iterations = refinement_loop(
                    summary_data,
                    evaluator_prompt,
                    improver_prompt,
                    pipe,
                    max_iterations=config['iteration']['max_iterations']
                )
            else:
                # TO DO
                raise NotImplementedError
        
        else: 
            print("Invalid arguments for --without_cot and --without_evaluator")
        
        results.append({
            "original_summary": row['Predicted'],
            "revised_summary": revised_summary,
            "Gold Summary": row['Actual'],
            "first_feedback": first_feedback,
            "feedback": feedback,
            "cot": cot,
            "perspective": row['perspective'],
            "question": row['question'],
            "starting score": starting_score,
            "final score": get_rouge_l_score(row['Actual'], revised_summary) , 
            "total_iterations": total_iterations
            })
        
        print(f"\n✅ Completed sample {idx + 1}")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ All samples processed. Results saved to: {output_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
