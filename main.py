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
    df = pd.read_csv(args.input_data)
    
    results = []
    
    # Determine output path
    input_filename = os.path.splitext(os.path.basename(args.input_data))[0]
    output_dir = config["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{input_filename}_improved.json")
    
    # use tqdm for progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        print(f"\n{'='*80}")
        print(f"Processing sample {idx + 1}/{len(df)}")
        print(f"{'='*80}\n")
        
        # Prepare summary data
        summary_data = {
            "Perspective": row['Perspective'],          
            "Perspective_Def": perspective_defs.get(row['Perspective'], ""),
            "Question": row['question'],
            "Answers": row['Answers'],
            "Input_spans": row['Input_spans'],
            "Predicted": row['Predicted'],
            "Actual": row['Actual'],
            "Given Summary": row['Predicted']  # Initial summary
        }
        # starting rouge l f score
        starting_score = get_rouge_l_score(row['Actual'], row['Predicted'])
        print(f"Initial ROUGE Scores: {starting_score}")
        # Run refinement loop
        feedback, revised_summary, cot = refinement_loop(
            summary_data,
            evaluator_prompt,
            improver_prompt,
            pipe,
            max_iterations=config['iteration']['max_iterations']
        )
        
        results.append({
            "original_summary": row['Predicted'],
            "revised_summary": revised_summary,
            "Gold Summary": row['Actual'],
            "feedback": feedback,
            "cot": cot,
            "perspective": row['Perspective'],
            "question": row['question'],
            "starting score": starting_score,
            "final score": get_rouge_l_score(row['Actual'], revised_summary) 
            })
        
        print(f"\n✅ Completed sample {idx + 1}")
    
    # for idx, row in df.iterrows():
    #     print(f"\n{'='*80}")
    #     print(f"Processing sample {idx + 1}/{len(df)}")
    #     print(f"{'='*80}\n")
        
    #     # Prepare summary data
    #     summary_data = {
    #         "Perspective": row['Perspective'],
    #         "Perspective_Def": perspective_defs.get(row['Perspective'], ""),
    #         "Question": row['question'],
    #         "Answers": row['Answers'],
    #         "Input_spans": row['Input_spans'],
    #         "Predicted": row['Predicted'],
    #         "Actual": row['Actual'],
    #         "Given Summary": row['Predicted']  # Initial summary
    #     }
    #     # starting rouge l f score
    #     starting_score = get_rouge_l_score(row['Actual'], row['Predicted'])
    #     print(f"Initial ROUGE Scores: {starting_score}")
    #     # Run refinement loop
    #     feedback, revised_summary, cot = refinement_loop(
    #         summary_data,
    #         evaluator_prompt,
    #         improver_prompt,
    #         pipe,
    #         max_iterations=config['iteration']['max_iterations']
    #     )
        
        # Store results
        # results.append({
        #     "original_summary": row['Predicted'],
        #     "revised_summary": revised_summary,
        #     "Gold Summary": row['Actual'],
        #     "feedback": feedback,
        #     "cot": cot,
        #     "perspective": row['Perspective'],
        #     "question": row['question'],
        #     "starting score": starting_score,
        #     "final score": get_rouge_l_score(row['Actual'], revised_summary) 
        #     })
        
        # print(f"\n✅ Completed sample {idx + 1}")
        
        
    # Save results
    # output_path = config['data']['output_dir'] + 'refined_summaries.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ All samples processed. Results saved to: {output_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
