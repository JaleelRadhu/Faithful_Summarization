import yaml
import pandas as pd
from models.model_loader import load_model
from pipeline.refinement_loop import run_trial
from utils.io_utils import save_results
import argparse
import os
import json
import pandas as pd
import torch
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
    
    parser = argparse.ArgumentParser(description="Run summary improvement pipeline")
    parser.add_argument("--input_data", type=str, required=True, help="Path to input CSV file")
    args = parser.parse_args()

    # Load config
    config = yaml.safe_load(open("config/default.yaml"))

    # Load model pipeline
    pipe = load_model(config["model"]["name"], quantization=config["model"]["quantization"])

    # Load data
    df = pd.read_csv(args.input_data)    
    
    evaluator_prompt = open(config["evaluator_prompt_path"]).read()
    improver_prompt = open(config["improver_prompt_path"]).read()
    
    
    # load json file of perspective definitions as p_def_dic
    p_def_path = config["perspective_definitions_path"]
    with open(p_def_path, 'r') as f:
        p_def_dic = json.load(f)
    
    # Determine output path
    input_filename = os.path.splitext(os.path.basename(args.input_data))[0]
    output_dir = config["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{input_filename}_improved.csv")
    
    results = []
    for _, row in df.iterrows():
        perspective_def = p_def_dic[row["Perspective"]]
        summary_data = {
            "Question": row["question"],
            "Answers": row["Answers"],
            "Perspective": row["Perspective"],
            "Predicted": row["Predicted"],
            "Input_spans": row["Input_spans"],
            "Actual": row["Actual"],
            "Perspective_Def": perspective_def
        }
        
        try: 
            feedback, revised, cot = run_trial(summary_data, evaluator_prompt, improver_prompt, pipe)
        except torch.cuda.OutOfMemoryError:
            print("CUDA Out of Memory. Skipping this row.")
            with open(f"{input_filename}_oom_errors.txt", "a") as f:
                f.write(f"Row with Actual: {summary_data['Actual']}\n")
            continue
        except Exception as e:
            print(f"Other Error processing")
            with open(f"{input_filename}_other_errors.txt", "a") as f:
                f.write(f"Row with Actual: {summary_data['Actual']}, Error: {e}\n")
                
        results.append([row["Perspective"], feedback, revised, cot, row["Actual"]])

    save_results(results, output_path)

if __name__ == "__main__":
    main()
