import json
import os
import argparse
import multiprocessing
import re
from utils.generation_new import ChatGenerator
from tqdm import tqdm
from functools import partial
import pandas as pd
from dotenv import load_dotenv

from main_new import format_answers, format_spans

# Load environment variables from .env file
load_dotenv()

def parse_scores_from_feedback(feedback_text: str):
    """
    Parses the scores for each of the four metrics from the evaluator's feedback.
    Returns a dictionary with the scores.
    """
    scores = {}
    
    # Regex to find "Score for METRIC: [score]"
    # It's flexible to handle variations in whitespace or casing.
    score_patterns = {
        'Extraneous Information': r"\*\*Score for Extraneous Information:\*\*\s*(\d)",
        'Contradiction': r"\*\*Score for Contradiction:\*\*\s*(\d)",
        'Perspective Misalignment': r"\*\*Score for Perspective Misalignment:\*\*\s*(\d)",
        'Redundancy': r"\*\*Score for Redundancy:\*\*\s*(\d)"
    }

    for metric, pattern in score_patterns.items():
        match = re.search(pattern, feedback_text, re.IGNORECASE)
        if match:
            try:
                scores[metric] = int(match.group(1))
            except (ValueError, IndexError):
                print("couldn't parse score")
                scores[metric] = None # Could not parse score
        else:
            scores[metric] = None # Metric not found
            
    return scores

def evaluate_item(item, original_items, perspective_definitions, evaluator_prompt_template, chat_generator, args, loaded_original_data, filename_pre):
    """
    Evaluates a single item (summary) and returns the scores.
    This function encapsulates the evaluation logic for parallel processing.
    """
    # Skip items that had errors during the main run
    if "error" in item:
        return None

    # --- Prepare data for the prompt ---
    perspective = item.get("perspective", "")
    question = item.get("question", "")
    original_summary = item.get("original_summary", "")
    revised_summary = item.get("revised_summary", "")
    
    if not original_summary or not revised_summary:
        print(f"⚠️ Warning: Skipping item due to empty original or revised summary for question '{question}' and perspective '{perspective}'.")
        return None

    # Find the corresponding item in the original data
    found_original_item = None
    for original_item in original_items:
        if original_item.get("question") == question and original_item.get("perspective") == perspective:
            found_original_item = original_item
            break
    
    if not found_original_item:
        print(f"⚠️ Warning: Could not find matching original item for question '{question}' and perspective '{perspective}' in original data. Skipping evaluation for this item.")
        return None

    # Extract answers and spans from the found original item
    answers_list = found_original_item.get("answers", [])
    spans_list = found_original_item.get("input_spans", [])

    answers_str = format_answers(answers_list)
    spans_str = format_spans(spans_list)

    perspective_def = perspective_definitions.get(perspective, "N/A")

    # --- Evaluate Original Summary ---
    original_summary_data = {
        "Perspective": perspective,
        "Perspective_Def": perspective_def,
        "Input_spans": spans_str,
        "Given Summary": original_summary,
        "Question": question,
        "Answers": answers_str
    }
    prompt_original = evaluator_prompt_template.format(**original_summary_data)
    original_evaluation_text = chat_generator(prompt_original)[0]['generated_text']
    # print("="*100)
    # print("--original eval start")
    # print(("="*100))
    # print(original_evaluation_text)
    # print("="*100)
    # print("--original eval end")
    # print(("="*100))
    original_scores = parse_scores_from_feedback(original_evaluation_text)
    
    # --- Evaluate Revised Summary ---
    revised_summary_data = {
        "Perspective": perspective,
        "Perspective_Def": perspective_def,
        "Input_spans": spans_str,
        "Given Summary": revised_summary,
        "Question": question,
        "Answers": answers_str
    }
        
    prompt_revised = evaluator_prompt_template.format(**revised_summary_data)
    revised_evaluation_text = chat_generator(prompt_revised)[0]['generated_text']
    # print("="*100)
    # print("--revised eval start")
    # print(("="*100))
    # print(revised_evaluation_text)
    # print("="*100)
    # print("--revised eval end")
    # print(("="*100))
    revised_scores = parse_scores_from_feedback(revised_evaluation_text)

    # --- Store scores for this item ---
    item_scores = {}
    for metric in original_scores:
        item_scores[f'original_{metric.replace(" ", "_")}'] = original_scores[metric]
        item_scores[f'revised_{metric.replace(" ", "_")}'] = revised_scores[metric]

    return item_scores






def main():
    parser = argparse.ArgumentParser(description="Evaluate summaries in a directory, extract scores, and calculate averages.")
    parser.add_argument("--input_dir", type=str, help="The directory containing the JSON output files to evaluate.")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="The name of the model to use for evaluation.")
    parser.add_argument("--data_dir", type=str, default="/home/abdullahm/jaleel/Faithfullness_Improver/data", help="Directory containing original data files to find answers/spans.")
    parser.add_argument("--result_dir", type=str, default="/home/abdullahm/jaleel/Faithfullness_Improver/results", help="Directory to save the evaluation results.")
    args = parser.parse_args()


    # --- Load necessary assets ---
    try:
        with open("prompts/perspective_defn.json", "r") as f:
            perspective_definitions = json.load(f)
        with open("prompts/evaluator_prompt.txt", "r") as f:
            evaluator_prompt_template = f.read()
    except FileNotFoundError as e:
        print(f"❌ Error: Could not load a required file: {e}. Make sure you are running from the project root.")
        return

    # --- Initialize ChatGenerator for evaluation ---
    system_prompt = "You are an expert evaluator. Your task is to evaluate the given summary based on the provided criteria. Follow the instructions EXACTLY."
    # Using VLLM_API_URL from .env for the evaluator model
    if args.model_name == "mistralai/Mistral-7B-Instruct-v0.3":
        vllm_api_url = os.environ.get("VLLM_API_URL_MISTRAL")
    elif args.model_name =="Qwen/Qwen2.5-7B-Instruct":
        vllm_api_url = os.environ.get("VLLM_API_URL_QWEN")
    else: 
        vllm_api_url = os.environ.get("VLLM_API_URL")
    if not vllm_api_url:
        print("⚠️ Warning: VLLM_API_URL not found in .env file. Using default HuggingFace endpoint.")

    chat_generator = ChatGenerator(
        model_name=args.model_name,
        system_prompt=system_prompt,
        vllm_api_url=vllm_api_url, 
        request_timeout=60,
        max_retries=1000
    )

    if not os.path.isdir(args.input_dir):
        print(f"❌ Error: Input directory not found at '{args.input_dir}'")
        return

    files_to_evaluate = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]
    print(f"Found {len(files_to_evaluate)} JSON files to evaluate in '{args.input_dir}'.\n")

    # Store loaded original data to avoid reloading the same file multiple times
    loaded_original_data = {}

    # --- Process each file ---  
    for filename in files_to_evaluate:
        print(f"\n{'='*80}\nProcessing file: {filename}\n{'='*80}")
        filepath = os.path.join(args.input_dir, filename)
        filename_pre = filename.split("_improved")[0] + ".json"
        original_data_filepath = os.path.join(args.data_dir, filename_pre)
        
        # Load original data if not already loaded
        if original_data_filepath not in loaded_original_data:
            try:
                with open(original_data_filepath, "r") as f:
                    loaded_original_data[original_data_filepath] = json.load(f)
                print(f"✅ Loaded original data from: {original_data_filepath}")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"❌ Error: Could not read or parse original data file {original_data_filepath}: {e}")
                continue # Skip this processed file if original data cannot be loaded

        original_items = loaded_original_data.get(original_data_filepath, [])
        
        try:
            with open(filepath, "r") as f:
                results_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Could not read or parse {filename}: {e}")
            continue
        
        # Extract the directory name
        input_dir_name = os.path.basename(args.input_dir.rstrip('/'))
        
        all_scores = []

        # --- Prepare the evaluation function with necessary context ---
        evaluation_func = partial(
            evaluate_item,
            original_items=original_items,
            perspective_definitions=perspective_definitions,
            evaluator_prompt_template=evaluator_prompt_template,
            chat_generator=chat_generator,
            args=args,
            loaded_original_data=loaded_original_data,
            filename_pre=filename_pre
        )

        # --- Use multiprocessing to evaluate items in parallel ---
        # num_processes = multiprocessing.cpu_count() * 2  # Example: Use double the number of CPU cores
        num_processes = 100
        with multiprocessing.Pool(processes=num_processes) as pool:
            item_scores_list = list(tqdm(
                pool.imap(evaluation_func, results_data),
                total=len(results_data),
                desc=f"Evaluating items in {filename}"
            ))

        all_scores = [item_scores for item_scores in item_scores_list if item_scores is not None]

        # --- Calculate and Display Average Scores for the file ---
        if not all_scores:
            print("No valid items found to evaluate in this file.")
            continue

        df_scores = pd.DataFrame(all_scores)
        
        print(f"\n📊 Average Scores for: {filename}")
        print("-" * 50)
        
        metrics = ['Extraneous Information', 'Contradiction', 'Perspective Misalignment', 'Redundancy']
        for metric in metrics:
            metric_key = metric.replace(" ", "_")
            avg_original = df_scores[f'original_{metric_key}'].mean()
            avg_revised = df_scores[f'revised_{metric_key}'].mean()
            print(f"{metric:<25} | Start: {avg_original:.2f} -> Final: {avg_revised:.2f}")
        
        print("-" * 50)



    # --- After processing all files, save aggregated results to a single JSON file ---
    all_results = {}
    for filename in files_to_evaluate:
        filepath = os.path.join(args.input_dir, filename)
        try:
            df_scores = pd.DataFrame(all_scores)
            average_scores = calculate_average_scores(df_scores)
            all_results[filename] = average_scores
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Could not read or parse {filename}: {e}")
            continue

    # Create the directory if it doesn't exist
    os.makedirs(args.result_dir, exist_ok=True)

    output_filename = f"results_full_model_{input_dir_name.lower()}.json" # Naming json file
    output_filepath = os.path.join(args.result_dir, output_filename)
    try:
        with open(output_filepath, "w") as outfile:
            json.dump(all_results, outfile, indent=4)
    except Exception as e:
        print(f"❌ Error: Could not save evaluation results to {output_filepath}: {e}")


def calculate_average_scores(df_scores):
    """Calculates and returns average scores for each metric."""
    average_scores = {}
    metrics = ['Extraneous Information', 'Contradiction', 'Perspective Misalignment', 'Redundancy']
    for metric in metrics:
        metric_key = metric.replace(" ", "_")
        avg_original = df_scores[f'original_{metric_key}'].mean()
        avg_revised = df_scores[f'revised_{metric_key}'].mean()
        average_scores[metric] = {
            "original": avg_original,
            "revised": avg_revised
        }
    return average_scores
if __name__ == "__main__":
    main()
