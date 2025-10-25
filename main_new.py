import yaml
import pandas as pd
# from models.model_loader import load_model # No longer needed
from pipeline.refinement_loop import refinement_loop
from utils.generation_new import ChatGenerator
from utils.evaluator import get_evaluator
# from utils.io_utils import save_results
import argparse
import os
import multiprocessing 
import json
import time
from tqdm import tqdm
import requests
import sys
from urllib.parse import urljoin
from urllib.parse import urlparse
from functools import partial

from dotenv import load_dotenv
#load hf token from environment variable
load_dotenv()
import ijson


def check_server_connection(api_url_str):
    """
    Pings the vLLM server to ensure it is reachable before starting the main process.
    """
    max_retries = 5
    retry_delay_seconds = 15 # Increased delay to give the server more time to load

    print(f"\n{'='*80}")
    print("🩺 Verifying connection to vLLM server...")

    for attempt in range(1, max_retries + 1):
        try:
            # Construct the base URL from the full completions URL
            base_url = urljoin(api_url_str, '.')
            # Correctly construct the health check URL from the server's root.
            parsed_url = urlparse(api_url_str)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            # The /v1/models endpoint is a lightweight way to check server health
            health_check_url = f"{base_url}/v1/models"
            
            print(f"   Attempt {attempt}/{max_retries}: Pinging {health_check_url}...")
            response = requests.get(health_check_url, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            print(f"✅ Connection successful. Server is responsive at {base_url}")
            print(f"{'='*80}\n")
            return # Exit the function if connection is successful

        except requests.exceptions.RequestException as e:
            print(f"   ❌ Connection failed on attempt {attempt}: {e}")
            if attempt < max_retries:
                print(f"   Retrying in {retry_delay_seconds} seconds...")
                time.sleep(retry_delay_seconds)

    # This part is only reached if all retries fail
    print(f"\n❌ FATAL: Failed to connect to the vLLM server at {api_url_str} after {max_retries} attempts.")
    print("   Please ensure the server is running, the URL in your .env file is correct,")
    print("   and there are no firewall issues.")
    sys.exit(1) # Exit the script if all retries fail


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

# Global object to be initialized by each worker
worker_globals = {}

def init_worker(config, evaluator_prompt_path, improver_prompt_path, perspective_definitions_path):
    """
    Initializes resources for each worker process.
    This prevents reloading models/prompts for every single item.
    """
    # Load prompts and perspective definitions
    evaluator_prompt, improver_prompt = load_prompts(
        evaluator_prompt_path,
        improver_prompt_path
    )
    perspective_defs = load_perspective_definitions(perspective_definitions_path)
    
    system_prompt = "You are an expert summary faithfulness evaluator and improver. Follow the instructions exactly and provide structured outputs."
    
    # Each worker gets its own ChatGenerator client instance
    pipe = ChatGenerator(
        model_name=config['model']['name'],
        system_prompt=system_prompt
    )

    worker_globals['pipe'] = pipe
    worker_globals['evaluator_prompt'] = evaluator_prompt
    worker_globals['improver_prompt'] = improver_prompt
    worker_globals['perspective_defs'] = perspective_defs
    worker_globals['config'] = config

def process_row(row_tuple, stopping_criteria):
    """
    The main processing function for a single row, to be executed by a worker process.
    """
    idx, row = row_tuple
    # We now access prompts, pipe, etc., from the initialized worker_globals
    pipe = worker_globals['pipe']
    evaluator_prompt = worker_globals['evaluator_prompt']
    improver_prompt = worker_globals['improver_prompt']
    perspective_defs = worker_globals['perspective_defs']
    config = worker_globals['config']

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
    evaluator = get_evaluator(stopping_criteria)
    starting_score = evaluator(row['Actual'], row['Predicted'])
    

    # This logic can be expanded as you implement the other branches
    first_feedback, feedback, revised_summary, cot, total_iterations = refinement_loop(
        summary_data,
        evaluator_prompt,
        improver_prompt,
        pipe,
        max_iterations=config['iteration']['max_iterations'],
        stopping_criteria=stopping_criteria
    )

    result = {
        "original_summary": row['Predicted'],
        "revised_summary": revised_summary,
        "Gold Summary": row['Actual'],
        "first_feedback": first_feedback,
        "feedback": feedback,
        "cot": cot,
        "perspective": row['perspective'],
        "question": row['question'],
        "starting score": starting_score,
        "final score": evaluator(row['Actual'], revised_summary),
        "total_iterations": total_iterations
    }
    return result

def stream_json_data(filepath):
    """
    Generator function to stream data from a large JSON file.
    This avoids loading the entire file into memory.
    It yields data in the same format as df.iterrows() for compatibility.
    """
    with open(filepath, 'rb') as f:
        # We assume the JSON is an array of objects at the top level.
        # ijson.items will yield each object one by one.
        for i, record in enumerate(ijson.items(f, 'item')):
            yield i, record

def count_json_items(filepath):
    """Helper function to count items in a JSON array for tqdm progress bar."""
    with open(filepath, 'rb') as f:
        return sum(1 for _ in ijson.items(f, 'item'))

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', required=True, help='Path to input CSV file')
    parser.add_argument('--config', default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--without_cot', default=False, help="to not include cot")
    parser.add_argument('--without_evaluator', default=False, help="to not include evaluator")
    parser.add_argument('--is_evaluator_and_improver_same', default=True, help="to keep the evaluator and improver model same or not")
    parser.add_argument('--stopping_criteria', default='rouge-l-f', help='the stopping criterial for the refinement loop')
    parser.add_argument('--max_iterations', default=5, help="max number iterations for the refinement loop")
    parser.add_argument('--num_workers', type=int, default=200, help='Number of parallel worker processes to use.')

    args = parser.parse_args()
    print(os.cpu_count())
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.is_evaluator_and_improver_same:
        print(f"Starting the Main Script...\n\t-Evaluator and Improver Model: {config['model']['name']}\n\t")
    else:
        print(f"Starting the Main Script...\n\t-Evaluator Model: {config['evaluator_model']['name']}\n\t-Improver Model: {config['improver_model']['name']}\n\t")
    
    static_text = "Make sure the model printed above is same as the one you are using as vllm. Starting in 10 seconds... "
    print(static_text, end="", flush=True)
    animation = "|/-\\"
    for i in range(100): 
        time.sleep(0.1)
        print(f"\r{static_text}{animation[i % len(animation)]}", end="", flush=True)
    print(f"\r{static_text}✅") # End with a checkmark

    # --- Verify Server Connection ---
    # Get the API URL from environment or default, then check the connection.
    api_url = os.environ.get("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
    # check_server_connection(api_url)

    # Load input data
    print("🚀 Preparing to stream data from input file...")
    # Get the total number of items for the progress bar without loading the file
    total_tasks = count_json_items(args.input_data)
    print(f"Found {total_tasks} items to process.")
    
    # Create a generator to stream tasks one by one
    tasks = stream_json_data(args.input_data)
    
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
    
    # Prepare for multiprocessing
    print(f"Using {args.num_workers} worker processes.")

    # --- Fully Streaming Pipeline ---
    # Create a partial function to pass the stopping_criteria from args
    # to the process_row worker function.
    process_row_with_criteria = partial(process_row, stopping_criteria=args.stopping_criteria)

    # This block now streams both input and output. It reads one item from the
    # input file, sends it to a worker, and writes the result to the output
    # file as soon as it's ready.
    with open(output_path, 'w') as f_out, multiprocessing.Pool(
            processes=args.num_workers,
            initializer=init_worker,
            initargs=(
                config,
                config['evaluator_prompt_path'],
                config['improver_prompt_path'],
                config['perspective_definitions_path']
            )
    ) as pool:
        f_out.write('[\n') # Start of the JSON array
        first_result = True
        # pool.imap_unordered is perfect for streaming as it takes an iterable (our generator)
        for result in tqdm(pool.imap_unordered(process_row_with_criteria, tasks), total=total_tasks, desc="Processing samples"):
            if result:
                if not first_result:
                    f_out.write(',\n') # Add a comma before each result except the first
                json.dump(result, f_out, indent=2)
                first_result = False
        f_out.write('\n]\n') # End of the JSON array
    
    print(f"\n{'='*80}")
    print(f"✅ All samples processed. Results saved to: {output_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
