import os
import json
import re
from collections import defaultdict

def process_model_scores(directory_path):
    """
    Processes JSON files in a directory to aggregate scores by base model name,
    and then by score category and type (original/revised).
    Also calculates the minimum score for each list.
    """
    # defaultdict for: base_model -> category -> score_type -> [scores]
    model_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Regex to extract the base model name from the keys inside the JSON file.
    # e.g., from 'filtered_complete_base_google_gemma-2-9b_improved...' it extracts 'google_gemma-2-9b'
    model_name_regex = re.compile(r'filtered_complete_(.*?)_improved')

    print(f"Searching for score files in: {directory_path}\n")

    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return

    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    for key, scores_dict in data.items():
                        match = model_name_regex.search(key)
                        if match:
                            base_model_name = match.group(1)
                            for category, values in scores_dict.items():
                                if 'original' in values:
                                    model_scores[base_model_name][category]['original'].append(values['original'])
                                if 'revised' in values:
                                    model_scores[base_model_name][category]['revised'].append(values['revised'])
                        else:
                            print(f"Warning: Could not extract base model name from key '{key}' in file '{filename}'.")

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from '{file_path}'. Skipping.")
            except Exception as e:
                print(f"An error occurred while processing '{file_path}': {e}")

    if not model_scores:
        print("No models found or processed.")
        return

    print("--- Aggregated Model Scores ---")
    # Convert defaultdict to dict for cleaner processing
    final_scores = {model: {cat: dict(types) for cat, types in categories.items()} for model, categories in model_scores.items()}
    
    # Add minimum scores to the data structure
    for model_name, categories in final_scores.items():
        for category, types in categories.items():
            if 'original' in types and types['original']:
                types['original_min'] = min(types['original'])
            if 'revised' in types and types['revised']:
                types['revised_min'] = min(types['revised'])

    # Print the results
    for model_name, categories in final_scores.items():
        print(f"\n--- Base Model: {model_name} ---")
        print(json.dumps(categories, indent=4))
        
    print("\n--- End of Report ---")


if __name__ == '__main__':
    # To use this script, replace '.' with the path to your directory.
    # For example: '/home/abdullahm/jaleel/Faithfullness_Improver/results_llm_eval_Qwen/'
    # Using '.' will scan the current directory where the script is run.
    target_directory = '/home/abdullahm/jaleel/Faithfullness_Improver/results_llm_eval_Qwen/'
    process_model_scores(target_directory)