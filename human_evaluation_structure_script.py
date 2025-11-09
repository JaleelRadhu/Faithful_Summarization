import argparse
import json
import random
import csv
import os

def load_json_file(filepath):
    """Loads a JSON file and returns its content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file {filepath} is not a valid JSON file.")
        return None

def prepare_evaluation_data(num_samples, source_file_name, eval_models):
    """
    Generates JSON and CSV files for human evaluation.
    """
    # 1. Load the source data file
    source_data_path = os.path.join('data', source_file_name)
    source_data = load_json_file(source_data_path)
    if source_data is None:
        return

    # 2. Select N random data points from the source file
    if num_samples > len(source_data):
        print(f"Warning: Requested {num_samples} samples, but only {len(source_data)} are available. Using all available data points.")
        num_samples = len(source_data)
        
    random_samples = random.sample(source_data, num_samples)

    # 3. Build the structured JSON and mapping
    output_json_data = {}
    mapping_data = {}

    # This will be the prefix for the summary filenames
    source_file_base_name = source_file_name.replace('.json', '')

    summary_file_lookups = {}
    summary_dir = os.path.join('outputs', 'all_qwen') # All summary files are in this directory
    print("EVAL MODELS: ", eval_models)
    for model in eval_models:
        # Dynamically find the summary file instead of hardcoding the name
        summary_filename = None
        try:
            # Find the file that corresponds to the source model.
            # e.g., source 'filtered_complete_base_google_gemma-2-9b.json' -> model 'gemma-2-9b'
            # The summary file will contain the model name.
            matching_files = [f for f in os.listdir(summary_dir) if model in f and f.endswith('.json')]

            if len(matching_files) == 1:
                summary_filename = matching_files[0]
                print(f"Found summary file for '{model}': {summary_filename}")
            elif len(matching_files) > 1:
                print(f"Warning: Found multiple possible summary files for model '{model}'. Skipping.")
                continue
            else:
                print(f"Warning: No summary file found for model '{model}' in {summary_dir}.")
                continue
        except FileNotFoundError:
            print(f"Warning: Directory not found for model '{model}': {summary_dir}")
            continue
        
        if summary_filename:
            summary_filepath = os.path.join(summary_dir, summary_filename)
            summary_data = load_json_file(summary_filepath)
            # Create a lookup key of (question, perspective) for each item
            summary_file_lookups[model] = {
                (item['question'], item['perspective']): item for item in summary_data
            }
        else:
            print(f"Warning: Could not load or find summary file for model '{model}'.")

    if not summary_file_lookups:
        print("Error: No summary files could be loaded. Exiting.")
        return

    for i, sample in enumerate(random_samples, 1):
        # Initialize the structure for the current data point
        output_json_data[str(i)] = {
            "question": sample.get("question", ""),
            "answers": sample.get("answers", []),
            "Input Spans": sample.get("input_spans", []),
            "Perspective": sample.get("perspective", [])
        }
        
        summaries_to_shuffle = []
        current_mapping = {}
        
        # Create the composite key for the current sample
        lookup_key = (sample.get("question"), sample.get("perspective"))

        # For each evaluation model, find the corresponding summaries
        for model in eval_models:
            base_summary = "N/A"
            revised_summary = "N/A"

            if model in summary_file_lookups and lookup_key in summary_file_lookups[model]:
                matched_summary_item = summary_file_lookups[model][lookup_key]
                # Assuming the keys are 'original_summary' and 'revised_summary' in the summary files
                base_summary = matched_summary_item.get('original_summary', 'N/A')
                revised_summary = matched_summary_item.get('revised_summary', 'N/A')
            else:
                print(f"Warning: No match found for model '{model}' with question/perspective: {lookup_key}")
            
            base_col_name = f"{model}_base_summary"
            revised_col_name = f"{model}_revised_summary"

            output_json_data[str(i)][base_col_name] = base_summary
            output_json_data[str(i)][revised_col_name] = revised_summary
            
            summaries_to_shuffle.extend([base_col_name, revised_col_name])

        # 4. Create a random order for the summaries
        num_summaries = len(summaries_to_shuffle)
        order_list = list(range(num_summaries))
        random.shuffle(order_list)
        output_json_data[str(i)]["order list"] = order_list
        
        # 5. Create the mapping for this data point
        for shuffled_idx, original_idx in enumerate(order_list):
            original_col_name = summaries_to_shuffle[original_idx]
            shuffled_col_name = f"summary_{shuffled_idx + 1}" # CSV column name
            current_mapping[shuffled_col_name] = original_col_name
        
        mapping_data[str(i)] = current_mapping

    # 5. Save the structured JSON and the mapping file
    Base_DIR = "/home/abdullahm/jaleel/Faithfullness_Improver"
    with open(os.path.join(Base_DIR, 'human_eval.json'), 'w', encoding='utf-8') as f:
        json.dump(output_json_data, f, indent=4)
    print("Successfully created human_eval.json")

    with open(os.path.join(Base_DIR, 'mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=4)
    print("Successfully created mapping.json to track summary origins.")

    # 7. Create the CSV for human evaluators
    if not output_json_data:
        print("No data to write to CSV.")
        return

    first_item_key = next(iter(output_json_data))
    num_summaries_in_csv = len(output_json_data[first_item_key].get("order list", []))

    # Define CSV headers
    headers = ['id', 'question', 'answers', 'Input Spans', 'Perspective']
    score_categories = [
        'Fluency',
        'Coherence',
        'Extraneous',
        'Contradiction',
        'Perspective Misalignment',
        'Redundancy'
    ]

    # Interleave summary and score headers for a more intuitive layout
    for j in range(num_summaries_in_csv):
        headers.append(f'summary_{j+1}')
        for category in score_categories:
            headers.append(f'summary_{j+1}_{category}')

    with open(os.path.join(Base_DIR,'human_eval.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for item_id, data in output_json_data.items():
            # Format answers for readability
            answers_list = data.get('answers', [])
            formatted_answers = "\n\n".join([f"Answer {i+1}: {ans}" for i, ans in enumerate(answers_list)])

            # Format input spans for readability
            spans_list = data.get('Input Spans', [])
            formatted_spans = "\n\n".join([f"span{i+1}: {span}" for i, span in enumerate(spans_list)])

            row = {
                'id': item_id,
                'question': data['question'],
                'answers': formatted_answers,
                'Input Spans': formatted_spans,
                # Perspective is a string, no need for json.dumps
                'Perspective': data.get('Perspective', '')
            }
            
            # Get all summaries for the current item
            all_item_summaries = []
            for model in eval_models:
                all_item_summaries.append(data[f"{model}_base_summary"])
                all_item_summaries.append(data[f"{model}_revised_summary"])

            # Add summaries in the randomized order
            order_list = data['order list']
            for k, original_idx in enumerate(order_list):
                row[f'summary_{k+1}'] = all_item_summaries[original_idx]
            
            # Add empty score columns
            for k in range(1, num_summaries_in_csv + 1):
                for category in score_categories:
                    row[f'summary_{k}_{category}'] = ''
                
            writer.writerow(row)
            
    print("Successfully created human_eval.csv for evaluation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepare human evaluation data by sampling from a source data file."
    )
    parser.add_argument(
        "-n", "--num_samples", type=int, required=True, help="The number of random data points to select (N)."
    )
    parser.add_argument(
        "-s", "--source_file_name", type=str, required=True, help="The name of the source JSON file inside the 'data/' directory."
    )
    parser.add_argument(
        "-m", "--eval_models", nargs='+', required=True, help="A list of evaluation model names, separated by spaces (e.g., qwen mistral)."
    )

    args = parser.parse_args()
    prepare_evaluation_data(args.num_samples, args.source_file_name, args.eval_models)
