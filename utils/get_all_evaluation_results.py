"""Run this to get all the evaluation metrics i.e general and llm based in the output dir"""

import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml 
import json
import argparse
from utils.process_results import process_and_plot_results

from utils.evaluator import get_rouge_l_score, get_rouge_1_score, get_rouge_2_score, get_meteor_score, get_BERTScore, get_BARTScore, get_llm_metrics, get_llm_metrics_from_a_model

def get_file_metrics(file_path):
    """
    arg is the improved summary json file

    Calculates various evaluation metrics for original and improved summaries
    from a given JSON file.

    The JSON file is expected to contain a list of dictionaries, where each
    dictionary represents an item with at least 'reference', 'original_summary',
    and 'improved_summary' keys.

    Args:
        file_path (str): The path to the JSON file containing summary data.

    Returns:
        dict: A dictionary where keys are metric names (e.g., 'rouge_l', 'llm_metrics').
              Each value is another dictionary containing 'starting_score' and
              'final_score' (averaged across all items).
              For 'llm_metrics', if it returns multiple sub-scores, the value
              will be a dictionary of sub-metric names, each with 'starting_score'
              and 'final_score'.
              Example:
              {
                  "rouge_l": {"starting_score": 0.5, "final_score": 0.6},
                  "llm_metrics": {
                      "faithfulness": {"starting_score": 0.7, "final_score": 0.8},
                      "coherence": {"starting_score": 0.65, "final_score": 0.75}
                  }
              }
    """

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Define metrics with their names for easier aggregation and clarity
    metric_funs_with_names = [
        ("rouge_l", get_rouge_l_score),
        ("rouge_1", get_rouge_1_score),
        ("rouge_2", get_rouge_2_score),
        ("meteor", get_meteor_score),
        # ("bertscore", get_BERTScore),
        # ("bartscore", get_BARTScore),
        # ("llm_metrics", get_llm_metrics) # This one might return multiple sub-scores
    ]

    # This list will store all individual scores for each item, before averaging them out.
    # Example structure for an item's scores:
    # {
    #   "rouge_l": {"starting_score": 0.1, "final_score": 0.2},
    #   "llm_metrics": {"faithfulness": {"starting_score": 0.7, "final_score": 0.8}, "coherence": {"starting_score": 0.6, "final_score": 0.7}}
    # }
    all_item_scores = []

    for item in data:
        reference = item.get("Gold Summary")
        original_summary = item.get("original_summary")
        improved_summary = item.get("revised_summary")
        start_feedback = item.get("first_feedback")
        final_feedback = item.get("feedback")

        if not all([reference, original_summary, improved_summary]):
            print(f"Warning: Skipping item (ID: {item.get('id', 'N/A')}) due to missing 'reference', 'original_summary', or 'improved_summary'.")
            continue

        item_metric_scores = {} # Scores for the current item

        for metric_name, evaluator_func in metric_funs_with_names:
            if metric_name == "llm_metrics":
                # print("correct this llm metrics part implementation")
                # Assuming get_llm_metrics takes (summary_text, source_text)
                # and returns either a single float or a dictionary of sub-scores.
                starting_llm_result = evaluator_func(start_feedback)
                final_llm_result = evaluator_func(final_feedback)

                if isinstance(starting_llm_result, dict) and isinstance(final_llm_result, dict):
                    llm_sub_scores = {}
                    if not starting_llm_result or not final_llm_result:
                        print(f"Warning: Skipping sub-metric calculation for item (ID: {item.get('id', 'N/A')}) due to missing starting or final LLM result.")
                        continue

                    for sub_metric in starting_llm_result.keys():
                        llm_sub_scores[sub_metric] = {
                            "starting_score": starting_llm_result.get(sub_metric).get("score"),
                            "final_score": final_llm_result.get(sub_metric).get("score")
                        }
                    item_metric_scores[metric_name] = llm_sub_scores
                else:
                    # If LLM metrics return a single score (float)
                    try:
                        item_metric_scores[metric_name] = {
                            "starting_score": starting_llm_result,
                            "final_score": final_llm_result
                        }
                    except Exception as e:
                        print(f"Error processing LLM metrics for item (ID: {item.get('id', 'N/A')}): {e}")
                        item_metric_scores[metric_name] = None
            else:
                # For general metrics (ROUGE, METEOR, BERTScore, BARTScore)


                # Assuming evaluator_func takes (candidate_summary, reference_summary)
                starting_score = evaluator_func( reference, original_summary,)
                final_score = evaluator_func( reference, improved_summary)
                item_metric_scores[metric_name] = {
                    "starting_score": starting_score,
                    "final_score": final_score
                }
        all_item_scores.append(item_metric_scores)

    # Aggregate scores across all items by averaging
    aggregated_results = {}

    for metric_name, _ in metric_funs_with_names:
        # Determine if this metric returns sub-scores (like LLM metrics might)
        is_sub_scored_metric = False
        for item_scores in all_item_scores:
            if metric_name in item_scores and isinstance(item_scores[metric_name], dict):
                # Check if the values within the metric's dictionary are themselves dictionaries
                # This indicates sub-scores (e.g., {"faithfulness": {...}, "coherence": {...}})
                if item_scores[metric_name] and all(isinstance(v, dict) for v in item_scores[metric_name].values()):
                    is_sub_scored_metric = True
                    break

        if is_sub_scored_metric:
            print(metric_name + " is a sub-scored metric")
            # Handle metrics that return multiple sub-scores (e.g., LLM faithfulness, coherence)
            sub_metric_names = set()
            for item_scores in all_item_scores:
                if metric_name in item_scores and isinstance(item_scores[metric_name], dict):
                    sub_metric_names.update(item_scores[metric_name].keys())

            aggregated_results[metric_name] = {}
            for sub_metric in sub_metric_names:
                # try:
                starting_scores_list = [
                    item_scores[metric_name][sub_metric]["starting_score"]
                    for item_scores in all_item_scores
                    if metric_name in item_scores and sub_metric in item_scores[metric_name] and
                       item_scores[metric_name][sub_metric]["starting_score"] is not None
                ]
                final_scores_list = [
                    item_scores[metric_name][sub_metric]["final_score"]
                    for item_scores in all_item_scores
                    if metric_name in item_scores and sub_metric in item_scores[metric_name] and
                       item_scores[metric_name][sub_metric]["final_score"] is not None
                ]
                # print("starting scores list: ", starting_scores_list)
                # print("final scores list: ", final_scores_list)
                
                aggregated_results[metric_name][sub_metric] = {
                    "starting_score": sum(starting_scores_list) / len(starting_scores_list) if starting_scores_list else 0.0,
                    "final_score": sum(final_scores_list) / len(final_scores_list) if final_scores_list else 0.0
                }
        else:
            # Handle metrics that return a single score (or LLM metrics returning a single score)
            starting_scores_list = [
                item_scores[metric_name]["starting_score"]
                for item_scores in all_item_scores
                if metric_name in item_scores and item_scores[metric_name]["starting_score"] is not None
            ]
            final_scores_list = [
                item_scores[metric_name]["final_score"]
                for item_scores in all_item_scores
                if metric_name in item_scores and item_scores[metric_name]["final_score"] is not None
            ]
            aggregated_results[metric_name] = {
                "starting_score": sum(starting_scores_list) / len(starting_scores_list) if starting_scores_list else 0.0,
                "final_score": sum(final_scores_list) / len(final_scores_list) if final_scores_list else 0.0
            }

    return aggregated_results


    

def main():
    
    #take config path in argument using argparser
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--output_sub_dir', default="")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = config["data"]["output_dir"] # Directory where all the improved summaries are saved
    if args.output_sub_dir != "":
        output_dir += "/" + args.output_sub_dir # Directory where all the improved summaries are saved
    
    # results_dir = config.get("results_dir", "results") # Directory to save the aggregated results
    results_dir = config["data"]["results_dir"]
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    final_results = {}
    
    # Iterate over all files in the output_dir
    for filename in os.listdir(output_dir):
        if filename.endswith(".json") : # Process only JSON files
            file_path = os.path.join(output_dir, filename)
            print(f"Processing file: {file_path}")

            try:
                # Run the function get_file_metrics for each improved summary file
                metrics = get_file_metrics(file_path)

                # Use the filename (without extension) as the key for final_results.
                # This can be customized further based on desired naming conventions
                # (e.g., model_name_result_<suffix_message/hyperparameter>).
                file_key = os.path.splitext(filename)[0]
                final_results[file_key] = metrics
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                # Optionally, store an error message or skip this file's results

    # Define the output file name for the aggregated results
    # Can be specified in config or use a default
    if args.output_sub_dir != "":
        output_results_filename = "results_full_model_g" + args.output_sub_dir+ ".json"
    else:
        output_results_filename = "results_full_model" +  ".json"
    output_results_path = os.path.join(results_dir, output_results_filename)

    # Dump final_results in result_file_name and save it in the results directory
    with open(output_results_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    image_file_name = "results_comparison" + args.output_sub_dir + ".png"
    image_path = os.path.join(results_dir, image_file_name)
    print(f"All evaluation results saved to: {output_results_path}")
    process_and_plot_results(output_results_path, output_image_path=image_path)

if __name__ == "__main__":
    main()

    
