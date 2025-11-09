import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_evaluation_results(json_file_path):
    """
    Reads evaluation results from a JSON file and generates a grouped bar chart.

    The chart will display 'original' vs 'revised' scores for different
    evaluation categories, grouped by the model.

    Args:
        json_file_path (str): The absolute path to the input JSON file.
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file was not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file {json_file_path}")
        return

    # --- Data Extraction and Preparation ---
    model_names_raw = list(data.keys())
    
    # Clean up model names for better readability on the chart's x-axis
    def clean_name(name):
        name = name.replace('_improved_stop-rouge-l-f_imp_eval_model-mistralai_Mistral_7B_Instruct_v0.3.json', '')
        name = name.replace('filtered_complete_base_', '')
        return name

    model_names = [clean_name(name) for name in model_names_raw]

    if not model_names:
        print("No data to plot.")
        return

    # Get categories from the first model (e.g., "Contradiction")
    categories = list(data[model_names_raw[0]].keys())
    
    original_scores = {cat: [] for cat in categories}
    revised_scores = {cat: [] for cat in categories}

    for model_key in model_names_raw:
        for category in categories:
            scores = data[model_key].get(category, {"original": 0, "revised": 0})
            original_scores[category].append(scores.get("original", 0))
            revised_scores[category].append(scores.get("revised", 0))

    # --- Plotting ---
    x = np.arange(len(model_names))  # the label locations
    width = 0.2  # the width of the bars
    n_categories = len(categories)
    
    fig, ax = plt.subplots(figsize=(18, 10))

    # Create color map for categories
    colors = plt.cm.get_cmap('viridis', n_categories * 2)

    for i, category in enumerate(categories):
        # Calculate offset for each group of bars
        offset = (i - n_categories / 2) * (width * 2.2) + width / 2
        
        # Plot original scores
        rects1 = ax.bar(x + offset - width/2, original_scores[category], width, 
                        label=f'{category} - Original', color=colors(i*2))
        
        # Plot revised scores
        rects2 = ax.bar(x + offset + width/2, revised_scores[category], width, 
                        label=f'{category} - Revised', color=colors(i*2 + 1))

    # --- Formatting and Labels ---
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Original vs. Revised Scores by Model and Category')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()

    # --- Save Figure ---
    output_filename = 'model_evaluation_results.png'
    plt.savefig(output_filename)
    print(f"Graph saved to {os.path.abspath(output_filename)}")
    
    # To display the plot directly if in an interactive environment
    # plt.show()

if __name__ == '__main__':
    # The absolute path to your results file.
    # Make sure to change this if your file is located elsewhere.
    results_file = '/home/abdullahm/jaleel/Faithfullness_Improver/results_llm_eval_Qwen/results_full_model_all_mistral.json'
    plot_evaluation_results(results_file)
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def plot_evaluation_results(json_file_path):
    """
    Reads evaluation results from a JSON file and generates a grouped bar chart.

    The chart will display 'original' vs 'revised' scores for different
    evaluation categories, grouped by the model.

    Args:
        json_file_path (str): The absolute path to the input JSON file.
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file was not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file {json_file_path}")
        return

    # --- Data Extraction and Preparation ---
    model_names_raw = list(data.keys())
    
    # Clean up model names for better readability on the chart's x-axis
    def clean_name(name):
        # This part is highly specific to your filename format.
        # It removes common prefixes and suffixes to get a clean model name.
        name = name.replace('_improved_stop-rouge-l-f_imp_eval_model-mistralai_Mistral_7B_Instruct_v0.3.json', '')
        name = name.replace('filtered_complete_base_', '')
        name = name.replace('filtered_complete_', '') # Added another common pattern
        return name

    model_names = [clean_name(name) for name in model_names_raw]

    if not model_names:
        print(f"No data to plot in {json_file_path}.")
        return

    # Get categories from the first model (e.g., "Contradiction")
    categories = list(data[model_names_raw[0]].keys())
    
    original_scores = {cat: [] for cat in categories}
    revised_scores = {cat: [] for cat in categories}

    for model_key in model_names_raw:
        for category in categories:
            scores = data[model_key].get(category, {"original": 0, "revised": 0})
            original_scores[category].append(scores.get("original", 0))
            revised_scores[category].append(scores.get("revised", 0))

    # --- Plotting ---
    x = np.arange(len(model_names))  # the label locations
    width = 0.2  # the width of the bars
    n_categories = len(categories)
    
    # Increased figure width for better readability
    fig, ax = plt.subplots(figsize=(22, 10))

    # Create color map for categories
    colors = plt.cm.get_cmap('viridis', n_categories * 2)

    for i, category in enumerate(categories):
        # Calculate offset for each group of bars
        offset = (i - n_categories / 2) * (width * 2.2) + width / 2
        
        # Plot original scores
        rects1 = ax.bar(x + offset - width/2, original_scores[category], width, 
                        label=f'{category} - Original', color=colors(i*2))
        
        # Plot revised scores
        rects2 = ax.bar(x + offset + width/2, revised_scores[category], width, 
                        label=f'{category} - Revised', color=colors(i*2 + 1))

    # --- Formatting and Labels ---
    ax.set_ylabel('Scores', fontsize=14)
    ax.set_title(f'Comparison of Original vs. Revised Scores ({os.path.basename(json_file_path)})', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()

    # --- Save Figure ---
    # Generate an output filename based on the input JSON file's name
    base_filename = os.path.basename(json_file_path)
    output_filename = os.path.splitext(base_filename)[0] + '.png'
    # Save the plot in the same directory as the script or specify a different one
    plt.savefig(output_filename)
    print(f"Graph saved to {os.path.abspath(output_filename)}")
    
    # Close the plot to free up memory before processing the next file
    plt.close(fig)

def main():
    """
    Main function to find and process all JSON result files in a directory.
    """
    # Directory containing the result files.
    results_directory = '/home/abdullahm/jaleel/Faithfullness_Improver/results_llm_eval_Qwen'
    
    # Use glob to find all files ending with .json in the specified directory
    json_files = glob.glob(os.path.join(results_directory, '*.json'))

    if not json_files:
        print(f"No JSON files found in the directory: {results_directory}")
        return

    print(f"Found {len(json_files)} JSON file(s) to process.")
    for json_file in json_files:
        print(f"\n--- Processing {os.path.basename(json_file)} ---")
        plot_evaluation_results(json_file)
    print("\nAll files processed.")


if __name__ == '__main__':
    main()
