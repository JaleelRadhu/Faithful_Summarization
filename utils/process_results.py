import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def process_and_plot_results(json_path: str, output_image_path: str = 'results_comparison.png'):
    """
    Reads experiment results from a JSON file, prints them as a table,
    and generates a comparative bar plot.

    Args:
        json_path (str): The path to the input JSON file.
        output_image_path (str): The path to save the output plot image.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{json_path}'.")
        return

    # --- Helper to flatten nested JSON ---
    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                # Check if it's a score dictionary
                if 'starting_score' in v and 'final_score' in v:
                    items.append((f"{new_key}_start", v['starting_score']))
                    items.append((f"{new_key}_final", v['final_score']))
                else:
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
        return dict(items)


    # --- 1. Process data into a DataFrame ---
    processed_data = []
    for experiment_name, metrics in data.items():
        # Clean up the long experiment names for readability
        clean_name = experiment_name.replace(
            '_improved_stop-rouge-l-f_imp_eval_model-Qwen_Qwen2.5_7B_Instruct', ''
        ).replace(
            'filtered_complete_base_', ''
        )
        
        row = {'Experiment': clean_name}
        row.update(flatten_dict(metrics))
        processed_data.append(row)

    df = pd.DataFrame(processed_data)
    df = df.set_index('Experiment')

    # --- 2. Display the data as a table ---
    print("--- Experiment Results Table ---")
    # Use to_string() to ensure the full table is printed without truncation
    print(df.to_string())
    print("\n" + "="*80 + "\n")

    # --- 3. Prepare data for plotting (long format) ---
    df_long = df.stack().reset_index()
    df_long.columns = ['Experiment', 'Metric_Type', 'Score']
    df_long[['Metric', 'Score_Type']] = df_long['Metric_Type'].str.rsplit('_', n=1, expand=True)
    df_long['Score_Type'] = df_long['Score_Type'].map({'start': 'Starting Score', 'final': 'Final Score'})
    
    # --- 4. Plot the data ---
    print(f"Generating plot and saving to '{output_image_path}'...")
    
    sns.set_theme(style="whitegrid")
    
    # Determine the number of metrics to create subplots
    metrics_to_plot = df_long['Metric'].unique()
    num_metrics = len(metrics_to_plot)
    
    fig, axes = plt.subplots(num_metrics, 1, figsize=(14, 6 * num_metrics), sharex=True)
    if num_metrics == 1:
        axes = [axes] # Make it iterable if there's only one subplot

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        metric_data = df_long[df_long['Metric'] == metric]
        
        sns.barplot(data=metric_data, x='Experiment', y='Score', hue='Score_Type', ax=ax, palette='viridis')
        
        ax.set_title(f'Comparison for {metric.upper()}', fontsize=16, weight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('') # Remove x-label for all but the last subplot
        ax.legend(title='Score Type')
        
        # Hide x-tick labels for all but the last plot
        if i < num_metrics - 1:
            ax.tick_params(axis='x', labelbottom=False)

        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points',
                        fontsize=9)

    # Configure the bottom-most subplot's x-axis
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    axes[-1].set_xlabel('Experiment', fontsize=12) # Set x-label only on the last plot
    plt.suptitle('Model Performance Improvement', fontsize=20, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout to make room for suptitle
    
    try:
        plt.savefig(output_image_path, bbox_inches='tight', dpi=300)
        print("Plot saved successfully.")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == '__main__':
    # The path to your JSON file as provided in the context
    json_file_path = '/home/abdullahm/jaleel/Faithfullness_Improver/results/results_full_model.json'
    process_and_plot_results(json_file_path)
