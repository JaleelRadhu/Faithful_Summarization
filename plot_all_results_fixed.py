import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class PlotConfig:
    """Configuration for plot styling."""
    figsize: tuple = (22, 10)
    title_fontsize: int = 16
    label_fontsize: int = 14
    tick_fontsize: int = 12
    legend_fontsize: int = 12
    bar_label_fontsize: int = 8
    colormap: str = 'viridis'
    grid_alpha: float = 0.7

def add_bar_labels(ax: plt.Axes, rects, font_size: int = 8):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90,
                    fontsize=font_size)

def plot_evaluation_results(json_file_path: Path, config: PlotConfig, show_labels: bool = False):
    """
    Reads evaluation results from a JSON file and generates a grouped bar chart.

    The chart will display 'original' vs 'revised' scores for different
    evaluation categories, grouped by the model.

    Args:
        json_file_path (Path): The path to the input JSON file.
        config (PlotConfig): The styling configuration for the plot.
        show_labels (bool): If True, adds score labels on top of each bar.
    """
    try:
        data = json.loads(json_file_path.read_text())
    except FileNotFoundError:
        print(f"Error: The file was not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file {json_file_path}")
        return

    # --- Data Extraction and Preparation ---
    model_names_raw = list(data.keys())
    
    def clean_name(name):
        """Cleans the model name from the long filename using regex."""
        # This regex is more robust and captures the model name more reliably.
        match = re.search(r'base_([^_]+(?:/[^/]+)?)_improved', name) or \
                re.search(r'complete_([^_]+(?:/[^/]+)?)_improved', name)
        if match:
            return match.group(1)
        return name.replace('.json', '').split('_improved')[0] # Fallback

    model_names = [clean_name(name) for name in model_names_raw]

    if not model_names:
        print(f"No data to plot in {json_file_path}.")
        return

    categories = list(data[model_names_raw[0]].keys())
    score_types = ["original", "revised"]
    
    scores: Dict[str, Dict[str, List[float]]] = {cat: {stype: [] for stype in score_types} for cat in categories}

    for model_key in model_names_raw:
        for category in categories:
            for score_type in score_types:
                score = data[model_key].get(category, {}).get(score_type, 0)
                scores[category][score_type].append(score)

    # --- Plotting ---
    x = np.arange(len(model_names))  # the label locations for models
    n_categories = len(categories)
    n_score_types = len(score_types)
    
    # Total number of bars per model group
    n_bars_per_group = n_categories * n_score_types
    bar_width = 1.0 / (n_bars_per_group + 2) # +2 for padding
    
    fig, ax = plt.subplots(figsize=config.figsize)

    colors = plt.cm.get_cmap(config.colormap, n_bars_per_group)

    # Iterate through each bar to plot it
    for i, category in enumerate(categories):
        for j, score_type in enumerate(score_types):
            bar_index = i * n_score_types + j
            
            # Calculate the offset for the current bar from the center of the group
            offset = (bar_index - (n_bars_per_group - 1) / 2) * bar_width
            
            current_scores = scores[category][score_type]
            rects = ax.bar(x + offset, current_scores, bar_width,
                           label=f'{category} - {score_type.capitalize()}',
                           color=colors(bar_index))
            
            if show_labels:
                add_bar_labels(ax, rects, font_size=config.bar_label_fontsize)

    # --- Formatting and Labels ---
    ax.set_ylabel('Scores', fontsize=config.label_fontsize)
    ax.set_title(f'Comparison of Original vs. Revised Scores ({json_file_path.name})', fontsize=config.title_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=config.tick_fontsize)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=config.legend_fontsize)
    ax.grid(axis='y', linestyle='--', alpha=config.grid_alpha)
    
    # Adjust y-axis limit if labels are shown to prevent them from being cut off
    if show_labels:
        ax.set_ylim(top=ax.get_ylim()[1] * 1.1)

    fig.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    # --- Save Figure ---
    output_filename = json_file_path.with_suffix('.png').name
    plt.savefig(output_filename)
    print(f"Graph saved to {Path(output_filename).resolve()}")
    plt.close(fig)

def main():
    """Main function to find and process all JSON result files in a directory."""
    results_directory = Path('/home/abdullahm/jaleel/Faithfullness_Improver/results_llm_eval_Qwen')
    json_files = list(results_directory.glob('*.json'))

    if not json_files:
        print(f"No JSON files found in the directory: {results_directory}")
        return

    plot_config = PlotConfig()

    print(f"Found {len(json_files)} JSON file(s) to process.")
    for json_file in json_files:
        print(f"\n--- Processing {json_file.name} ---")
        # Set show_labels=True to display scores on bars
        plot_evaluation_results(json_file, config=plot_config, show_labels=False)
    print("\nAll files processed.")

if __name__ == '__main__':
    main()