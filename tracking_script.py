import pandas as pd
import os

RESULTS_FILE = 'human_eval_results.csv'

def track_evaluation_progress():
    """
    Reads the results file and prints a summary of each evaluator's progress.
    """
    # 1. Check if the results file exists
    if not os.path.exists(RESULTS_FILE):
        print(f"The results file '{RESULTS_FILE}' was not found.")
        print("No evaluations have been submitted yet.")
        return

    # 2. Load the results CSV into a pandas DataFrame
    try:
        df_results = pd.read_csv(RESULTS_FILE)
    except pd.errors.EmptyDataError:
        print(f"The results file '{RESULTS_FILE}' is empty.")
        print("No evaluations have been submitted yet.")
        return

    # 3. Check if the DataFrame is empty
    if df_results.empty:
        print(f"The results file '{RESULTS_FILE}' has no data.")
        print("No evaluations have been submitted yet.")
        return

    print("--- Evaluation Progress Report ---")

    # 4. Group by evaluator name
    evaluator_progress = df_results.groupby('evaluator_name')

    # 5. Iterate through each evaluator and print their progress
    for name, group in evaluator_progress:
        evaluated_ids = sorted(group['sample_id'].unique().tolist())
        count = len(evaluated_ids)
        
        print(f"\nEvaluator: {name}")
        print(f"  - Total Samples Evaluated: {count}")
        print(f"  - Evaluated Sample IDs: {evaluated_ids}")

    print("\n--- End of Report ---")

if __name__ == '__main__':
    track_evaluation_progress()