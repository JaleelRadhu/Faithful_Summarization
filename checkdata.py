import json
import os
from collections import defaultdict

def clean_and_validate_json(input_file_path: str, output_file_path: str):
    """
    Scans a JSON file to find and remove objects that are missing required keys
    or have empty values for specific keys. Writes the cleaned data to a new file.

    Args:
        input_file_path (str): The path to the JSON file to check.
        output_file_path (str): The path to save the cleaned JSON file.
    """
    # Based on your tracebacks, these keys are essential for processing.
    required_keys = [
        'question',
        'perspective',
        'Actual',
        'Predicted',
        'answers',
        'input_spans'
    ]
    
    # Keys that should not have empty string values, which caused the ROUGE error.
    non_empty_keys = [
        'answers',
        'input_spans'
    ]

    print(f"🔍 Validating and cleaning file: {input_file_path}\n")

    if not os.path.exists(input_file_path):
        print(f"❌ Error: File not found at '{input_file_path}'")
        return

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON from '{input_file_path}': {e}")
        return
    except Exception as e:
        print(f"❌ An error occurred while reading the file: {e}")
        return

    if not isinstance(data, list):
        print("❌ Error: JSON file does not contain a list of objects.")
        return

    total_items = len(data)
    cleaned_data = []
    removed_items_count = 0
    issue_summary = defaultdict(int)

    print(f"Found {total_items} items. Scanning for items to remove...\n")

    for i, item in enumerate(data):
        current_item_issues = []
        
        # 1. Check for missing keys
        for key in required_keys:
            if key not in item:
                current_item_issues.append(f"Missing key: '{key}'")
                issue_summary[f"missing_{key}"] += 1

        # 2. Check for empty or None values in critical fields
        for key in non_empty_keys:
            if key in item and not item.get(key): # Handles None, "", [], {}
                current_item_issues.append(f"Empty value for key: '{key}'")
                issue_summary[f"empty_{key}"] += 1
        
        if not current_item_issues:
            # If there are no issues, add the item to our clean list
            cleaned_data.append(item)
        else:
            # If there are issues, report and skip it
            removed_items_count += 1
            print(f"--- 🗑️  Removing item at index {i} due to issues: ---")
            for issue in current_item_issues:
                print(f"  - {issue}")
            print("-" * 40)

    print("\n--- ✅ Check Complete ---")
    if removed_items_count == 0:
        print("🎉 No problematic items found. The original file is already clean.")
        # Save a copy anyway to have a consistent output file
        output_file_path = input_file_path
    else:
        print(f"Removed {removed_items_count} problematic items out of {total_items} total items.\n")
        print("Summary of issues:")
        for issue_type, count in issue_summary.items():
            print(f"  - {issue_type}: {count} occurrences")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✨ Successfully saved {len(cleaned_data)} valid items to:\n{output_file_path}")

if __name__ == "__main__":
    # Path to the JSON file you want to check
    input_path = '/home/abdullahm/jaleel/Faithfullness_Improver/data/filtered_complete_plasma_test.json'
    
    # Define the output path for the cleaned file
    output_path = '/home/abdullahm/jaleel/Faithfullness_Improver/data/cleaned_filtered_complete_plasma_test.json'
    
    clean_and_validate_json(input_path, output_path)
