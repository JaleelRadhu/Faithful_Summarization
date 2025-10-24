import json
import csv
import argparse
import sys

def process_json_to_csv(input_file, output_file):
    """
    Reads the input JSON file and converts it to the specified CSV format.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Input file '{input_file}' is not valid JSON. {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)

    # These are the headers you requested
    headers = ["Perspective", "Answers", "Input_spans", "question", "Actual"]
    
    processed_rows = []

    # Loop through each main object in the JSON list (e.g., each 'uri')
    for item in data:
        question = item.get("question")
        
        # The 'answers' list is the same for all perspectives in this item.
        # We serialize it as a JSON string to store it neatly in one CSV cell.
        answers_list = item.get("answers", [])
        answers_str = json.dumps(answers_list)
        
        perspectives_spans = item.get("labelled_answer_spans", {})
        perspectives_summaries = item.get("labelled_summaries", {})

        # Loop through each perspective found in the item (e.g., "INFORMATION")
        for perspective_name in perspectives_spans.keys():
            
            # 1. Get Perspective
            # perspective_name is "INFORMATION"

            # 2. Get Answers (already done)
            # answers_str is "[\"LOL cute question...\", ...]"

            # 3. Get Input_spans
            # This is the list of span dicts for that perspective.
            # We also serialize this as a JSON string.
            spans_list = perspectives_spans.get(perspective_name, [])
            spans_list = ",".join([f"[{i["txt"]}]" for i in spans_list])
            spans_str = json.dumps(spans_list)
            
            # 4. Get question (already done)
            # question is "Do men produce more gas..."

            # 5. Get Actual (Summary)
            # The summary key matches the perspective name + "_SUMMARY"
            summary_key = f"{perspective_name}_SUMMARY"
            actual_summary = perspectives_summaries.get(summary_key)

            # Only add the row if we found a matching summary
            if actual_summary:
                processed_rows.append([
                    perspective_name,
                    answers_str,
                    spans_str,
                    question,
                    actual_summary
                ])

    # Now, write all processed rows to the CSV file
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)  # Write the header row
            writer.writerows(processed_rows) # Write all the data rows
        
        print(f"Successfully converted {len(processed_rows)} items from '{input_file}' to '{output_file}'.")

    except IOError:
        print(f"Error: Could not write to output file '{output_file}'. Check permissions.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Convert perspective-aware JSON data to a flat CSV.")
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        dest="input_file",
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        dest="output_file",
        help="Path for the resulting CSV file."
    )
    
    args = parser.parse_args()
    process_json_to_csv(args.input_file, args.output_file)

if __name__ == "__main__":
    main()