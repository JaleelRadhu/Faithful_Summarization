import json
import argparse
import sys
from pathlib import Path

def process_json_to_flat_json(input_file, output_file):
    """
    Reads a nested JSON file and converts it into a flattened JSON format,
    creating a separate entry for each perspective.
    """
    print("hi")
    try:
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        # Create parent directory for output if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with input_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Input file '{input_file}' is not valid JSON. {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}", file=sys.stderr)
        sys.exit(1)

    flattened_data = []

    # Loop through each main object in the JSON list
    for item in data:
        question = item.get("question")
        answers_list = item.get("answers", [])
        
        perspectives_spans = item.get("labelled_answer_spans", {})
        perspectives_summaries = item.get("labelled_summaries", {})

        # Loop through each perspective found in the item (e.g., "INFORMATION", "CAUSE")
        for perspective_name in perspectives_spans.keys():
            
            # Get the list of span texts for the current perspective
            spans_for_perspective = perspectives_spans.get(perspective_name, [])
            input_spans = [span.get("txt") for span in spans_for_perspective if span.get("txt")]

            # Get the corresponding summary for the current perspective
            summary_key = f"{perspective_name}_SUMMARY"
            base_summary = perspectives_summaries.get(summary_key)

            # Only create an entry if a summary exists for that perspective
            if base_summary:
                new_entry = {
                    "question": question,
                    "perspective": perspective_name,
                    "answers": answers_list,
                    "input_spans": input_spans,
                    "Actual": base_summary
                }
                flattened_data.append(new_entry)

    # Write the new flattened data to the output JSON file
    try:
        print("heee")
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(flattened_data, f, indent=4)
        
        print(f"Successfully converted and flattened {len(flattened_data)} perspective entries.")
        print(f"Output saved to '{output_file}'.")

    except IOError as e:
        print(f"Error: Could not write to output file '{output_file}'. {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Convert a nested perspective-aware JSON to a flat JSON format."
    )
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        dest="input_file",
        help="Path to the input JSON file (e.g., PUMA_complete_Data.json)."
    )
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        dest="output_file",
        help="Path for the resulting flattened JSON file."
    )
    
    args = parser.parse_args()
    process_json_to_flat_json(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
