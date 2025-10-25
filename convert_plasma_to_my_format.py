import csv
import json
import os

def convert_csv_to_json(csv_file_path, json_file_path):
    """
    Reads a CSV file and converts specific columns into a JSON file.

    This script processes a CSV with headers and extracts the 'question',
    'Perspective', 'Actual', and 'Predicted' columns from each row.
    It then writes this data into a JSON file as a list of objects.

    Args:
        csv_file_path (str): The path to the input CSV file.
        json_file_path (str): The path for the output JSON file.
    """
    json_data_list = []
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(json_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
            # Use DictReader to treat each row as a dictionary
            csv_reader = csv.DictReader(csv_file)
            # Convert the iterator to a list to get the count and iterate over it
            rows = list(csv_reader)
            print(f"Number of rows in CSV: {len(rows)}")
            
            print(f"Reading from '{csv_file_path}'...")
            for row in rows:
                print("hi")
                # Create a new dictionary for each row with the desired format
                # The user requested "perspective" in lowercase for the JSON key.
                record = {
                    "question": row.get("question", ""),
                    "perspective": row.get("Perspective", ""),
                    "Actual": row.get("Actual", ""),
                    "Predicted": row.get("Predicted", "")
                }
                json_data_list.append(record)
            print(len(json_data_list))
        with open(json_file_path, mode='w', encoding='utf-8') as json_file:
            # Write the list of dictionaries to the JSON file
            # Using indent=4 for pretty-printing and readability
            json.dump(json_data_list, json_file, indent=4)
            
        print(f"Successfully converted data to '{json_file_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
    except KeyError as e:
        print(f"Error: A required column is missing from the CSV file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define the input and output file paths based on the provided context
    input_csv_path = "/home/abdullahm/jaleel/Faithfullness_Improver/data/old_gauri's/plasma_test.csv"
    
    # Create the output JSON file in the same directory with a new extension
    output_json_path = os.path.splitext(input_csv_path)[0] + ".json"
    
    convert_csv_to_json(input_csv_path, output_json_path)
