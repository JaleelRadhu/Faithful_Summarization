import json
import os

def enrich_json_data(target_file_path, source_file_path):
    """
    Enriches data points in a target JSON file by adding 'answers' and 'input_spans'
    from a source JSON file based on matching 'question' and 'perspective'.

    Args:
        target_file_path (str): The path to the JSON file to be enriched (e.g., "plasma_test.json").
        source_file_path (str): The path to the JSON file containing the enrichment data
                                (e.g., "/home/abdullahm/jaleel/Faithfullness_Improver/data/complete_base_mistralai_Mistral-7B-Instruct-v0.3.json").

    Returns:
        str: The path to the newly created enriched JSON file, or None if an error occurred.
    """
    print(f"Loading target data from: '{target_file_path}'")
    try:
        with open(target_file_path, 'r', encoding='utf-8') as f:
            target_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Target file not found at '{target_file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{target_file_path}'. Please ensure it's a valid JSON array.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading '{target_file_path}': {e}")
        return None

    print(f"Loading source data from: '{source_file_path}'")
    try:
        with open(source_file_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Source file not found at '{source_file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{source_file_path}'. Please ensure it's a valid JSON array.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading '{source_file_path}': {e}")
        return None

    # Create a lookup dictionary for efficient matching: (question, perspective) -> entry
    source_lookup = {}
    for idx, entry in enumerate(source_data):
        question = entry.get("question")
        perspective = entry.get("perspective")
        
        if question is not None and perspective is not None:
            key = (question, perspective)
            # If duplicate keys exist in source_data, the last one encountered will be used.
            # For this task, we assume question+perspective is a unique identifier.
            source_lookup[key] = entry
        else:
            print(f"Warning: Skipping source entry at index {idx} due to missing 'question' or 'perspective'.")

    enriched_count = 0
    matching_source_data = []
    for idx, target_entry in enumerate(target_data):
        question = target_entry.get("question")
        perspective = target_entry.get("perspective")


        if question is not None and perspective is not None:
            key = (question, perspective)
            if key in source_lookup:
                source_match = source_lookup[key]
                matching_source_data.append(source_match)
                # Add/update 'answers' and 'input_spans' fields
                if "answers" in source_match:
                    target_entry["answers"] = source_match["answers"]
                if "input_spans" in source_match:
                    target_entry["input_spans"] = source_match["input_spans"]
                if "Actual" in source_match:
                    target_entry["Actual"] = source_match["Actual"]
                
                enriched_count += 1
            else:
                pass
                # print(f"Info: No matching entry found in source for target entry {idx} (Question: '{question}', Perspective: '{perspective}'). 'answers' and 'input_spans' will not be added/updated for this entry.")
        else:
            print(f"Warning: Skipping target entry at index {idx} due to missing 'question' or 'perspective'.")

    # Derive the output file name based on the instruction:
    # "saves those data points with name test_input_file_name with complete at start of input file name removed."
    # This is interpreted as: if the original input file name *starts with* "complete_",
    # then the output file name should be that name *without* "complete_".
    # Otherwise, if the input file name *does not* start with "complete_",
    # then the output file name should be `complete_` + original_name.
    
    base_name = os.path.basename(target_file_path)
    dir_name = os.path.dirname(target_file_path)
    
    output_base_name = base_name
    if base_name.startswith("complete_"):
        output_base_name = base_name[len("complete_"):]
    else:
        output_base_name = f"complete_{base_name}"

    source_base_name = os.path.basename(source_file_path)
    source_dir_name = os.path.dirname(source_file_path)

    final_output_file_path = os.path.join(dir_name, output_base_name)
    
    # Save the file with only the matching data points from the source
    filtered_source_filename = f"filtered_{source_base_name}"
    filtered_source_filepath = os.path.join(dir_name, filtered_source_filename)
    print(f"Attempting to save filtered source data to: '{filtered_source_filepath}'")
    try:
        with open(filtered_source_filepath, 'w', encoding='utf-8') as f:
            json.dump(matching_source_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(matching_source_data)} matching source data points to '{filtered_source_filepath}'")
    except IOError as e:
        print(f"Error: Could not write to filtered source file '{filtered_source_filepath}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing to '{filtered_source_filepath}': {e}")

    print(f"Attempting to save enriched data to: '{final_output_file_path}'")
    try:
        with open(final_output_file_path, 'w', encoding='utf-8') as f:
            json.dump(target_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully enriched {enriched_count} data points.")
        print(f"Enriched data saved to '{final_output_file_path}'")
        return final_output_file_path
    except IOError as e:
        print(f"Error: Could not write to output file '{final_output_file_path}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while writing to '{final_output_file_path}': {e}")
        return None

    


# --- Main execution block ---
if __name__ == "__main__":
    # Define the path to the plasma_test.json file.
    # IMPORTANT: Replace "plasma_test.json" with the actual absolute path to your plasma_test.json file
    # if it's not in the same directory as this script.
    target_json_file = "/home/abdullahm/jaleel/Faithfullness_Improver/data/plasma_test.json" 
    
    # Define the path to the source JSON file containing the enrichment data.
    # This path is provided in your context.
    # source_json_file = "/home/abdullahm/jaleel/Faithfullness_Improver/data/complete_base_mistralai_Mistral-7B-Instruct-v0.3.json"
    source_list = [ "/home/abdullahm/jaleel/Faithfullness_Improver/data/complete_base_google_gemma-2-9b.json", 
                   "/home/abdullahm/jaleel/Faithfullness_Improver/data/complete_base_google_gemma-3-12b-it.json", 
                   "/home/abdullahm/jaleel/Faithfullness_Improver/data/complete_base_meta-llama_Llama-3.1-8B-Instruct.json", 
                   "/home/abdullahm/jaleel/Faithfullness_Improver/data/complete_base_meta-llama_Llama-3.1-8B.json", 
                   "/home/abdullahm/jaleel/Faithfullness_Improver/data/complete_base_microsoft_Phi-3-mini-4k-instruct.json", 
                   "/home/abdullahm/jaleel/Faithfullness_Improver/data/complete_base_Qwen_Qwen2.5-7B-Instruct.json", 
                   "/home/abdullahm/jaleel/Faithfullness_Improver/data/complete_base_mistralai_Mistral-7B-Instruct-v0.3.json"
        
    ]# all the files in data directory except plasma_test.json
    # Run the enrichment process
    
    for source_json_file in source_list:
        enriched_file = enrich_json_data(target_json_file, source_json_file)
        
        if enriched_file:
            print(f"\nScript finished. Check '{enriched_file}' for the enriched data.")
        else:
            print("\nScript finished with errors.")
