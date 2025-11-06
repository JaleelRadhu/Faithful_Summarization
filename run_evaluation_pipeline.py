
import json
import os
import argparse
import re
from utils.generation_new import ChatGenerator
from pipeline.feedback_generator import generate_feedback

def extract_details_from_original_summary(original_summary):
    """
    Extracts details like answers and input spans from the original_summary string.
    This is a helper function to deal with the messy string format.
    """
    answers = []
    input_spans = []

    # Regex to find answers
    answer_matches = re.findall(r"Answer \d+: (.*?)(?=\nAnswer \d+:|$)", original_summary, re.DOTALL)
    if answer_matches:
        answers = [ans.strip() for ans in answer_matches]

    # Regex to find input spans
    input_span_matches = re.findall(r"Provided Input Spans from the answers for this perspective:(.*?)(?=\n\n)", original_summary, re.DOTALL)
    if input_span_matches:
        input_spans = [span.strip() for span in input_span_matches]

    return answers, input_spans

def main():
    parser = argparse.ArgumentParser(description="Evaluate summaries in a given directory using the feedback generator.")
    parser.add_argument("input_dir", type=str, help="The directory containing the JSON files to evaluate.")
    parser.add_argument("--model_name", type=str, default="default_model", help="The name of the model to use for evaluation.")
    parser.add_argument("--output_dir", type=str, required=True, help="The directory to save the evaluation results.")
    args = parser.parse_args()

    # Load perspective definitions
    with open("prompts/perspective_defn.json", "r") as f:
        perspective_definitions = json.load(f)

    # Load evaluator prompt
    with open("prompts/evaluator_prompt.txt", "r") as f:
        evaluator_prompt_template = f.read()

    # Initialize ChatGenerator
    system_prompt = "You are an expert evaluator. Your task is to evaluate the given summary based on the provided criteria."
    chat_generator = ChatGenerator(model_name=args.model_name, system_prompt=system_prompt)

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at {args.input_dir}")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    files_to_evaluate = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]

    for filename in files_to_evaluate:
        filepath = os.path.join(args.input_dir, filename)
        with open(filepath, "r") as f:
            data = json.load(f)

        results = []
        for item in data:
            original_summary = item.get("original_summary", "")
            revised_summary = item.get("revised_summary", "")
            perspective = item.get("perspective", "")
            question = item.get("question", "")

            answers, input_spans = extract_details_from_original_summary(original_summary)
            perspective_def = perspective_definitions.get(perspective, "")
            
            cleaned_original_summary = original_summary.split('\n\n**Question:**')[0]

            summary_data_original = {
                "Perspective": perspective,
                "Perspective_Def": perspective_def,
                "Input_spans": "\n".join(input_spans),
                "Given Summary": cleaned_original_summary,
                "Question": question,
                "Answers": "\n".join(answers)
            }

            summary_data_revised = {
                "Perspective": perspective,
                "Perspective_Def": perspective_def,
                "Input_spans": "\n".join(input_spans),
                "Given Summary": revised_summary,
                "Question": question,
                "Answers": "\n".join(answers)
            }

            original_feedback = generate_feedback(summary_data_original, evaluator_prompt_template, chat_generator)
            revised_feedback = generate_feedback(summary_data_revised, evaluator_prompt_template, chat_generator)

            results.append({
                "question": question,
                "original_summary": cleaned_original_summary,
                "revised_summary": revised_summary,
                "original_summary_evaluation": original_feedback,
                "revised_summary_evaluation": revised_feedback
            })

        output_filename = os.path.join(args.output_dir, f"evaluation_{filename}")
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation for {filename} to {output_filename}")

if __name__ == "__main__":
    main()
