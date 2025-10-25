import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import os
import sys
from tqdm.auto import tqdm

def load_data(file_path):
    """Loads the input JSON data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_prompt_template(file_path):
    """Loads the prompt template from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def format_answers(answers_list):
    """Converts a list of answers into a single formatted string."""
    formatted = []
    for i, ans in enumerate(answers_list):
        formatted.append(f"Answer {i+1}: {ans}")
    return "\n".join(formatted)

def format_spans(spans_list):
    """Converts a list of spans into a single formatted string."""
    formatted = []      
    for i, span in enumerate(spans_list):
        formatted.append(f"Span {i+1}: {span}")
    return "\n".join(formatted)


def save_results(data, output_path):
    """Saves the completed data to an output JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"\nSuccessfully saved results to {output_path}")
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")

def log_gpu_memory(step_name=""):
    """Logs the current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"--- GPU Memory at '{step_name}' ---")
        print(f"  Allocated: {allocated:.2f} MB")
        print(f"  Reserved:  {reserved:.2f} MB")
        print("------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Generate summaries using a specified LLM.")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--prompt_template_file", type=str, required=True, help="Path to the .txt prompt template.")
    parser.add_argument("--output_prefix", type=str, default="complete_base", help="Prefix for the output JSON file.")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Max tokens for summary.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for GPU inference. Adjust based on VRAM.")
    parser.add_argument("--force", action='store_true', help="Overwrite output file if it exists.")
    args = parser.parse_args()

    # --- Load environment variables for HF_TOKEN ---
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    # --- 0. Check if output file already exists ---
    model_name_safe = args.model_id.replace("/", "_")
    output_filename = f"{args.output_prefix}_{model_name_safe}.json"
    
    if os.path.exists(output_filename) and not args.force:
        print(f"Output file {output_filename} already exists. Skipping. Use --force to overwrite.")
        sys.exit(0) # Exit successfully

    # --- 1. Load Data and Template ---
    print(f"--- Processing model: {args.model_id} ---")
    data = load_data(args.input_file)
    prompt_template = load_prompt_template(args.prompt_template_file)
    if data is None or prompt_template is None:
        print("Data or prompt template loading failed. Exiting.")
        return

    # --- 2. Set up Model and Pipeline ---
    print(f"Loading model: {args.model_id}. This may take a while...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            token=hf_token
        )
        log_gpu_memory("After model load")

    except Exception as e:
        print(f"Error loading model {args.model_id}: {e}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # More robust check for instruction-tuned models
    model_id_lower = args.model_id.lower()
    # Keywords that often indicate an instruction-tuned or chat model
    instruct_keywords = ["instruct", "-it", "chat", "gpt-oss"]
    is_instruct_model = any(keyword in model_id_lower for keyword in instruct_keywords)

    # Special case for gemma-3-12b-it where '-it' is at the end
    if model_id_lower.endswith("-it"):
        is_instruct_model = True
    print(f"Model type detected as: {'Instruct/Chat' if is_instruct_model else 'Base'}")

    def create_prompt_generator(data, prompt_template, model_id_lower, is_instruct_model, tokenizer):
        """A generator that yields formatted prompts one by one."""
        for item in data:
            prompt_content = build_individual_prompt(item, prompt_template)
            
            if is_instruct_model:
                messages = [{"role": "user", "content": prompt_content}]
                yield tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                yield prompt_content

    def build_individual_prompt(item, prompt_template):
        """Builds a single prompt string from a data item."""
        answers_str = format_answers(item['answers'])
        perspective_spans_str = format_spans(item['input_spans'])
        return prompt_template.format(
            question=item['question'],
            answers=answers_str,
            perspective=item['perspective'],
            input_spans=perspective_spans_str
        )


    # --- 3. Set up Prompt Generator and Run Inference ---
    print("Setting up prompt generator...")
    prompt_generator = create_prompt_generator(data, prompt_template, model_id_lower, is_instruct_model, tokenizer)

    print(f"Generating {len(data)} summaries with batch size {args.batch_size}...")
    outputs_generator = pipe(
        prompt_generator,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        batch_size=args.batch_size,
    )

    # --- 4. Collate Results ---
    print("Generating and collating results...")
    results = []
    # By zipping the data and the generator, we process each item as it's generated,
    # preventing GPU memory from accumulating.
    for i, (item, out) in enumerate(tqdm(zip(data, outputs_generator), total=len(data), desc=f"Generating with {args.model_id}")):
        generated_text = out[0]['generated_text']
        
        # To remove the prompt from the output, we must reconstruct it.
        prompt_content = build_individual_prompt(item, prompt_template)
        if is_instruct_model:
            messages = [{"role": "user", "content": prompt_content}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = prompt_content
        predicted_summary = generated_text[len(prompt):].strip()
        
        new_item = item.copy()
        new_item['Predicted'] = predicted_summary
        results.append(new_item)
        
        # Log memory usage periodically inside the loop
        if (i + 1) % 50 == 0:
            log_gpu_memory(f"Inside loop, item {i + 1}")

    log_gpu_memory("After generation loop")

    # --- 5. Save Final Output ---
    save_results(results, output_filename)
    print(f"--- Finished model: {args.model_id} ---")
    # Explicitly clean up to be safe
    del model, tokenizer, pipe
    torch.cuda.empty_cache()
    log_gpu_memory("After cleanup")

    # (When this script exits, Python releases all VRAM, making it clean for the next model)

if __name__ == "__main__":
    main()