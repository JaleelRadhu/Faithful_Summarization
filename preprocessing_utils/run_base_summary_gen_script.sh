#!/usr/bin/env bash

# --- Configuration ---
# Add all the model IDs you want to test here
MODELS_TO_RUN=(
    "google/gemma-3-12b-it"
    # "mistralai/Mistral-7B-Instruct-v0.3"
    # "meta-llama/Llama-3.1-8B"
    # "meta-llama/Llama-3.1-8B-Instruct"
    "microsoft/Phi-3-mini-4k-instruct"
    # "openai/gpt-oss-20b"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "google/gemma-2-9b"
   
    
    
    # Add your other models here
    # "mistralai/Mistral-7B-v0.1" 
)

INPUT_FILE="complete_puma.json"

PROMPT_FILE="base_summary_gen_prompt.txt"
BATCH_SIZE=16 # Adjust this based on your VRAM. 8 is a safe start.

# --- End Configuration ---

# It's good practice to ensure your libraries are up-to-date for new models.
# You can uncomment the following line to automatically upgrade before running.
# pip install --upgrade transformers accelerate torch

# --- Environment Check ---
# Ensure a virtual environment (venv or conda) is active.
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "ERROR: No Python virtual environment is activated."
    echo "Please activate your venv or conda environment first (e.g., 'conda activate your_env' or 'source path/to/venv/bin/activate')."
    exit 1
fi

echo "Starting summary generation for ${#MODELS_TO_RUN[@]} models..."

# Loop over each model
for model_id in "${MODELS_TO_RUN[@]}"; do
    echo "=========================================================="
    echo "Running: $model_id"
    echo "=========================================================="
    
    python base_summary_generator.py \
        --model_id "$model_id" \
        --input_file "$INPUT_FILE" \
        --prompt_template_file "$PROMPT_FILE" \
        --batch_size $BATCH_SIZE
        # Add --force here if you want to re-generate files every time
        # --force

    echo "Finished: $model_id"
done

echo "=========================================================="
echo "All models processed."
echo "=========================================================="