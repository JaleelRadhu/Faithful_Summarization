# ==============================================================================
# Makefile for running the vLLM OpenAI API Server
# ==============================================================================

# --- Configuration Variables ---
# These can be overridden from the command line.
# Example: make run MODEL_NAME="meta-llama/Llama-2-7b-chat-hf" PORT=8001

# The name of the model to use from the Hugging Face Hub.
MODEL_NAME ?= "Qwen/Qwen2.5-7B-Instruct"

# The number of GPUs to use for tensor parallelism.
TENSOR_PARALLEL_SIZE ?= 2

# The port to run the API server on.
PORT ?= 8000

# The maximum GPU memory utilization (a value between 0.0 and 1.0).
GPU_MEM_UTIL ?= 0.50

# The host to bind the server to. Use 0.0.0.0 to allow remote connections.
HOST ?= "0.0.0.0"

# --- Phony Targets ---
# Declares targets that are not actual files, ensuring they always run.
.PHONY: run help

# --- Main Target ---

## run: Starts the vLLM OpenAI API server with the specified configuration.
run:
	@echo "🚀 Starting vLLM server..."
	@echo "   - Model: $(MODEL_NAME)"
	@echo "   - Port: $(PORT)"
	@echo "   - Tensor Parallel Size: $(TENSOR_PARALLEL_SIZE)"
	@echo "   - Host: $(HOST) (0.0.0.0 allows remote connections)"
	@echo "   - GPU Memory Utilization: $(GPU_MEM_UTIL)"
	@echo "--------------------------------------------------"
	python -m vllm.entrypoints.openai.api_server \
		--model $(MODEL_NAME) \
		--tensor-parallel-size $(TENSOR_PARALLEL_SIZE) \
		--port $(PORT) \
		--host $(HOST) \
		--gpu-memory-utilization $(GPU_MEM_UTIL)

# --- Helper Target ---

## help: Displays this help message.
help:
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
