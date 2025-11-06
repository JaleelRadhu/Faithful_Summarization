# Define variables for paths and arguments to make the file more readable and maintainable.
PYTHON_CMD := python
SCRIPT := evaluation_analyzer.py
INPUT_BASE_DIR := /home/abdullahm/jaleel/Faithfullness_Improver/outputs
# INPUT_DIRs := all_mistral all_qwen mistral_general mistral_qwen qwen_mistral qwen_mistral_no_cot qwen_mistral_no_eval
INPUT_DIRs := trial
INPUT_DIRS_LIST := $(foreach dir,$(INPUT_DIRs),$(INPUT_BASE_DIR)/$(dir))  # Create full paths
MODEL_NAME := Qwen/Qwen2.5-7B-Instruct
DATA_DIR := /home/abdullahm/jaleel/Faithfullness_Improver/data
RESULT_DIR := results_llm_eval_Qwen

# The 'run' target executes the python script.
run:
	@for dir in $(INPUT_DIRS_LIST); do \
		echo "Running evaluation $$dir..."; \
		$(PYTHON_CMD) $(SCRIPT) --result_dir $(RESULT_DIR) --input_dir $$dir --model_name "$(MODEL_NAME)" --data_dir $(DATA_DIR); \
	done
