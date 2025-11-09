NUM_SAMPLES = 2
SOURCE_FILE_NAME=filtered_complete_base_google_gemma-2-9b.json
EVAL_MODELS = Qwen_Qwen2.5-7B-Instruct_improved mistral gemma-3-12b-it

run :
	@echo "running human evaluaation csv and mapping generator script"
	python human_evaluation_structure_script.py --num_samples $(NUM_SAMPLES) --source_file_name $(SOURCE_FILE_NAME) --eval_models $(EVAL_MODELS)