# Define variables for paths and files to make the Makefile cleaner and easier to maintain.
PROJECT_ROOT   := /home/abdullahm/jaleel/Faithfullness_Improver
PYTHON_ENV     := /home/abdullahm/miniconda3/envs/jaleel_FS_env/bin/python
DATA_DIR       := $(PROJECT_ROOT)/data

# --- List of input files to process ---
# Add the base names of your input files from the data/ directory here.

# INPUT_FILES    := filtered_complete_base_google_gemma-2-9b.json
INPUT_FILES    := dumm.json


run: 
	@echo "--- Starting summary improvement process for multiple files ---"
	@# Loop over each file defined in the INPUT_FILES variable
	@for file in $(INPUT_FILES); do \
		echo "\n[INFO] Processing file: $$file"; \
		$(PYTHON_ENV) $(PROJECT_ROOT)/main_new.py --input_data $(DATA_DIR)/$$file --is_evaluator_and_improver_same False --general_eval_and_improver True || echo "[WARN] Failed to process $$file. Continuing..."; \
	done
	
# 	@echo "\n[INFO] Running final evaluation script..."
# 	$(PYTHON_ENV) $(PROJECT_ROOT)/utils/get_all_evaluation_results.py 

# run this to get the improved summaries. it will be in output/ directory