# Define variables for paths and files to make the Makefile cleaner and easier to maintain.
PROJECT_ROOT   := /home/abdullahm/jaleel/Faithfullness_Improver
PYTHON_ENV     := /home/abdullahm/miniconda3/envs/jaleel_FS_env/bin/python
DATA_DIR       := $(PROJECT_ROOT)/data

# --- List of input files to process ---
# Add the base names of your input files from the data/ directory here.
INPUT_FILES    := complete_base_dummy.json 

run: 
	@echo "--- Starting summary improvement process for multiple files ---"
	@# Loop over each file defined in the INPUT_FILES variable
	@for file in $(INPUT_FILES); do \
		echo "\n[INFO] Processing file: $$file"; \
		$(PYTHON_ENV) $(PROJECT_ROOT)/main_new.py --input_data $(DATA_DIR)/$$file || echo "[WARN] Failed to process $$file. Continuing..."; \
	done
	
	@echo "\n[INFO] Running final evaluation script..."
	$(PYTHON_ENV) $(PROJECT_ROOT)/utils/get_all_evaluation_results.py 

# run this to get the improved summaries. it will be in output/ directory