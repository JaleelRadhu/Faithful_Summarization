import streamlit as st
import pandas as pd
import os
import json
import time
import gspread
from google.oauth2.service_account import Credentials

DATA_FILE = 'human_eval.csv'
PERSPECTIVE_DEFN_FILE = 'prompts/perspective_defn.json'
GSHEET_NAME = "Human Evaluation Results" # The name of the Google Sheet you created

# --- Helper Functions ---

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'evaluator_name' not in st.session_state:
        st.session_state.evaluator_name = ""
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'scores' not in st.session_state:
        st.session_state.scores = {}
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None

def load_data():
    """Load the evaluation data from the CSV file."""
    if not os.path.exists(DATA_FILE):
        st.error(f"Error: The data file '{DATA_FILE}' was not found. Please run the data preparation script first.")
        st.stop()
    return pd.read_csv(DATA_FILE)

def load_perspective_definitions():
    """Loads the perspective definitions from the JSON file."""
    if not os.path.exists(PERSPECTIVE_DEFN_FILE):
        st.error(f"Error: The perspective definitions file '{PERSPECTIVE_DEFN_FILE}' was not found.")
        return {}
    with open(PERSPECTIVE_DEFN_FILE, 'r') as f:
        return json.load(f)

@st.cache_resource
def get_gsheet():
    """Connect to Google Sheets and return the specific sheet."""
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(st.secrets["google_credentials"], scopes=scopes)
    client = gspread.authorize(creds)
    return client.open(GSHEET_NAME).sheet1

def get_all_results_df():
    """Fetch all data from the Google Sheet and return as a DataFrame."""
    data = get_gsheet().get_all_records()
    st.session_state.results_df = pd.DataFrame(data)

def save_results(evaluator_name, sample_id, scores):
    """Save or update the evaluation scores in the Google Sheet."""
    sheet = get_gsheet()
    df_results = st.session_state.results_df

    # Prepare the new data row as a dictionary
    new_row_data = {'evaluator_name': evaluator_name, 'sample_id': int(sample_id)}
    for summary_key, metrics in scores.items():
        for metric_name, score_value in metrics.items():
            column_name = f"{summary_key}_{metric_name}"
            new_row_data[column_name] = int(score_value)

    # Convert the new row to a DataFrame for local update
    new_row_df = pd.DataFrame([new_row_data])

    # Check if an entry for this evaluator and sample already exists
    if not df_results.empty:
        existing_indices = df_results[(df_results['evaluator_name'] == evaluator_name) & (df_results['sample_id'] == sample_id)].index
    else:
        existing_indices = []

    if len(existing_indices) > 0:
        # Update the existing row in the Google Sheet
        # gspread row indices are 1-based, and we add 1 for the header
        sheet_row_index = existing_indices[0] + 2
        # Get header to find column indices
        header = sheet.row_values(1)
        update_cells = []
        for col_name, value in new_row_data.items():
            try:
                # gspread col indices are 1-based
                col_index = header.index(col_name) + 1
                update_cells.append(gspread.Cell(sheet_row_index, col_index, value))
            except ValueError:
                # Column not in sheet, skip
                pass
        if update_cells:
            sheet.update_cells(update_cells)
        # Update the local DataFrame in session state
        for col, val in new_row_data.items():
            if col in df_results.columns:
                st.session_state.results_df.loc[existing_indices[0], col] = val
    else:
        # Append a new row to the Google Sheet
        header = sheet.row_values(1)
        # Order the new_row_data dict to match the header order
        ordered_row = [new_row_data.get(col, "") for col in header]
        sheet.append_row(ordered_row)
        # Update the local DataFrame in session state
        st.session_state.results_df = pd.concat([df_results, new_row_df], ignore_index=True)


def show_definitions_modal(modal_type):
    """Display a modal with definitions for perspectives or metrics."""
    definitions = load_perspective_definitions()
    with st.expander(f"View {modal_type.capitalize()} Definitions", expanded=False):
        if modal_type == "perspective":
            if definitions:
                for perspective, definition in definitions.items():
                    st.markdown(f"- **{perspective}**: {definition}")
            else:
                st.warning("Perspective definitions could not be loaded.")
        elif modal_type == "metrics":
            st.markdown("""
            #### 1. Fluency
            How fluent and grammatically correct is the summary?
            
            ---
            #### 2. Coherence
            How logically connected and well-structured the summary is.
            
            ---
            #### 3. Extraneous Information
            Measures how much information in the summary is unsupported or irrelevant to the provided answers.
            - **5**: No extraneous information, fully supported
            - **4**: Minimal extraneous info, mostly aligned
            - **3**: Moderate extraneous info, somewhat aligned
            - **2**: Significant extraneous info, poorly aligned
            - **1**: Mostly or entirely extraneous, not supported
            
            ---
            #### 4. Contradiction
            Evaluates whether the summary contains statements contradicting the given answers.
            - **5**: No contradictions at all
            - **4**: Minor contradictions
            - **3**: Some noticeable contradictions
            - **2**: Major contradictions
            - **1**: Completely inconsistent with the answers
            
            ---
            #### 5. Perspective Misalignment
            Assesses how well the summary aligns with the given perspective (e.g., Suggestion, Information, Experience, Cause, or Question).
            - **5**: Fully aligned with the given perspective
            - **4**: Mostly aligned, with small deviations
            - **3**: Balanced between correct and different perspectives
            - **2**: Primarily presents a different perspective
            - **1**: Completely misaligned with the given perspective
            
            ---
            #### 6. Redundancy
            Measures the level of repetition or unnecessary restating of information.
            - **5**: No redundancy, concise and clear
            - **4**: Slight redundancy, does not harm clarity
            - **3**: Moderate redundancy, some repetition
            - **2**: High redundancy, frequent repetition
            - **1**: Extremely redundant, overwhelming repetition
            """)

def show_instructions_modal():
    """Display a modal with the evaluation instructions."""
    with st.expander("View Instructions", expanded=False):
        st.markdown("""
        ### Evaluation Instructions Summary
        The summary is of input spans from answers on reddit to healthcare related question.
        You are required to evaluate and score summaries based on the provided Questions, List of Answers, Perspective, and Relevant Spans from the answers.
        
        **Note:** When evaluating, compare each summary **only against the provided input spans** from the answers. Use the full answers only for context if any span seems unclear or incomplete. The final evaluation should be based solely on how well the summary aligns with the input spans, not the entire answers.
        
        Each summary must be evaluated on six metrics (Fluency, Coherence, Extraneous, Contradiction, Perspective Misalignment, Redundancy). All metrics are scored on a 1–5 scale (1 = poor, 5 = excellent).
        """)

# --- Page Rendering Functions ---

def render_login_page():
    """Render the initial login page."""
    st.title("Human Evaluation Login")
    st.info(
        "**Welcome!** Please enter your name to begin or resume an evaluation session. "
        "If you are returning, use the exact same name to continue where you left off."
    )
    name = st.text_input("Please enter your name to begin:")
    if st.button("Start Evaluation"):
        if name:
            st.session_state.evaluator_name = name
            st.session_state.page = 'instructions'
            st.rerun()
        else:
            st.warning("Please enter your name.")

def render_instructions_page():
    """Render the instructions page."""
    st.title(f"Welcome, {st.session_state.evaluator_name}!")
    st.header("Evaluation Instructions")
    st.markdown("""
    Thank you for helping with this evaluation task. 
    Your feedback is crucial for evaluating our models.
    You will be given summaries of input spans from answers on reddit to healthcare related questions. The summaries are supposed to from a particular perspective as defined ahead.
    
    **Your Task:** You are required to evaluate and score summaries based on the provided Questions, List of Answers, Perspective, and Relevant Spans from the answers.

    **Metrics:** Each summary must be evaluated on six metrics: Fluency, Coherence, Extraneous, Contradiction, Perspective Misalignment, and Redundancy. All metrics are scored on a 1–5 scale (1 = poor, 5 = excellent).
    [You will have the definitions of each ahead !]
    
    
    ### **CRITICAL NOTE**
    When evaluating, compare each summary **only against the provided input spans** from the answers.
    
    Use the full answers **only for context** if any span seems unclear or incomplete.
    
    The final evaluation should be based solely on how well the summary aligns with the **input spans**, not the entire answers.
    """)
    if st.button("I Understand, Let's Begin!"):
        st.session_state.page = 'evaluation'
        st.rerun()

def render_evaluation_page(df):
    """Render the main evaluation interface."""
    # Scroll to the top of the page on every rerun using a slight delay
    # This ensures the script runs after the DOM has updated.
    st.markdown(
        """
        <script>
            setTimeout(function() { window.scrollTo(0, 0); }, 0);
        </script>
        """,
        unsafe_allow_html=True
    )

    # Ensure results are loaded into the session state if they aren't already
    if st.session_state.results_df is None:
        get_all_results_df()

    # Find the next un-evaluated sample for the current evaluator
    # This logic runs only when the page is first loaded for the user
    if 'initialized' not in st.session_state:
        # Fetch results ONCE and store in session state
        get_all_results_df()
        df_results = st.session_state.results_df

        if not df_results.empty:
            evaluated_samples = df_results[df_results['evaluator_name'] == st.session_state.evaluator_name]['sample_id'].unique()
            unevaluated_df = df[~df['id'].isin(evaluated_samples)]
            if unevaluated_df.empty:
                st.session_state.page = 'thank_you'
                st.rerun()
            else:
                st.session_state.current_index = unevaluated_df.index[0]
        else:
            st.session_state.current_index = 0

        # Mark that the initial index has been set for this session
        st.session_state.initialized = True

    # This line ensures we don't reset the index on every rerun
    sample = df.iloc[st.session_state.get('current_index', 0)]
    sample_id = sample['id']

    # --- Load existing scores for this sample to pre-fill the form ---
    existing_scores = {}
    df_results = st.session_state.results_df # Use the cached DataFrame
    if not df_results.empty:
        score_row = df_results[(df_results['evaluator_name'] == st.session_state.evaluator_name) & (df_results['sample_id'] == sample_id)]
        if not score_row.empty:
            # Convert the first found row to a dictionary
            existing_scores = score_row.iloc[0].to_dict()

    st.title(f"Evaluation for Sample ID: {sample_id}")
    
    st.progress((st.session_state.current_index + 1) / len(df))
    st.write(f"Sample {st.session_state.current_index + 1} of {len(df)}")

    # Create a two-column layout
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        # --- Display Context in the left column ---
        st.subheader("Context")
        st.markdown(f"**Question:** {sample['question']}")
        st.markdown(f"**Perspective:** `{sample['Perspective']}`")
        
        # Place full answers inside an expander button
        with st.expander("Show Full Reference Answers"):
            st.text(sample['answers'])
        
        # Highlight the Input Spans section
        st.markdown("**Input Spans (Primary Evaluation Source):**")
        st.warning(
            "**IMPORTANT:** Base your evaluation primarily on these **Input Spans**. "
            "Use the 'Show Full Reference Answers' button above only for additional context if a span is unclear."
        )
        st.info(sample['Input Spans'])

        st.markdown("---")
        st.markdown("_The buttons below are for your reference if you forget the instructions or definitions._")
        show_instructions_modal()
        show_definitions_modal("perspective")
        show_definitions_modal("metrics")

    with right_col:
        # --- Scoring Section in the right column (scrollable) ---
        st.subheader("Summaries for Evaluation")
        
        with st.container(height=800): # This makes the content below scrollable
            score_categories = ['Fluency', 'Coherence', 'Extraneous', 'Contradiction', 'Perspective Misalignment', 'Redundancy']
            
            # Initialize scores for the current sample
            st.session_state.scores[sample_id] = {}

            for i in range(1, 7):
                summary_col = f'summary_{i}'
                st.session_state.scores[sample_id][summary_col] = {}
                
                st.markdown(f"---")
                st.markdown(f"**Summary {i}**")
                st.info(sample[summary_col])

                # Create columns for click-based scoring
                score_cols = st.columns(len(score_categories))
                for idx, category in enumerate(score_categories):
                    with score_cols[idx]:
                        # Check for a pre-existing score to set the default
                        score_column_name = f"{summary_col}_{category}"
                        previous_score = existing_scores.get(score_column_name)
                        default_index = int(previous_score - 1) if pd.notna(previous_score) else 0

                        # Use radio buttons for scoring
                        score = st.radio(
                            label=category,
                            options=[1, 2, 3, 4, 5],
                            index=default_index,
                            key=f"score_{sample_id}_{i}_{category}",
                            horizontal=True
                        )
                        st.session_state.scores[sample_id][summary_col][category] = score

        st.divider()

        # --- Navigation Buttons ---
        nav_cols = st.columns([1, 1, 1]) # Give equal space
        with nav_cols[0]:
            if st.button("⬅️ Previous Sample", disabled=(st.session_state.current_index == 0)):
                st.session_state.current_index -= 1
                st.rerun()

        with nav_cols[2]:
            if st.button("Submit and Go to Next ➡️", type="primary"):
                # Save the collected scores
                save_results(st.session_state.evaluator_name, sample_id, st.session_state.scores[sample_id])
                
                # Clear scores for the next round and show success message
                if sample_id in st.session_state.scores:
                    del st.session_state.scores[sample_id]
                
                st.success(f"Scores for sample {sample_id} submitted successfully!")
                time.sleep(1) # Give user time to see the message
                
                # Move to the next sample or finish
                if st.session_state.current_index + 1 >= len(df):
                    st.session_state.page = 'thank_you'
                else:
                    # Simply move to the next index
                    st.session_state.current_index += 1
                
                st.rerun()

def render_thank_you_page():
    """Render the final thank you page."""
    st.balloons()
    st.title("Evaluation Complete!")
    st.header(f"Thank you, {st.session_state.evaluator_name}, for your contribution!")
    st.markdown("You have successfully evaluated all the available samples. You can now close this window.")
    
    st.markdown("### Download All Results")
    df_results = st.session_state.results_df
    if not df_results.empty:
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download all results as CSV",
            data=csv,
            file_name="human_eval_results.csv",
            mime="text/csv",
        )

# --- Main App Logic ---

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide", page_title="Summary Evaluation")
    st.success("Thank you for your time and effort!")
    
    initialize_session_state()
    df = load_data()

    if st.session_state.page == 'login':
        render_login_page()
    elif st.session_state.page == 'instructions':
        render_instructions_page()
    elif st.session_state.page == 'evaluation':
        render_evaluation_page(df)
    elif st.session_state.page == 'thank_you':
        render_thank_you_page()

if __name__ == '__main__':
    main()
