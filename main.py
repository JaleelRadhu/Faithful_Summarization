# import os
# from together import Together
import os
import sys
import json
import yaml
import torch
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoModelForCausalLM
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from rouge import Rouge
import csv
import re


# ----j's edit -------------

from dotenv import load_dotenv
load_dotenv()
from transformers import AutoModelForCausalLM, AutoTokenizer , pipeline
hf_token = os.getenv('HF_TOKEN')
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token = hf_token, device_map={"":1})
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# ---------------------------------------------------------------------------
# client = Together(api_key="b474c6d88c3b7e36d699745dbbce60aa915c53c56265b533187438528cf80fef")

add_data =[]
add_data.append(['PERSPECTIVE','FEEDBACK','REVISED_SUMMARY','COT_IMPROVE','ACTUAL_OUTPUT','QUESTION','ANSWERS','INPUT_SPANS','STARTING_SUMMARY'])
l=[]
l.append(['PERSPECTIVE','FEEDBACK','REVISED_SUMMARY','COT_IMPROVE','ACTUAL_OUTPUT','QUESTION','ANSWERS','INPUT_SPANS','STARTING_SUMMARY'])

# def call_together_api(prompt):
#     # api_key = os.environ.get("TOGETHER_API_KEY")
#     # client = Together(api_key=api_key)
#     response = client.chat.completions.create(
#         model="mistralai/Mistral-7B-Instruct-v0.2",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message.content

def call_hf_api(prompt: str):
    response = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]#       


def calculate_rouge_scores(reference_summary, candidate_summary):
    rouge = Rouge()
    scores = rouge.get_scores(candidate_summary, reference_summary, avg=True)
    return scores

def generate_feedback_and_revise(summary_data, score_reasoning_improvement_prompt, corrector_prompt, previous_summary, past_experience):


    question = summary_data['question']
    # print(type(question))
    answers = summary_data['answers']
    non_empty_sentences = ' '.join([sentence.replace('\n', '') for sentence in answers])
    # print(type(non_empty_sentences))
    perspective = summary_data['perspective']
    # print(type(perspective))
    input_spans = summary_data['input_spans']
    # print(type(input_spans))
    # print(type(system_output))
    actual = summary_data['Actual']
    
    defn = ''
    if  perspective ==  "SUGGESTION":
        defn = "Defined as advice or recommendations to assist users in making informed medical decisions, solving problems, or improving health issues."

    if  perspective == "INFORMATION":
        defn = "Defined as knowledge about diseases, disorders, and health-related facts, providing insights into symptoms and diagnosis."
        
    if  perspective == "EXPERIENCE":
        defn = "Defined as individual experiences, anecdotes, or firsthand insights related to health, medical treatments, medication usage, and coping strategies"

    if  perspective == "CAUSE":
        defn = "Defined as reasons responsible for the occurrence of a particular medical condition, symptom, or disease"
    
    if  perspective == "QUESTION":
        defn = "Defined as inquiry made for deeper understanding."
    
    
    cur_prompt = score_reasoning_improvement_prompt.replace('{{Perspective}}', perspective).replace('{{Question}}', question).replace('{{Perspective_Def}}', defn).replace('{{Answers}}', answers).replace('{{Input spans}}', input_spans).replace('{{Given Summary}}', previous_summary).replace('{{Perspective_Def}}', defn)
    # print("__________________________PROMPT______________________",cur_prompt)
    # input_ids = tokenizer(cur_prompt, return_tensors="pt").input_ids.to('cuda')
    # feedback = call_together_api(cur_prompt)
    feedback = call_hf_api(cur_prompt) #j's edit
    # output = model.generate(input_ids, do_sample=True, top_k=10, top_p=0.95, repetition_penalty=1.2, temperature=0.8, max_new_tokens=1000)
    # _response = tokenizer.decode(output[0], skip_special_tokens=True).split("- Score for Redundancy: (Add your score here)")[1]
    # feedback = tokenizer.decode(output[0], skip_special_tokens=True).split("- Using all the Reasonings for each of the above error types and provide chain of thought improvements to eliminate these error types from the given summary thus making it faithful. List down step-by-step improvements.")[1]

    print("===FEEDBACK===" * 3)
    print(feedback)
    print("===FEEDBACK_END===" * 3)

    correct_prompt = corrector_prompt.replace('{{Perspective}}', perspective).replace('{{Question}}', question).replace('{{Perspective_Def}}', defn).replace('{{Answers}}', answers).replace('{{Input spans}}', input_spans).replace('{{Given Summary}}', previous_summary).replace('{{Feedback}}', feedback)
    # print("_______________________CORRECTOR___PROMPT______________________",correct_prompt)
    # input_ids_2 = tokenizer(correct_prompt, return_tensors="pt").input_ids.to('cuda')
    # output_2 = model.generate(input_ids_2, do_sample=True, top_k=10, top_p=0.95, repetition_penalty=1.2, temperature=0.8, max_new_tokens=1000)
    # _response = tokenizer.decode(output[0], skip_special_tokens=True).split("- Score for Redundancy: (Add your score here)")[1]
    # revised_summary = tokenizer.decode(output_2[0], skip_special_tokens=True).split("- [Provide the revised summary here incorporating all improvements.]")[1]
    # call_together_api(correct_prompt)
    # revised_summary = call_together_api(correct_prompt)
    revised_summary = call_hf_api(correct_prompt) #j's edit
    print("===REVISED_SUMMARY===" * 3)
    print(revised_summary)
    print("===REVISED_SUMMARY_END===" * 3)
    parts = re.split(r"Part 2: Summary Revision|Rewritten Summary|Revised Summary:", revised_summary)
    print("PARTS________",parts)
    # parts = revised_summary.split("Part 2: Summary Revision")
    if len(parts) == 1:
    # If there's only one part, assume no headers were found and the entire content is the revised summary
        revised_summary = parts[0]
        chain_of_thought_improvement = ""
    else:
        # Otherwise, assume the first part is the chain of thought improvements, and the last part is the revised summary
        chain_of_thought_improvement = parts[0].strip() if len(parts) > 0 else ""
        revised_summary = parts[-1].strip() if len(parts) > 1 else ""
        chain_of_thought_improvement = parts[0]  if len(parts) > 0 else ""
        print("chain_of_thought_improvement",chain_of_thought_improvement)
        revised_summary = parts[-1] if len(parts) > 1 else ""
        print("revised_summary",revised_summary)
    

  

    return feedback, revised_summary, chain_of_thought_improvement


def run_trial(summary_data, score_reasoning_improvement_prompt, corrector_prompt) :

    previous_summary = summary_data['predicted']
    iteration_count = 0
    max_iterations = 5 
    previous_rouge_score = None
    tolerance = 0.01  # Define tolerance for stopping condition based on summary change
    past_experience='No experience'
    while iteration_count < max_iterations:

        feedback, revised_summary, chain_of_thought_improvement = generate_feedback_and_revise(summary_data, score_reasoning_improvement_prompt, corrector_prompt, previous_summary, past_experience)


        print()
        current_rouge_scores = calculate_rouge_scores(revised_summary, summary_data['Actual'])
        print("ROUGE Scores:", current_rouge_scores)
        current_rouge_score = current_rouge_scores['rouge-l']['f']  # For example, focusing on F1-score of ROUGE-1

        print(f"Iteration {iteration_count + 1}: ROUGE-1 F1-score = {current_rouge_score}")

        # Check if the ROUGE score has improved
        if previous_rouge_score is not None and current_rouge_score <= previous_rouge_score or iteration_count == max_iterations - 1:
            print("No improvement in ROUGE score. Stopping iterations.")
            summary_data["Final_feedback"] = feedback
            summary_data["Final_revised_summary"] = revised_summary
            break
       
        previous_summary = revised_summary
        previous_rouge_score = current_rouge_score
        past_experience= feedback
        iteration_count += 1

    
    add_data.append([summary_data['perspective'], feedback, revised_summary, chain_of_thought_improvement ,summary_data['Actual'], summary_data['question'], summary_data['answers'],  summary_data['input_spans'], summary_data['predicted']])

    l.pop()
    l.append([summary_data['perspective'], feedback, revised_summary,chain_of_thought_improvement ,summary_data['Actual'], summary_data['question'], summary_data['answers'],  summary_data['input_spans'], summary_data['predicted']])
    with open('/home/abdullahm/jaleel/FAITH_RL/SELF/checkpoint/flan_remian_j.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(l)
    return revised_summary, feedback, chain_of_thought_improvement

        
def main():
    # model_name =  'mistralai/Mistral-7B-Instruct-v0.2'
    # bnb_config = BitsAndBytesConfig(
    # load_in_4bit=True,
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16
    # )
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # mymodel = AutoModelForCausalLM.from_pretrained(model_name,  quantization_config=bnb_config,)

    # max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    # dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    # load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    # mymodel, tokenizer = FastLanguageModel.from_pretrained(
    # model_name =   "unsloth/mistral-7b-instruct-v0.2-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    # max_seq_length = max_seq_length,
    # dtype = dtype,
    # load_in_4bit = load_in_4bit,
    # )

    ###########   PROMPTS ############
    score_reasoing_improvement_prompt = open('/home/abdullahm/jaleel/FAITH_RL/SELF/prompts/f_score_reason.txt').read()
    corrector_prompt = open('/home/abdullahm/jaleel/FAITH_RL/SELF/prompts/cot1.txt').read()
    ###########   DATA ############
    csv_file_path = "/home/abdullahm/jaleel/FAITH_RL/SELF/data/flant5_remaining.csv"

    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    # sampled_df = df.sample(n=200, random_state=42)

    # Display the first few rows of the dataframe to understand its structure
    print(df.head())

    for i in range(df.shape[0]):
        row_data = df.iloc[i] 

        summary_data = {
            'question': row_data['question'],  
            'answers': row_data['Answers'], 
            'perspective': row_data['Perspective'],
            'predicted' : row_data['Predicted'],
            'input_spans': row_data['Input_spans'],
            'Actual' : row_data['Actual']
        }

        print(summary_data)
        revised_summary, feedback, chain_of_thought_improvement = run_trial(summary_data, score_reasoing_improvement_prompt, corrector_prompt)
        # print("Revised Summary:", revised_summary)
        # print("Evaluation:", evaluation)


    with open('/home/abdullahm/jaleel/FAITH_RL/SELF/checkpoint/refined_summary_data_results_final_flan_remin_j.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(add_data)

if __name__ == "__main__":
    main()
    #test the hf_client
    
    # test_prompt = "Write a poem about the sea."
    # print("HF API Response:", call_hf_api(test_prompt))