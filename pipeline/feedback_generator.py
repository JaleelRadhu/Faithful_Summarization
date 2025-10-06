from utils.prompt_builder import fill_prompt

def generate_feedback(summary_data, evaluator_prompt, pipe):
    # cur_prompt = fill_prompt(evaluator_prompt, summary_data)
    cur_prompt = evaluator_prompt.format(**summary_data)
    response = pipe(cur_prompt, max_new_tokens=512, temperature=0.7, return_full_text=False)
    return response[0]["generated_text"]

