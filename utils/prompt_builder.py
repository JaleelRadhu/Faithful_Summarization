from utils.prompt_builder import fill_prompt

def generate_feedback(summary_data, score_prompt, pipe):
    cur_prompt = prompt_builder(score_prompt, summary_data)
    response = pipe(cur_prompt, max_new_tokens=512, temperature=0.7)
    return response[0]["generated_text"]
