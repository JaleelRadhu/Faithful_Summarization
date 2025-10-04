import re
from utils.prompt_builder import fill_prompt

def revise_summary(summary_data, feedback, corrector_prompt, pipe):
    correct_prompt = fill_prompt(corrector_prompt, {**summary_data, "Feedback": feedback})
    response = pipe(correct_prompt, max_new_tokens=512, temperature=0.7)
    revised_text = response[0]["generated_text"]

    # Split into CoT + revised summary
    parts = re.split(r"Part 2: Summary Revision|Revised Summary:", revised_text)
    if len(parts) == 1:
        return "", revised_text
    return parts[0].strip(), parts[-1].strip()
