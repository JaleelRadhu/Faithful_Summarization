import re

def revise_summary(summary_data, feedback, improver_prompt, pipe):
    # Fill the improver prompt with summary, perspective, etc.
    
    # final_improver_prompt = fill_prompt(improver_prompt, {**summary_data, "Feedback": feedback})
    final_improver_prompt = improver_prompt.format(**{**summary_data, "Feedback": feedback})
    
    # Get model response
    response = pipe(final_improver_prompt, max_new_tokens=512, temperature=0.7)
    revised_text = response[0]["generated_text"]

    # --- Extract Part 1 (Improvements) ---
    part1 = None
    match_part1 = re.search(r"Part 1:.*?(?=Part 2:)", revised_text, re.DOTALL)
    if match_part1:
        part1 = match_part1.group(0).strip()

    # --- Extract Part 2 (only final revised summary) ---
    part2_summary = None
    match_part2 = re.search(r"Rewritten Summary:\s*[-–]?\s*(.*)", revised_text, re.DOTALL)
    if match_part2:
        part2_summary = match_part2.group(1).strip()

    return {
        "full_output": revised_text.strip(),
        "part1_improvements": part1,
        "revised_summary": part2_summary
    }
