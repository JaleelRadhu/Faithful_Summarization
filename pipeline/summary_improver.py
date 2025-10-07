import re

def revise_summary(summary_data, feedback, improver_prompt, pipe):
    # Fill the improver prompt with summary, perspective, etc.
    final_improver_prompt = improver_prompt.format(**{**summary_data, "Feedback": feedback})
    
    # Get model response
    response = pipe(final_improver_prompt, max_new_tokens=512, temperature=0.7, return_full_text=False)
    revised_text = response[0]["generated_text"]
    
    # --- Extract Part 1 (Improvements) ---
    part1 = None
    # Updated regex to match "### Part 1:" format
    match_part1 = re.search(r"###\s*Part 1:.*?(?=###\s*Part 2:)", revised_text, re.DOTALL | re.IGNORECASE)
    if match_part1:
        part1 = match_part1.group(0).strip()
    
    # --- Extract Part 2 (only final revised summary) ---
    part2_summary = None
    
    # Try multiple patterns to match different output formats
    patterns = [
        # Pattern 1: **Rewritten Summary:** followed by bullet point
        r"\*\*Rewritten Summary:\*\*\s*\n\s*-\s*(.*?)(?=\n\s*---|$)",
        # Pattern 2: - **Rewritten Summary**:
        r"-\s*\*\*Rewritten Summary\*\*:\s*(.*?)(?=\n\s*---|$)",
        # Pattern 3: **Rewritten Summary:** without bullet
        r"\*\*Rewritten Summary:\*\*\s*(.*?)(?=\n\s*---|$)",
        # Pattern 4: Generic fallback - capture everything after "Rewritten Summary"
        r"Rewritten Summary[:\*\s]*\n?\s*-?\s*(.*?)(?=\n\s*---|$)",
    ]
    
    for pattern in patterns:
        match_part2 = re.search(pattern, revised_text, re.DOTALL | re.IGNORECASE)
        if match_part2:
            part2_summary = match_part2.group(1).strip()
            # Clean up any remaining artifacts
            part2_summary = re.sub(r'\s*---\s*$', '', part2_summary).strip()
            break
    
    
    return {
        "full_output": revised_text.strip(),
        "part1_improvements": part1,
        "revised_summary": part2_summary
    }