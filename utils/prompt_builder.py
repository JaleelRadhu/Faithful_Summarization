
def fill_prompt(template: str, data: dict):
    template = template.format(**data)
    return template




if __name__ == "__main__":
    template = open("prompts/test.txt").read()
    data = {"name": "Alice", "day": "Monday"}
    result = fill_prompt(template, data)
    print(result)  # Output: Hello, Alice! Today is Monday.