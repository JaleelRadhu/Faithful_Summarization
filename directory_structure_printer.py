import os

def print_dir_structure(root_dir, indent=""):
    for root, dirs, files in os.walk(root_dir):
        # print(root, dirs, files)
        if ".git" in root:
            # print("Skipping .git directory")
            continue
        level = root.replace(root_dir, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 4 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")

# Example
print_dir_structure("/home/abdullahm/jaleel/Faithfullness_Improver")
