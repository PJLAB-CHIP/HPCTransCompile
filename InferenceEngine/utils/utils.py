import os
from os.path import join, exists
import re

# Combined pattern for both tag types and code blocks
CODE_PATTERN = re.compile(
    R"\s*(?:<(cpp|cpu|cuda)>(.*?PYBIND11_MODULE.*?)</\1>|"
    R"```(?:cpp|cpu|cuda)\n(.*?PYBIND11_MODULE.*?)```)",
    re.DOTALL,
)


def extract_code(content, action):
    if action == "conversion":
        return content

    # Determine tag type based on action
    tag_type = "cuda" if action == "translation" else "cpu"

    # Find all matches of our pattern
    matches = list(CODE_PATTERN.finditer(content))

    if not matches:
        print(f"Fail to find code block in the expected format.")
        return None

    # Process the last match
    first_match = matches[-1]
    tag, tag_content, code_block = first_match.groups()

    # Function to clean nested code blocks
    def clean_code(code):
        lines = code.strip().split("\n")
        # Remove leading code block markers if present
        if lines and (
            lines[0].startswith("```cpp") or lines[0].startswith("```cpu")
        ):
            lines = lines[1:]
        # Remove trailing marker if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)

    if tag == tag_type:
        # Tag matched (cpu or cuda)
        return clean_code(tag_content)
    elif tag is None and code_block and tag_type == "cpu":
        # Code block matched and we're looking for CPU code
        return clean_code(code_block)
    elif tag_type == "cuda":
        print("Fail to find <cuda></cuda> template.")
        return None

    return None


def save_to_file(content, file_name, suffix, action, save_path=None):
    code = extract_code(content, action)
    if save_path != None:
        file_path = join(save_path, f"{file_name}{suffix}")
        content_path = join(save_path, f"{file_name}.txt")
    else:
        file_path = f"{file_name}{suffix}"
        content_path = f"{file_name}.txt"

    # Content
    if content != None:
        with open(content_path, "w") as content_file:
            content_file.write(content)
    # Code
    code = extract_code(content, action)
    if code != None:
        with open(file_path, "w") as file:
            file.write(code)


def load_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content
