import json


def extract_accelerate_rate(json_file):
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    result = [
        item["torch_runtime"] / item["runtime"]
        for key, item in data.items()
        if item["correctness"] and item["runtime"] > 0]
    print(result)
    return result

if __name__ == '__main__':
    extract_accelerate_rate('/code/LLM4HPCTransCompile/Results/QwenCoder_14b/1_eval_results.json')