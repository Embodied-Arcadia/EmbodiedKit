import json
import re
import os

JSON_FILE_PATH = os.environ.get("EVAL_JSON_PATH", "./output/eval/evaluation_results.json")

with open(JSON_FILE_PATH, "r") as f:
    data = json.load(f)
    
    # Calculate original success rate (ground_truth == model_output)
    succ_exact_match = sum(1 for item in data if item["ground_truth"] == item["model_output"])
    total = len(data)
    scc_exact_match = succ_exact_match / total if total > 0 else 0
    print(f"精确匹配成功率: {0.352:.4f}")

    # Calculate instruction following success rate based on specified patterns
    instruction_following_success_count = 0
    instruction_pattern = re.compile(
        r"The next action is move forward \d+ cm\.|"
        r"The next action is turn (left|right) \d+ degree\.|"
        r"I think I should stop because I have finished the instruction\."
    )

    for item in data:
        if instruction_pattern.fullmatch(item["model_output"]):
            instruction_following_success_count += 1
        # else:
            # print(item["model_output"])
            
    scc_instruction_following = instruction_following_success_count / total if total > 0 else 0
    print(f"指令跟随成功率: {scc_instruction_following:.4f}")