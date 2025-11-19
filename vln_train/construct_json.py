'''
Source JSON format:
[
    {
        "video_id": "914-23",
        "q": "Turn and walk away from the bathroom into the bedroom area. Walk through the open door on the other side of the room and into the dining room area at the end of the hallway. Stop near the couches in the next room passed the kitchen. ",
        "a": "The next action is move forward 75 cm.",
        "frames": [
            "914/frame_0.jpg",
            "914/frame_1.jpg",
            "914/frame_2.jpg",
            "914/frame_3.jpg",
            "914/frame_4.jpg",
            "914/frame_5.jpg",
            "914/frame_6.jpg",
            "914/frame_7.jpg",
            "914/frame_8.jpg",
            "914/frame_9.jpg",
            "914/frame_10.jpg",
            "914/frame_11.jpg",
            "914/frame_12.jpg",
            "914/frame_13.jpg",
            "914/frame_14.jpg",
            "914/frame_15.jpg",
            "914/frame_16.jpg",
            "914/frame_17.jpg",
            "914/frame_18.jpg",
            "914/frame_19.jpg",
            "914/frame_20.jpg",
            "914/frame_21.jpg",
            "914/frame_22.jpg",
            "914/frame_23.jpg",
            "914/frame_24.jpg",
            "914/frame_25.jpg",
            "914/frame_26.jpg",
            "914/frame_27.jpg",
            "914/frame_28.jpg",
            "914/frame_29.jpg",
            "914/frame_30.jpg",
            "914/frame_31.jpg",
            "914/frame_32.jpg",
            "914/frame_33.jpg",
            "914/frame_34.jpg",
            "914/frame_35.jpg",
            "914/frame_36.jpg",
            "914/frame_37.jpg",
            "914/frame_38.jpg",
            "914/frame_39.jpg",
            "914/frame_40.jpg",
            "914/frame_41.jpg",
            "914/frame_42.jpg",
            "914/frame_43.jpg",
            "914/frame_44.jpg",
            "914/frame_45.jpg",
            "914/frame_46.jpg",
            "914/frame_47.jpg",
            "914/frame_48.jpg"
        ]
    },
    ...
]

Target JSON format:
[
  {
    "id": "000000033471",
    "image": ["000000033471.jpg", "000000033472.jpg"],
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n<image>\nIs the perspective of the camera differnt?"
      },
      {
        "from": "gpt",
        "value": "Yes, It the perspective of the camera is different."
      }
    ]
  }
  ...
]
'''

import json
import os

import argparse

pic_num = []


def convert_r2r_to_qwenvl(source_path, target_path, max_frames=33):
    with open(source_path, 'r') as f:
        source_data = json.load(f)

    target_data = []
    for item in source_data:
        # 生成唯一id（可用video_id或自定义）
        vid = item.get('video_id', '')
        # 取frames列表，假设每条数据至少有两帧
        images = item.get('frames', [])[-max_frames:]  # 取最近的 max_frames 帧
        pic_num.append(len(images))
        # 构造对话内容
        question = item.get('q', '').strip()
        answer = item.get('a', '').strip()
        conversations = [
            {
                "from": "human",
                "value": "<image>\n" * len(images) + 
f"""You are an expert navigation robot operating in an indoor environment. Your primary goal is to meticulously follow a multi-step instruction to reach a final destination.

At every step, you receive a sequence of historical RGB images that show your journey so far. You must first analyze this history to understand your current progress within the overall instruction, and then decide on the single next action.

**## Overall Navigation Instruction ##**
{question}

**## Your Thought Process (Crucial for success) ##**
Before outputting an action, you must internally reason as follows:
1.  **Review the Goal**: Re-read the complete instruction to remember the final destination and all intermediate steps.
2.  **Analyze Your History**: Look at the sequence of images from start to now. What key objects or turns have you already passed?
3.  **Localize Your Progress**: Compare your visual history to the instruction text. Determine exactly which part of the instruction you have just completed and which part is next. For example, think "I have just walked through the doorway, now I need to make a right."
4.  **Determine the Next Action**: Based on your current view (the latest image) and the *immediate next step* of the instruction, decide on the single most appropriate action.

**## Action Space and Required Output Format ##**
You must output **only one** of the following actions per turn, in the exact format specified.

1.  **Rotation Action**: If you need to turn to face the correct direction.
    *   **Format**: `The next action is turn <direction> <degrees> degree.`
    *   `<direction>` must be either `left` or `right`.
    *   `<degrees>` must be one of the discrete values: `5, 10, 15`.
    *   *Mapping*: A heading offset of `-15` corresponds to `left 15`, `+10` corresponds to `right 10`.

2.  **Forward Action**: If you are already facing the correct direction and need to move forward.
    *   **Format**: `The next action is move forward <distance> cm.`
    *   `<distance>` must be one of the discrete values in centimeters: `25, 75, 125, 175, 225, 275`.
    *   *Mapping*: This corresponds to the metric values `0.25m, 0.75m, ...`.

3.  **Stop Action**: Only when you are certain that you have fully completed *all parts* of the instruction.
    *   **Format**: You must output this exact sentence: `I think I should stop because I have finished theinstruction.`

**## Example of Your Task ##**

*   **Overall Instruction**: `Step out of the bath tub and walk to the door way on the left and walk through that door way. Then make a right and walk past the book case. After the book case make a left and walk past the desk to the double glass doors. walk through the double glass doors and wait by the picture of a mountain.`

*   **Scenario 1**: You are at the start, inside the bathtub, looking at the door on the left.
    *   *Internal Thought*: "I am at the beginning. The first step is 'walk to the door way on the left'. I am facing the door. I should move towards it."
    *   **Your Output**: `The next action is move forward 75 cm.`

*   **Scenario 2**: You have walked through the doorway and are now in a hallway. The bookcase is on your right.
    *   *Internal Thought*: "I have completed 'walk through that door way'. The next step is 'make a right and walk past the book case'. The bookcase is on my right, so first I need to turn right."
    *   **Your Output**: `The next action is turn right 15 degree.`

*   **Scenario 3**: You have walked through the double glass doors and are standing directly in front of a picture of a mountain.
    *   *Internal Thought*: "I have passed the bookcase, the desk, and walked through the double doors. The final instruction is 'wait by the picture of a mountain'. My current view confirms I am here. The instruction is complete."
    *   **Your Output**: `I think I should stop because I have finished the instruction.`


**## Your Turn ##**
Based on the historical images above and following the thought process and output format strictly, what is the next action to take?
"""
            },
            {
                "from": "gpt",
                "value": answer
            }
        ]
        target_data.append({
            "id": vid,
            "image": images,
            "conversations": conversations
        })
        # break
    with open(target_path, 'w') as f:
        json.dump(target_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert R2R-style JSON to Qwen-VL LLaVA-style JSON")
    parser.add_argument("--source", required=True, help="Source JSON path")
    parser.add_argument("--target", required=True, help="Target JSON path")
    parser.add_argument("--max-frames", type=int, default=33, help="Maximum number of frames to keep per example")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Source JSON not found: {args.source}")
    out_dir = os.path.dirname(args.target)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    convert_r2r_to_qwenvl(args.source, args.target, max_frames=args.max_frames)
    if pic_num:
        print(f"Mean number of images per epoch: {sum(pic_num) / len(pic_num)}")
        print(f"Max number of images per epoch: {max(pic_num)}")
        print(f"Min number of images per epoch: {min(pic_num)}")
    else:
        print("No images were processed.")