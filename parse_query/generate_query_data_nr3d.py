# SeeGround/parse_query/generate_query_data_nr3d.py
import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
from openai import OpenAI
import csv # [수정] csv 라이브러리 사용

def is_explicitly_view_dependent(tokens):
    target_words = {"front", "behind", "back", "right", "left", "facing", "leftmost", "rightmost", "looking", "across"}
    return len(set(tokens).intersection(target_words)) > 0

def decode_stimulus_string(s):
    parts = s.split("-", maxsplit=4)
    if len(parts) == 4:
        scene_id, instance_label, n_objects, target_id = parts
        distractor_ids = ""
    else:
        scene_id, instance_label, n_objects, target_id, distractor_ids = parts
    instance_label = instance_label.replace("_", " ")
    n_objects = int(n_objects)
    target_id = int(target_id)
    distractor_ids = [int(i) for i in distractor_ids.split("-") if i]
    assert len(distractor_ids) == n_objects - 1
    return scene_id, instance_label, n_objects, target_id, distractor_ids

# [수정] 비정상적인 JSON을 올바르게 파싱하도록 함수 전체 재작성
def load_ref_data(anno_file, scan_id_file):
    split_scan_ids = set(x.strip() for x in open(scan_id_file, "r"))
    ref_data = []
    with open(anno_file, 'r') as f:
        for line in f:
            # 각 줄은 {"key":"value"} 형태의 JSON으로 되어 있음
            data = json.loads(line)
            # key와 value를 추출
            keys_str, values_str = list(data.items())[0]
            
            # key 문자열을 파싱하여 헤더 리스트 생성
            headers = keys_str.split(',')
            
            # value 문자열을 파싱 (CSV 리더 사용)
            # 따옴표로 묶인 콤마를 처리하기 위해 csv 모듈 활용
            values = next(csv.reader([values_str]))
            
            # 헤더와 값을 묶어 dictionary 생성
            item = dict(zip(headers, values))
            
            # 이제 올바른 item 객체를 사용하여 필터링
            if item["scan_id"] in split_scan_ids:
                # 숫자형 데이터는 타입 변환
                item['target_id'] = int(item['target_id'])
                item['tokens'] = eval(item['tokens']) # 문자열을 리스트로 변환
                ref_data.append(item)

    print(f"Loaded {len(ref_data)} references for the given scan_ids.")
    return ref_data

def process_reference_item(ref, program_prompt, client, args):
    caption = ref["utterance"]
    print("-" * 20)
    print(caption)
    print(ref["scan_id"])

    hardness = decode_stimulus_string(ref["stimulus_id"])[2]
    easy_context_mask = hardness <= 2
    view_dep_mask = is_explicitly_view_dependent(ref["tokens"])
    input_prompt = f"Query: {caption}"

    messages = [
        {"role": "system", "content": program_prompt},
        {"role": "user", "content": [{"type": "text", "text": input_prompt}]},
    ]
    chat_response = client.chat.completions.create(
        model=args.model_name, messages=messages
    )
    answer = chat_response.choices[0].message.content

    try:
        answer = answer.replace("'", '"')
        answer = json.loads(answer)
        print(answer)
    except:
        print(answer)
        print("!!! Warning, Error in answer")
        answer = {}

    return {
        "scan_id": ref["scan_id"],
        "target_id": ref["target_id"],
        "caption": ref["utterance"],
        "parsed_query": answer,
        "easy": easy_context_mask,
        "view_dep": view_dep_mask,
    }

def save_processed_data(new_data, save_dir, scan_id):
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/{scan_id}.json", "w") as f:
        json.dump(new_data, f, indent=4)
    print(f"Saved scan {scan_id} data to {save_dir}/{scan_id}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process rooms for object detection.")
    parser.add_argument("--openai_api_key", type=str, default="your_openai_api_key")
    parser.add_argument("--save_dir", type=str, default="data/nr3d/query")
    parser.add_argument("--openai_api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model_name", type=str, default="Qwen2-VL-72B-Instruct")
    parser.add_argument("--anno_file", type=str, default="data/Nr3D/nr3d.jsonl")
    parser.add_argument("--scan_id_file", type=str, default="data/scannet/scannetv2_val.txt")
    parser.add_argument("--prompt_file", type=str, default="prompts/parsing_query.txt")
    args = parser.parse_args()

    ref_data = load_ref_data(args.anno_file, args.scan_id_file)
    with open(args.prompt_file, "r") as f:
        program_prompt = f.read()

    grouped_data = defaultdict(list)
    for ref in ref_data:
        grouped_data[ref["scan_id"]].append(ref)

    sorted_scan_ids = sorted(grouped_data.keys())

    for scan_id in sorted_scan_ids:
        if os.path.exists(f"{args.save_dir}/{scan_id}.json"):
            print(f"Skipping {scan_id}. Already exists.")
            continue

        entries = grouped_data[scan_id]
        new_data = []
        for ref in tqdm(entries):
            new_data.append(
                process_reference_item(
                    ref,
                    program_prompt,
                    OpenAI(api_key=args.openai_api_key, base_url=args.openai_api_base),
                    args,
                )
            )
        save_processed_data(new_data, args.save_dir, scan_id)