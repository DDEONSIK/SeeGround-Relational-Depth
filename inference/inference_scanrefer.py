import sys
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import cv2
import re

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from inference.projection import render_point_cloud_with_pytorch3d_with_objects
from inference.utils import (
    calc_iou,
    encode_img,
    save_to_file,
    fuzzy_match,
    stem_match,
    load_json,
    load_bboxes,
    load_scene_pcd,
)

def generate_objects_info_concise(objects):
    """
    [최종 수정] 객체 정보를 ID, Label, 그리고 '중심 좌표'만 포함하도록 수정합니다.
    크기 정보는 제외하여 프롬프트 길이를 최적화합니다.
    """
    objects_info = []
    for obj_id, obj_data in objects.items():
        bbox_3d = obj_data['bbox_3d']
        center_coords = bbox_3d[:3]
        center_info = ", ".join([f"{coord:.2f}" for coord in center_coords])
        
        info_str = (
            f"ID: {obj_id}, "
            f"Label: {obj_data['target']}, "
            f"Center: ({center_info})"
        )
        objects_info.append(info_str)
    return "\n".join(objects_info)

SYSTEM_INFO = (
    "You are an expert system. Your task is to identify a single object ID from a provided list based on a description. "
    "You MUST respond ONLY in the following JSON format. Do NOT add any other text before or after the JSON."
)
RESPONSE_FORMAT = '{"id": <the_object_id>, "reason": "<your_brief_explanation>"}'


def parse_json_response(response: str):
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            predicted_id = int(data.get("id"))
            explanation = data.get("reason", "No explanation provided.")
            return predicted_id, explanation
    except (json.JSONDecodeError, TypeError, AttributeError):
        match = re.search(r'\d+', response)
        if match:
            return int(match.group(0)), f"Explanation extracted via fallback: {response}"
    return None, response


def create_openai_messages(query, objects_info, use_image=False, image_path=None):
    user_prompt = (
        f"Object list with 3D center coordinates:\n{objects_info}\n\n"
        f"User's description: '{query}'\n\n"
        f"Identify the object that best matches the user's description and respond ONLY in this JSON format: {RESPONSE_FORMAT}"
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_INFO},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]

    if use_image and image_path:
        img_url = encode_img(image_path)
        messages[1]["content"].insert(0, {"type": "image_url", "image_url": {"url": img_url}})

    return messages

def process_query(
    query, objects_info, openai_api_key, openai_api_base, use_image=False,
    image_path=None, model_name="Qwen2-VL-72B-Instruct", log_file=None,
):
    assert objects_info is not None and query is not None
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    messages = create_openai_messages(query, objects_info, use_image, image_path)
    if log_file and not os.path.exists(log_file):
        save_to_file(log_file, str(messages))
    chat_response = client.chat.completions.create(model=model_name, messages=messages)
    return chat_response.choices[0].message.content.replace("\\n", "\n")


def process_room(
    dataset, room, pcd_dir, split, output_dir, language_annotation_file,
    gt_bbox_dir, pred_bbox_dir, openai_api_key, openai_api_base,
    use_image=False, model_name=None, verbose=True,
):
    data = load_json(language_annotation_file)
    queries = [it for it in data if it.get("scan_id") == room]
    if not queries:
        print(f"No queries found for room {room} in {language_annotation_file}")
        return

    gt_bboxes = load_bboxes(room, gt_bbox_dir, "gt")
    mask3d_bboxes = load_bboxes(room, pred_bbox_dir, "pred")
    object_names = [obj["target"] for obj in mask3d_bboxes.values()]
    objects_info = generate_objects_info_concise(mask3d_bboxes)

    output_file = os.path.join(output_dir, "pred", f"{room}.json")
    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping")
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    log_file = os.path.join(output_dir, "room_info", f"{room}.txt")
    print(f"Saved objects_info to {log_file}")

    correct_predictions = 0
    total_predictions = 0
    results = []
    queries = sorted(queries, key=lambda x: int(x["target_id"]))

    for i, d in enumerate(tqdm(queries)):
        total_predictions += 1
        query = d["caption"]
        gt_id = int(d["target_id"])
        image_path = None
        print(f"\nQuery: {query}")

        try:
            target_name = d["parsed_query"]["Target"]
            anchor_name = d["parsed_query"]["Anchor"]
        except (KeyError, TypeError):
            target_name, anchor_name = "", ""
        print(f"Parsed target: {target_name}; anchor: {anchor_name}")

        matched_targets = fuzzy_match(target_name, object_names).union(
            stem_match(target_name, object_names)
        )
        matched_anchors = fuzzy_match(anchor_name, object_names).union(
            stem_match(anchor_name, object_names)
        )
        print(f"Matched target: {matched_targets}; anchor: {matched_anchors}")

        targets = [obj for obj in mask3d_bboxes.values() if obj["target"] in matched_targets]
        anchors = [obj for obj in mask3d_bboxes.values() if obj["target"] in matched_anchors]
        if not targets: targets = list(mask3d_bboxes.values())
        if not anchors: anchors = targets

        if use_image:
            scan_pc, center = load_scene_pcd(room, pcd_dir)
            image_path = render_point_cloud_with_pytorch3d_with_objects(
                mask3d_bboxes.values(),
                targets,
                anchors,
                center,
                scan_pc,
                save_dir=f"outputs/projection_img/{dataset}/{room}/{i}",
                image_size=680,
                draw_id=True,
                draw_img=True,
            )

        response = process_query(
            query, objects_info, openai_api_key, openai_api_base,
            use_image, image_path, model_name, log_file
        )
        print(f"VLM Raw Response: {response}")
        
        predicted_id, explanation = parse_json_response(response)
        print(f"GT id is {gt_id}; Pred id is {predicted_id}")

        gt_bbox = gt_bboxes[gt_id]
        try:
            pred_bbox = mask3d_bboxes[predicted_id]
            iou = calc_iou(gt_bbox["bbox_3d"], pred_bbox["bbox_3d"])
        except:
            pred_bbox, iou = None, 0
        print("iou is ", iou)

        if iou > 0.25:
            correct_predictions += 1

        results.append({
            "query": query, "gt_id": gt_id, "predicted_id": predicted_id,
            "pred_bbox": pred_bbox["bbox_3d"] if pred_bbox else None,
            "gt_bbox": gt_bbox["bbox_3d"], "image_path": image_path,
            "parsed_query": d.get("parsed_query"),
            "explanation": explanation,
            "unique": d["unique"],
        })
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Accuracy: {accuracy:.4f}")

    log_file = os.path.join(output_dir, "room_acc", f"{room}_acc.txt")
    save_to_file(log_file, f"Accuracy after {total_predictions} predictions: {accuracy * 100:.2f}%")
    save_to_file(output_file, json.dumps(results, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scanrefer", help="Dataset name")
    parser.add_argument("--split", default="val", help="Dataset split")
    parser.add_argument("--output_dir", default="outputs/scanrefer/val", help="Directory to store the output")
    parser.add_argument("--language_annotation_dir", default="data/scanrefer/query/", help="Parsed language annotation file path")
    parser.add_argument("--gt_bbox_dir", default="data/seeground_object_lookup_table/scanrefer/gt", help="Ground truth bounding box directory")
    parser.add_argument("--pred_bbox_dir", default="data/seeground_object_lookup_table/scanrefer/pred", help="Predicted bounding box directory")
    parser.add_argument("--pcd_dir", default='referit3d/scan_data/pcd_with_global_alignment/', help="")
    parser.add_argument("--openai_api_key", default="your_openai_api_key", help="OpenAI API Key")
    parser.add_argument("--openai_api_base", default="http://localhost:8000/v1", help="OpenAI API Base URL")
    parser.add_argument("--use_image", default=True, help="Whether to use image rendering")
    parser.add_argument("--model_name", default="Qwen2-VL-72B-Instruct", help="Model name")
    args = parser.parse_args()

    scan_ids = sorted([f.split('.')[0] for f in os.listdir(args.language_annotation_dir) if f.endswith('.json')])
    print(f"Found {len(scan_ids)} scans in {args.language_annotation_dir}")

    for room in tqdm(scan_ids, desc="Process rooms"):
        language_annotation_file = os.path.join(args.language_annotation_dir, f"{room}.json")
        process_room(
            dataset=args.dataset,
            room=room,
            split=args.split,
            output_dir=args.output_dir,
            pcd_dir=args.pcd_dir,
            language_annotation_file=language_annotation_file,
            gt_bbox_dir=args.gt_bbox_dir,
            pred_bbox_dir=args.pred_bbox_dir,
            openai_api_key=args.openai_api_key,
            openai_api_base=args.openai_api_base,
            use_image=args.use_image,
            model_name=args.model_name,
        )