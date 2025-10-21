import sys
import argparse
import base64
import json
import os
import random
import numpy as np
import open3d as o3d
from tqdm import tqdm
from openai import OpenAI
import cv2 # cv2 import 추가

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from inference.projection import render_point_cloud_with_pytorch3d_with_objects
from inference.utils import (
    parse_response,
    calc_iou,
    encode_img,
    read_file_to_list,
    save_to_file,
    stem_match,
    fuzzy_match,
    load_json,
    load_bboxes,
    generate_objects_info,
    load_scene_pcd,
)

SYSTEM_INFO = "You are a helpful assistant designed to identify objects based on image and descriptions."
COOR_INFO = "The 3D spatial coordinate system is defined as follows: X-axis and Y-axis represent horizontal dimensions, Z-axis represents the vertical dimension."
ASK_INFO = "Please review the provided image and object 3D spatial descriptions, then select the object ID that best matches the given description. "
RESPONSE_FORMAT = "Respond in the format: 'Predicted ID: <ID>\\nExplanation: <explanation>', where <ID> is the object ID and <explanation> is your reasoning."


def create_openai_messages(query, objects_info, use_image=False, image_path=None):
    """Create OpenAI API messages."""
    messages = [
        {"role": "system", "content": SYSTEM_INFO},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{COOR_INFO}\\n\\nObject IDs and their positions:\\n{objects_info}\\n\\n{ASK_INFO}\\n\\n{RESPONSE_FORMAT}\\n\\nThe given description is: {query}",
                }
            ],
        },
    ]

    if use_image and image_path:
        img_url = encode_img(image_path)
        messages[1]["content"][0] = {
            "type": "text",
            "text": f"As shown in the image, this is a rendered image of a room, and the picture reflects your current view. Each object in the room is labeled by a unique number (ID) in red color on its surface. \\n\\nObject IDs and their 3D spatial information are as follows:\\n{objects_info}\\n\\n{COOR_INFO}\\n\\n{ASK_INFO}\\n\\n{RESPONSE_FORMAT}\\n\\nThe given description is: {query}",
        }
        messages[1]["content"].insert(
            0, {"type": "image_url", "image_url": {"url": img_url}}
        )

    return messages


def process_query(
    query,
    objects_info,
    openai_api_key,
    openai_api_base,
    use_image=False,
    image_path=None,
    model_name="Qwen2-VL-72B-Instruct",
    log_file=None,
):
    """Process query and return model's response."""
    assert objects_info is not None
    assert query is not None

    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    messages = create_openai_messages(query, objects_info, use_image, image_path)

    if log_file and not os.path.exists(log_file):
        save_to_file(log_file, str(messages))

    chat_response = client.chat.completions.create(model=model_name, messages=messages)
    result = chat_response.choices[0].message.content
    return result.replace("\\n", "\n")


def process_room(
    dataset,
    room,
    pcd_dir,
    split,
    output_dir,
    language_annotation_file,
    gt_bbox_dir,
    pred_bbox_dir,
    openai_api_key,
    openai_api_base,
    use_image=False,
    model_name=None,
    verbose=True,
):
    """Process a single room with queries and predictions."""
    data = load_json(language_annotation_file)
    queries = [it for it in data if it["scan_id"] == room]
    gt_bboxes = load_bboxes(room, gt_bbox_dir, "gt")
    mask3d_bboxes = load_bboxes(room, pred_bbox_dir, "pred")
    object_names = [obj["target"] for obj in mask3d_bboxes.values()]

    objects_info = generate_objects_info(mask3d_bboxes.values())

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
        print()
        print(f"Query: {query}")

        try:
            target_name = d["parsed_query"]["Target"]
            anchor_name = d["parsed_query"]["Anchor"]
        except:
            target_name = ""
            anchor_name = ""
        print(f"Parsed target: {target_name}; anchor: {anchor_name}")
        matched_targets = fuzzy_match(target_name, object_names).union(
            stem_match(target_name, object_names)
        )
        matched_anchors = fuzzy_match(anchor_name, object_names).union(
            stem_match(anchor_name, object_names)
        )
        print(f"Matched target: {matched_targets}; anchor: {matched_anchors}")

        targets = [
            obj for obj in mask3d_bboxes.values() if obj["target"] in matched_targets
        ]
        anchors = [
            obj for obj in mask3d_bboxes.values() if obj["target"] in matched_anchors
        ]

        if len(targets) == 0:
            targets = list(mask3d_bboxes.values())
        if len(anchors) == 0:
            anchors = targets

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
            query,
            objects_info,
            openai_api_key,
            openai_api_base,
            use_image,
            image_path,
            model_name,
            log_file,
        )
        predicted_id, explanation = parse_response(response)
        print(f"GT id is {gt_id}; Pred id is {predicted_id}")

        gt_bbox = gt_bboxes[gt_id]
        try:
            pred_bbox = mask3d_bboxes[predicted_id]
        except:
            pred_bbox = None

        if predicted_id == gt_id:
            correct_predictions += 1

        results.append(
            {
                "query": query,
                "gt_id": gt_id,
                "predicted_id": predicted_id,
                "pred_bbox": pred_bbox["bbox_3d"] if pred_bbox else None,
                "gt_bbox": gt_bbox["bbox_3d"],
                "image_path": image_path,
                "parsed_query": d["parsed_query"],
                "explanation": explanation,
                "easy": d["easy"],
                "view_dep": d["view_dep"],
            }
        )

        accuracy = correct_predictions / total_predictions
        print(f"Accuracy: {accuracy:.4f}")

    log_file = os.path.join(output_dir, "room_acc", f"{room}_acc.txt")
    save_to_file(
        log_file,
        f"Accuracy after {total_predictions} predictions: {accuracy * 100:.2f}%",
    )
    save_to_file(output_file, json.dumps(results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [수정] 모든 하드코딩된 경로를 현재 환경에 맞는 상대 경로로 변경
    parser.add_argument("--dataset", default="nr3d", help="Dataset name")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--output_dir", default="outputs/nr3d/val/", help="Directory to store the output")
    parser.add_argument("--language_annotation_dir", default="data/nr3d/query", help="Parsed language annotation file path")
    parser.add_argument("--gt_bbox_dir", default="data/seeground_object_lookup_table/nr3d/gt", help="Ground truth bounding box directory")
    parser.add_argument("--pred_bbox_dir", default="data/seeground_object_lookup_table/nr3d/pred", help="Predicted bounding box directory")
    parser.add_argument("--pcd_dir", default='referit3d/scan_data/pcd_with_global_alignment/', help="")
    parser.add_argument("--openai_api_key", default="your_openai_api_key", help="OpenAI API Key")
    parser.add_argument("--openai_api_base", default="http://localhost:8000/v1", help="OpenAI API Base URL")
    parser.add_argument("--use_image", default=True, help="Whether to use image rendering")
    parser.add_argument("--model_name", default="Qwen2-VL-72B-Instruct", help="Model name")
    parser.add_argument("--val_file", type=str, default="data/scannet/scannetv2_val.txt", help="Path to the validation split file.")
    args = parser.parse_args()

    scan_ids = list(os.listdir(args.language_annotation_dir))
    scan_ids = sorted([i.split(".")[0] for i in scan_ids])
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