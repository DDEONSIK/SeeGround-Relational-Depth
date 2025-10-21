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
import cv2
import matplotlib.pyplot as plt

# ==============================================================================
# Depth Anything V2 모듈 임포트
# ==============================================================================
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Depth-Anything-V2"))

import torch
from PIL import Image
from depth_anything_v2.dpt import DepthAnythingV2

# 실제 존재하는 클래스들을 정확히 임포트
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
# ==============================================================================

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from inference.DepthV2_projection import render_point_cloud_with_pytorch3d_with_objects
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

# --- 기존 상수 선언 ---
SYSTEM_INFO = "You are a helpful assistant designed to identify objects based on image and descriptions."
COOR_INFO = "The 3D spatial coordinate system is defined as follows: X-axis and Y-axis represent horizontal dimensions, Z-axis represents the vertical dimension."
ASK_INFO = "Please review the provided image and object 3D spatial descriptions, then select the object ID that best matches the given description. "
RESPONSE_FORMAT = "Respond in the format: 'Predicted ID: <ID>\\nExplanation: <explanation>', where <ID> is the object ID and <explanation> is your reasoning."

# ==============================================================================
# Depth Anything V2의 이미지 변환 파이프라인
# ==============================================================================
class DepthAnythingV2Transform:
    def __init__(self, image_shape=(518, 518)):
        self.transform = [
            Resize(
                width=image_shape[1],
                height=image_shape[0],
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            PrepareForNet(),
        ]

    def __call__(self, sample):
        for transform in self.transform:
            sample = transform(sample)
        return sample
# ==============================================================================


# --- Depth Anything V2 모델 로더 및 추론 함수 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEPTH_MODEL = None
DEPTH_TRANSFORM = None

def load_depth_model(encoder='vitl'):
    global DEPTH_MODEL, DEPTH_TRANSFORM
    if DEPTH_MODEL is None:
        print("Loading Depth Anything V2 model for the first time...")
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        checkpoint_path = '/workspace/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth'
        
        DEPTH_MODEL = DepthAnythingV2(**model_configs[encoder])
        DEPTH_MODEL.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        DEPTH_MODEL.eval().to(DEVICE)
        
        DEPTH_TRANSFORM = DepthAnythingV2Transform()
        print("Depth Anything V2 model loaded successfully.")


def get_depth_map(image_input):
    if DEPTH_MODEL is None or DEPTH_TRANSFORM is None:
        raise RuntimeError("Depth model is not loaded. Call load_depth_model() first.")
    
    try:
        if isinstance(image_input, str):
            raw_image = np.array(Image.open(image_input).convert("RGB"))
        else:
            raw_image = image_input
            
        image = DEPTH_TRANSFORM({'image': raw_image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            depth = DEPTH_MODEL(image)
        
        return depth.squeeze().cpu().numpy()
    except Exception as e:
        error_source = image_input if isinstance(image_input, str) else "the provided image data"
        print(f"Error processing depth for {error_source}: {e}")
        return None


def save_depth_map_visualization(depth_map, save_path):
    """
    [최종 수정] np.inf 값을 검은색 배경으로 처리하여 시각화.
    """
    if depth_map is None:
        return

    # 컬러맵 복사 및 배경색 설정
    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad(color='black') # 값이 유효하지 않은(NaN, inf) 픽셀을 검은색으로 설정

    # 배경(inf) 픽셀을 제외한 유효한 깊이 값의 최소/최대값 계산
    finite_depths = depth_map[np.isfinite(depth_map)]
    if len(finite_depths) == 0:
        vmin, vmax = 0, 1 # 모든 픽셀이 배경인 경우
    else:
        vmin, vmax = finite_depths.min(), finite_depths.max()

    # 배경(inf) 픽셀을 마스킹 처리하여 컬러맵에 적용
    masked_map = np.ma.masked_invalid(depth_map)
    
    plt.imsave(save_path, masked_map, cmap=cmap, vmin=vmin, vmax=vmax)
    # 시각화 저장은 콘솔에 너무 많은 로그를 남기므로 print 문은 주석 처리
    print(f"Depth map visualization saved to {save_path}")


def generate_depth_text_from_coords(marker_coords_2d, depth_map, targets, anchors, query, image_size=680, easy=False, view_dep=False):
    """
    [K-NN 적용] KNN Relational Text 추가 (k=3 nearest). Nr3D: easy/view_dep 차별화. 키워드 확장.
    """
    if depth_map is None or not marker_coords_2d:
        return "N/A"

    # 1. 패치 샘플링
    depth_info = []
    patch_radius = 2
    h, w = depth_map.shape
    for obj_id_str, (cx, cy) in marker_coords_2d.items():
        cx, cy = int(cx), int(cy)
        x_start, x_end = max(0, cx - patch_radius), min(w, cx + patch_radius + 1)
        y_start, y_end = max(0, cy - patch_radius), min(h, cy + patch_radius + 1)
        
        if x_start >= x_end or y_start >= y_end: continue

        depth_patch = depth_map[y_start:y_end, x_start:x_end]
        valid_depths = depth_patch[~np.isinf(depth_patch)]
        
        if valid_depths.size > 0:
            min_depth = np.min(valid_depths)
            depth_info.append({"id": int(obj_id_str), "depth": min_depth, "coords": (cx, cy)})

    if not depth_info: return "No valid depth detected for any object."
    depth_info.sort(key=lambda p: p["depth"], reverse=True)
    
    # depth 정규화
    depths = [d['depth'] for d in depth_info]
    min_d, max_d = min(depths), max(depths)
    def depth_category(depth):
        norm = (depth - min_d) / (max_d - min_d) if max_d > min_d else 0.5
        if norm < 0.33: return "shallow"
        elif norm < 0.67: return "medium"
        else: return "deep"
    
    # 2. 객체 추출
    target_ids = {t['bbox_id'] for t in targets}
    anchor_ids = {a['bbox_id'] for a in anchors}
    main_target = next((d for d in depth_info if d['id'] in target_ids), None)
    main_anchor = next((d for d in depth_info if d['id'] in anchor_ids), None)

    # 3. KNN Relational
    def get_knn(target, k=3):
        if not target: return []
        dists = [(abs(target['depth'] - o['depth']), o['id']) for o in depth_info if o['id'] != target['id']]
        dists.sort()
        return [id for _, id in dists[:k]]

    # 4. 텍스트 생성
    generated_texts = []
    query_lower = query.lower()
    depth_keywords = ['close', 'far', 'front', 'behind', 'near', 'further', 'closer', 'nearest', 'farthest', 'next to', 'on', 'under', 'beside', 'facing']
    spatial_keywords = ['left', 'right', 'above', 'below', 'top', 'bottom', 'side', 'between', 'middle', 'edge', 'on the', 'to the', 'opposite']

    if main_target:
        target_cat = depth_category(main_target['depth'])
        generated_texts.append(f"Target (ID {main_target['id']}) at {target_cat} depth.")
        
        knn_ids = get_knn(main_target)
        if knn_ids and view_dep:  # view_dep (복잡) 시 KNN 추가
            generated_texts.append(f"Target near objects {', '.join(map(str, knn_ids))}.") 
        
        if main_anchor:
            depth_diff = abs(main_target['depth'] - main_anchor['depth'])
            unit_str = f"{depth_diff:.2f} units"
            anchor_cat = depth_category(main_anchor['depth'])

            if any(keyword in query_lower for keyword in depth_keywords):
                comp = "closer" if main_target['depth'] > main_anchor['depth'] else "farther" # Raw Depth 값이 더 큰 쪽이 더 가까우므로, 부등호 방향을 '<'에서 '>'로 변경.
                generated_texts.append(f"Target {comp} than anchor (ID {main_anchor['id']}) by {unit_str} ({target_cat} vs {anchor_cat}).")

            if any(keyword in query_lower for keyword in spatial_keywords):
                tx, ty = main_target['coords']
                ax, ay = main_anchor['coords']
                if abs(tx - ax) > abs(ty - ay) * 1.5:
                    generated_texts.append(f"Target to the {'left' if tx < ax else 'right'} of anchor.")
                elif abs(ty - ay) > abs(tx - ax) * 1.5:
                    generated_texts.append(f"Target {'above' if ty < ay else 'below'} anchor.")
                if 'between' in query_lower or 'middle' in query_lower:
                    avg_x = (tx + ax) / 2
                    generated_texts.append(f"Target in middle, x ~{avg_x:.2f}.")

        else:
            if depth_info:
                avg_depth = np.mean(depths)
                depth_diff = abs(main_target['depth'] - avg_depth)
                unit_str = f"{depth_diff:.2f} units"
                avg_cat = depth_category(avg_depth)

                if any(keyword in query_lower for keyword in depth_keywords):
                    comp = "closer" if main_target['depth'] > avg_depth else "farther" # Raw Depth 값이 더 큰 쪽이 더 가까우므로, 부등호 방향을 '<'에서 '>'로 변경.
                    generated_texts.append(f"Target {comp} than average by {unit_str} ({target_cat} vs {avg_cat}).")

                if any(keyword in query_lower for keyword in spatial_keywords):
                    tx, ty = main_target['coords']
                    side = "left" if tx < image_size / 2 else "right"
                    generated_texts.append(f"Target on {side} side.")

    # 5. easy/view_dep 차별 + 요약
    if generated_texts:
        text = " ".join(generated_texts)
        return text[:200] if easy else text  # easy: 짧게, view_dep: 상세
    else:
        closest = depth_info[0]
        farthest = depth_info[-1]
        if len(depth_info) > 1:
            return f"Closest ID {closest['id']} ({depth_category(closest['depth'])}), farthest ID {farthest['id']} ({depth_category(farthest['depth'])})."
        return f"Object {closest['id']} at {depth_category(closest['depth'])} depth."
    
    return "N/A"
# ==============================================================================

def create_openai_messages(query, objects_info, use_image=False, image_path=None, depth_text=None):
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
        prompt_text = (
            f"As shown in the image, this is a rendered image of a room, and the picture reflects your current view. "
            f"Each object in the room is labeled by a unique number (ID) in red color on its surface. \n\n"
            f"Object IDs and their 3D spatial information are as follows:\n{objects_info}\n\n"
            f"{COOR_INFO}\n\n"
        )
        if depth_text:
            prompt_text += f"Critical spatial hints from depth:\n{depth_text}\nUse these hints to resolve ambiguities.\n\n"
        prompt_text += f"{ASK_INFO}\n\n{RESPONSE_FORMAT}\n\nThe given description is: {query}"
        messages[1]["content"][0] = {"type": "text", "text": prompt_text}
        messages[1]["content"].insert(0, {"type": "image_url", "image_url": {"url": img_url}})
    return messages

def process_query(
    query, objects_info, openai_api_key, openai_api_base, use_image=False, 
    image_path=None, model_name="Qwen2-VL-7B-Instruct", log_file=None, depth_text=None
):
    assert objects_info is not None and query is not None
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    messages = create_openai_messages(query, objects_info, use_image, image_path, depth_text)
    if log_file and not os.path.exists(log_file):
        save_to_file(log_file, str(messages))
    chat_response = client.chat.completions.create(model=model_name, messages=messages)
    result = chat_response.choices[0].message.content
    return result.replace("\\n", "\n")

def process_room(
    dataset, room, pcd_dir, split, output_dir, language_annotation_file, 
    gt_bbox_dir, pred_bbox_dir, openai_api_key, openai_api_base, 
    use_image=False, use_depth=False, model_name=None, verbose=True
):
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
    
    correct_predictions, total_predictions, results = 0, 0, []
    queries = sorted(queries, key=lambda x: int(x["target_id"]))

    if use_depth:
        load_depth_model()

    for i, d in enumerate(tqdm(queries)):
        total_predictions += 1
        query = d["caption"]
        gt_id = int(d["target_id"])
        image_path, depth_text, marker_coords_2d = None, None, {}
        
        try:
            target_name, anchor_name = d["parsed_query"]["Target"], d["parsed_query"]["Anchor"]
        except:
            target_name, anchor_name = "", ""
        
        matched_targets = fuzzy_match(target_name, object_names).union(stem_match(target_name, object_names))
        matched_anchors = fuzzy_match(anchor_name, object_names).union(stem_match(anchor_name, object_names))
        
        targets = [obj for obj in mask3d_bboxes.values() if obj["target"] in matched_targets] or list(mask3d_bboxes.values())
        anchors = [obj for obj in mask3d_bboxes.values() if obj["target"] in matched_anchors] or targets  # Nr3D 특성

        if use_image:
            scan_pc, center = load_scene_pcd(room, pcd_dir)
            
            base_save_dir = f"outputs/projection_img/{dataset}/{room}/{i}"

            # 1. Depth Map 생성을 위한 마커 없는 이미지(I) 렌더링
            raw_save_dir = os.path.join(base_save_dir, "raw")
            raw_image_path, _, alpha_mask = render_point_cloud_with_pytorch3d_with_objects(
                mask3d_bboxes.values(), targets, anchors, center, scan_pc,
                save_dir=raw_save_dir,
                image_size=680, draw_id=False, draw_img=True, return_marker_coords=True
            )

            # 2. VLM 입력을 위한 마커 있는 이미지(Im) 렌더링
            prompted_save_dir = os.path.join(base_save_dir, "prompted")
            image_path, marker_coords_2d, _ = render_point_cloud_with_pytorch3d_with_objects(
                mask3d_bboxes.values(), targets, anchors, center, scan_pc,
                save_dir=prompted_save_dir,
                image_size=680, draw_id=True, draw_img=True, return_marker_coords=True
            )

            if use_depth and raw_image_path and alpha_mask is not None:
                print("\nGenerating and refining depth map...")
                
                # 마커 없는 raw_image_path를 입력으로 사용
                dirty_depth_map = get_depth_map(raw_image_path)
                
                clean_depth_map = None
                if dirty_depth_map is not None:
                    target_size = (alpha_mask.shape[1], alpha_mask.shape[0]) 
                    resized_dirty_depth_map = cv2.resize(dirty_depth_map, target_size, interpolation=cv2.INTER_NEAREST)

                    clean_depth_map = resized_dirty_depth_map.copy()
                    background_pixels = alpha_mask == 0
                    clean_depth_map[background_pixels] = np.inf
                    
                depth_map_save_dir = os.path.dirname(image_path)
                depth_map_filename = f"DepthMap_{os.path.basename(image_path)}"
                depth_map_save_path = os.path.join(depth_map_save_dir, depth_map_filename)
                
                save_depth_map_visualization(clean_depth_map, depth_map_save_path)
                
                depth_text = generate_depth_text_from_coords(marker_coords_2d, clean_depth_map, targets, anchors, query, image_size=680, easy=d["easy"], view_dep=d["view_dep"])
                print(f"Generated Depth Text: {depth_text}")

        response = process_query(
            query, objects_info, openai_api_key, openai_api_base, use_image, 
            image_path, model_name, depth_text=depth_text
        )
        parsed_data = parse_response(response)
        if parsed_data:
            predicted_id, explanation = parsed_data
        else:
            predicted_id, explanation = None, "Failed to parse response"
            
        print(f"GT id is {gt_id}; Pred id is {predicted_id}")

        gt_bbox = gt_bboxes.get(gt_id)
        pred_bbox = mask3d_bboxes.get(predicted_id)
        
        if predicted_id is not None and int(predicted_id) == gt_id:
            correct_predictions += 1
            
        results.append({
            "query": query, "gt_id": gt_id, "predicted_id": predicted_id,
            "pred_bbox": pred_bbox["bbox_3d"] if pred_bbox else None, 
            "gt_bbox": gt_bbox["bbox_3d"] if gt_bbox else None,
            "image_path": image_path, "parsed_query": d["parsed_query"], "explanation": explanation,
            "easy": d["easy"], "view_dep": d["view_dep"], "depth_text": depth_text
        })
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Final Accuracy for room {room}: {accuracy:.4f}")
    save_to_file(output_file, json.dumps(results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nr3d", help="Dataset name")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--output_dir", default="outputs/DepthV2_nr3d/val/", help="Directory to store the output")
    parser.add_argument("--language_annotation_dir", default="data/nr3d/query", help="Parsed language annotation file path")
    parser.add_argument("--gt_bbox_dir", default="data/seeground_object_lookup_table/nr3d/gt", help="Ground truth bounding box directory")
    parser.add_argument("--pred_bbox_dir", default="data/seeground_object_lookup_table/nr3d/pred", help="Predicted bounding box directory")
    parser.add_argument("--pcd_dir", default='referit3d/scan_data/pcd_with_global_alignment/', help="")
    parser.add_argument("--openai_api_key", default="your_openai_api_key", help="OpenAI API Key")
    parser.add_argument("--openai_api_base", default="http://localhost:8000/v1", help="OpenAI API Base URL")
    parser.add_argument("--use_image", action='store_true', default=True, help="Whether to use image rendering")
    parser.add_argument("--use_depth", action='store_true', help="Enable depth information processing.")
    parser.add_argument("--model_name", default="Qwen2-VL-7B-Instruct", help="Model name")
    parser.add_argument("--val_file", type=str, default="data/scannet/scannetv2_val.txt", help="Path to the validation split file.")
    args = parser.parse_args()

    scan_ids = sorted([i.split(".")[0] for i in os.listdir(args.language_annotation_dir)])
    print(f"Found {len(scan_ids)} scans in {args.language_annotation_dir}")

    for room in tqdm(scan_ids, desc="Process rooms"):
    # for room in tqdm(scan_ids[:5], desc="Process rooms (TEST RUN on 5 rooms)"): # 테스트용으로 5개 룸만 처리
        
        language_annotation_file = os.path.join(args.language_annotation_dir, f"{room}.json")
        process_room(
            dataset=args.dataset, room=room, split=args.split, output_dir=args.output_dir,
            pcd_dir=args.pcd_dir, language_annotation_file=language_annotation_file,
            gt_bbox_dir=args.gt_bbox_dir, pred_bbox_dir=args.pred_bbox_dir,
            openai_api_key=args.openai_api_key, openai_api_base=args.openai_api_base,
            use_image=args.use_image, use_depth=args.use_depth, model_name=args.model_name
        )

