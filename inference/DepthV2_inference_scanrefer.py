import sys
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import cv2
import re
import torch
from PIL import Image
import matplotlib.pyplot as plt

# --- 경로 설정 및 유틸리티 임포트 ---
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from inference.DepthV2_projection import render_point_cloud_with_pytorch3d_with_objects
from inference.utils import (
    calc_iou, encode_img, save_to_file, fuzzy_match, stem_match,
    load_json, load_bboxes, load_scene_pcd,
)
# ==============================================================================
# Depth Anything V2 모듈 임포트
# ==============================================================================
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
# ==============================================================================

# --- Depth Anything V2 관련 전역 변수 및 헬퍼 함수 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEPTH_MODEL = None
DEPTH_TRANSFORM = None

class DepthAnythingV2Transform:
    def __init__(self, image_shape=(518, 518)):
        self.transform = [
            Resize(
                width=image_shape[1], height=image_shape[0], resize_target=False,
                keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            PrepareForNet(),
        ]
    def __call__(self, sample):
        for transform_op in self.transform:
            sample = transform_op(sample)
        return sample

def load_depth_model(encoder='vitl'):
    global DEPTH_MODEL, DEPTH_TRANSFORM
    if DEPTH_MODEL is None:
        print("Loading Depth Anything V2 model for the first time...")
        model_configs = {'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}}
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


def generate_depth_text_from_coords(marker_coords_2d, depth_map, targets, anchors, query, image_size=680, unique=False):
    """
    [추가 개선] KNN Relational Text 추가 (k=3 nearest). Unique/Multiple 차별화. 키워드 확장.
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
    
    # 깊이 정규화
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

    # 3. KNN Relational (새로 추가: 타겟 중심 nearest k=3)
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
        if knn_ids and not unique:
            generated_texts.append(f"Target near objects {', '.join(map(str, knn_ids))}.")  # Multiple 시 KNN 추가
        
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

    # 5. Unique/Multiple 차별 + 요약
    if generated_texts:
        text = " ".join(generated_texts)
        return text[:200] if unique else text  # Unique: 짧게, Multiple: 상세 (과부하 방지)
    else:
        closest = depth_info[0]
        farthest = depth_info[-1]
        if len(depth_info) > 1:
            return f"Closest ID {closest['id']} ({depth_category(closest['depth'])}), farthest ID {farthest['id']} ({depth_category(farthest['depth'])})."
        return f"Object {closest['id']} at {depth_category(closest['depth'])} depth."
    
    return "N/A"
# ==============================================================================

def generate_objects_info_concise(objects):
    objects_info = []
    for obj_id, obj_data in objects.items():
        center_coords = obj_data['bbox_3d'][:3]
        center_info = ", ".join([f"{coord:.2f}" for coord in center_coords])
        info_str = f"ID: {obj_id}, Label: {obj_data['target']}, Center: ({center_info})"
        objects_info.append(info_str)
    return "\n".join(objects_info)

SYSTEM_INFO = "You are an expert system. Your task is to identify a single object ID from a provided list based on a description. You MUST respond ONLY in the following JSON format. Do NOT add any other text before or after the JSON."
RESPONSE_FORMAT = '{"id": <the_object_id>, "reason": "<your_brief_explanation>"}'

def parse_json_response(response: str):
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return int(data.get("id")), data.get("reason", "")
    except (json.JSONDecodeError, TypeError, AttributeError): pass
    match = re.search(r'\d+', response)
    return (int(match.group(0)), f"Fallback extraction: {response}") if match else (None, response)

def create_openai_messages(query, objects_info, use_image=False, image_path=None, depth_text=None):
    user_prompt = f"Object list with 3D center coordinates:\n{objects_info}\n\n"
    if use_image and image_path:
        user_prompt = (
            f"As shown in the image, each object is labeled by a unique ID in red.\n\n"
            f"Object list with 3D spatial information:\n{objects_info}\n\n"
        )
    if depth_text:
        user_prompt += f"Critical spatial hints from depth:\n{depth_text}\nUse these hints to resolve ambiguities in the description.\n\n"
    user_prompt += (
        f"User's description: '{query}'\n\n"
        f"Identify the object that best matches the user's description and respond ONLY in this JSON format: {RESPONSE_FORMAT}"
    )
    messages = [{"role": "system", "content": SYSTEM_INFO}, {"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
    if use_image and image_path:
        img_url = encode_img(image_path)
        messages[1]["content"].insert(0, {"type": "image_url", "image_url": {"url": img_url}})
    return messages

def process_query(query, objects_info, client, use_image=False, image_path=None, model_name="Qwen2-VL-7B-Instruct", depth_text=None):
    messages = create_openai_messages(query, objects_info, use_image, image_path, depth_text)
    chat_response = client.chat.completions.create(model=model_name, messages=messages)
    return chat_response.choices[0].message.content.replace("\\n", "\n")

def process_room(dataset, room, pcd_dir, output_dir, language_annotation_file, gt_bbox_dir, pred_bbox_dir, client, use_image=False, use_depth=False, model_name=None):
    data = load_json(language_annotation_file)
    queries = [it for it in data if it.get("scan_id") == room]
    if not queries: return

    gt_bboxes, mask3d_bboxes = load_bboxes(room, gt_bbox_dir, "gt"), load_bboxes(room, pred_bbox_dir, "pred")
    object_names, objects_info = [obj["target"] for obj in mask3d_bboxes.values()], generate_objects_info_concise(mask3d_bboxes)

    output_file = os.path.join(output_dir, "pred", f"{room}.json")
    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping")
        return
    
    if use_depth: load_depth_model()

    results = []
    for i, d in enumerate(tqdm(queries, desc=f"Processing {room}")):
        query, gt_id = d["caption"], int(d["target_id"])
        image_path, depth_text, marker_coords_2d = None, None, {}
        
        target_name = d["parsed_query"].get("Target", "")
        anchor_name = d["parsed_query"].get("Anchor", "")
        matched_targets = fuzzy_match(target_name, object_names).union(stem_match(target_name, object_names))
        matched_anchors = fuzzy_match(anchor_name, object_names).union(stem_match(anchor_name, object_names))
        
        targets = [obj for obj in mask3d_bboxes.values() if obj["target"] in matched_targets] or list(mask3d_bboxes.values())
        anchors = [obj for obj in mask3d_bboxes.values() if obj["target"] in matched_anchors]

        if use_image:
            scan_pc, center = load_scene_pcd(room, pcd_dir)

            base_save_dir = f"outputs/projection_img/{dataset}/{room}/{i}"

            # 1. Depth Map 생성을 위한 마커 없는 이미지(I) 렌더링
            raw_save_dir = os.path.join(base_save_dir, "raw")
            raw_image_path, _, alpha_mask = render_point_cloud_with_pytorch3d_with_objects(
                mask3d_bboxes.values(), targets, anchors, center, scan_pc,
                save_dir=raw_save_dir,
                draw_id=False, draw_img=True, return_marker_coords=True
            )

            # 2. VLM 입력을 위한 마커 있는 이미지(Im) 렌더링
            prompted_save_dir = os.path.join(base_save_dir, "prompted")
            image_path, marker_coords_2d, _ = render_point_cloud_with_pytorch3d_with_objects(
                mask3d_bboxes.values(), targets, anchors, center, scan_pc,
                save_dir=prompted_save_dir,
                draw_id=True, draw_img=True, return_marker_coords=True
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
                
                depth_map_path = os.path.join(os.path.dirname(image_path), f"DepthMap_{os.path.basename(image_path)}")

                save_depth_map_visualization(clean_depth_map, depth_map_path)
                # unique 전달 추가
                depth_text = generate_depth_text_from_coords(marker_coords_2d, clean_depth_map, targets, anchors, query, image_size=680, unique=d.get("unique", False))
                print(f"Generated Depth Text: {depth_text}")

        response = process_query(query, objects_info, client, use_image, image_path, model_name, depth_text)
        predicted_id, explanation = parse_json_response(response)
        print(f"GT id is {gt_id}; Pred id is {predicted_id}")

        gt_bbox, pred_bbox = gt_bboxes.get(gt_id), mask3d_bboxes.get(predicted_id)
        
        results.append({
            "query": query, "gt_id": gt_id, "predicted_id": predicted_id,
            "pred_bbox": pred_bbox["bbox_3d"] if pred_bbox else None,
            "gt_bbox": gt_bbox["bbox_3d"] if gt_bbox else None,
            "image_path": image_path, "parsed_query": d.get("parsed_query"),
            "explanation": explanation, "unique": d.get("unique"), "depth_text": depth_text
        })
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results for {room} to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scanrefer")
    parser.add_argument("--output_dir", default="outputs/DepthV2_scanrefer/val")
    parser.add_argument("--language_annotation_dir", default="data/scanrefer/query")
    parser.add_argument("--gt_bbox_dir", default="data/seeground_object_lookup_table/scanrefer/gt")
    parser.add_argument("--pred_bbox_dir", default="data/seeground_object_lookup_table/scanrefer/pred")
    parser.add_argument("--pcd_dir", default='referit3d/scan_data/pcd_with_global_alignment/')
    parser.add_argument("--openai_api_key", default="EMPTY")
    parser.add_argument("--openai_api_base", default="http://localhost:8000/v1")
    parser.add_argument("--use_image", action='store_true', default=True)
    parser.add_argument("--model_name", default="Qwen2-VL-7B-Instruct")
    parser.add_argument("--use_depth", action='store_true', help="Enable depth information processing.")
    args = parser.parse_args()

    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_api_base)
    
    scan_ids = sorted([f.split('.')[0] for f in os.listdir(args.language_annotation_dir) if f.endswith('.json')])
    print(f"Found {len(scan_ids)} scans in {args.language_annotation_dir}")

    for room in tqdm(scan_ids, desc="Process rooms"):
    # for room in tqdm(scan_ids[:5], desc="Process rooms (TEST RUN on 5 rooms)"): # 테스트용으로 5개 룸만 처리

        language_annotation_file = os.path.join(args.language_annotation_dir, f"{room}.json")
        process_room(
            dataset=args.dataset, room=room, output_dir=args.output_dir,
            pcd_dir=args.pcd_dir, language_annotation_file=language_annotation_file,
            gt_bbox_dir=args.gt_bbox_dir, pred_bbox_dir=args.pred_bbox_dir,
            client=client, use_image=args.use_image, use_depth=args.use_depth,
            model_name=args.model_name
        )

