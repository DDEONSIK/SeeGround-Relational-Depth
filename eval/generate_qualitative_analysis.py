import sys
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random

# --- 경로 설정 및 모든 필요 모듈 임포트 ---
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from inference.utils import load_json, load_bboxes, load_scene_pcd, fuzzy_match, stem_match
from inference.DepthV2_projection import render_point_cloud_with_pytorch3d_with_objects

# Depth Anything V2 모듈 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
# ==============================================================================

# --- Depth Anything V2 헬퍼 함수 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEPTH_MODEL = None
DEPTH_TRANSFORM = None
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
                image_interpolation_method=cv2.INTER_CUBIC
            ),
            NormalizeImage(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375]
            ),
            PrepareForNet()
        ]
    
    def __call__(self, sample):
        for transform_op in self.transform:
            sample = transform_op(sample)
        return sample

def load_depth_model(encoder='vitl'):
    global DEPTH_MODEL, DEPTH_TRANSFORM
    if DEPTH_MODEL is None:
        print("Loading Depth Anything V2 model...")
        
        model_configs = {
            'vitl': {
                'encoder': 'vitl',
                'features': 256,
                'out_channels': [256, 512, 1024, 1024]
            }
        }
        
        checkpoint_path = '/workspace/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth'
        
        DEPTH_MODEL = DepthAnythingV2(**model_configs[encoder])
        DEPTH_MODEL.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        DEPTH_MODEL.eval().to(DEVICE)
        
        DEPTH_TRANSFORM = DepthAnythingV2Transform()
        
        print("Depth Anything V2 model loaded.")


def get_depth_map(image_input):
    """
    이미지 파일 경로(str) 또는 전처리된 이미지 배열(np.ndarray)을 모두 처리.
    """
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
    np.inf 값을 검은색 배경으로 처리하여 시각화.
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



def generate_depth_text_from_coords(marker_coords_2d, depth_map, targets, anchors, query):
    """
    [Nr3D 최종 버전 이식] 쿼리 인식 텍스트 생성 + 안전한 Fallback 로직 추가.
    """
    if depth_map is None or not marker_coords_2d:
        return "N/A"

    # 1. 강건한 5x5 패치 기반 깊이 샘플링
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
    # depth_info.sort(key=lambda p: p["depth"]) #text 거꾸로 생성됨
    depth_info.sort(key=lambda p: p["depth"], reverse=True)
    
    # 2. 쿼리 중심 객체 정보 추출
    target_ids = {t['bbox_id'] for t in targets}
    anchor_ids = {a['bbox_id'] for a in anchors}
    main_target = next((d for d in depth_info if d['id'] in target_ids), None)
    main_anchor = next((d for d in depth_info if d['id'] in anchor_ids), None)

    # 3. 쿼리-인식 조건부 텍스트 생성
    generated_texts = []
    if main_target and main_anchor:
        query_lower = query.lower()
        depth_keywords = ['close', 'far', 'front', 'behind', 'near', 'further', 'closer']
        spatial_keywords = ['left', 'right', 'above', 'below', 'top', 'bottom', 'side']

        if any(keyword in query_lower for keyword in depth_keywords):
            if main_target['depth'] > main_anchor['depth']: # Raw Depth 값이 더 큰 쪽이 더 가까우므로, 부등호 방향을 '<'에서 '>'로 변경.
                generated_texts.append(f"Target object (ID {main_target['id']}) is closer than anchor object (ID {main_anchor['id']}).")
            else:
                generated_texts.append(f"Target object (ID {main_target['id']}) is farther than anchor object (ID {main_anchor['id']}).")

        if any(keyword in query_lower for keyword in spatial_keywords):
            tx, ty = main_target['coords']
            ax, ay = main_anchor['coords']
            if abs(tx - ax) > abs(ty - ay) * 1.5:
                generated_texts.append(f"Target is to the {'left' if tx < ax else 'right'} of the anchor.")
            elif abs(ty - ay) > abs(tx - ax) * 1.5:
                generated_texts.append(f"Target is {'above' if ty < ay else 'below'} the anchor.")
    
    # 4. 최종 서술문 생성 또는 안전한 Fallback
    if generated_texts:
        return " ".join(generated_texts)
    else:
        closest_obj = depth_info[0]
        farthest_obj = depth_info[-1]
        if len(depth_info) > 1 and closest_obj['id'] != farthest_obj['id']:
            return f"From the current viewpoint, object {closest_obj['id']} is the closest and object {farthest_obj['id']} is the farthest."
        elif len(depth_info) >= 1:
            return f"From the current viewpoint, object {closest_obj['id']} is visible."
            
    return "N/A"
# ==============================================================================


def collect_discrepancy_cases(baseline_dir, ours_dir, dataset, gt_data):
    """
    [Phase 1: 수집] 모든 씬을 스캔하여 예측이 다른 모든 사례의 '기본 정보'만 수집.
    """
    print(f"\n[Phase 1] Collecting all discrepancy cases for {dataset.upper()}...")
    
    all_success_cases = []  # 실패 제거
    baseline_files = sorted(os.listdir(baseline_dir))

    gt_key = 'caption'
    gt_id_key = 'target_id'

    for filename in tqdm(baseline_files, desc="Scanning predictions"):
        scene_id = filename.replace('.json', '')
        if not filename.endswith('.json') or scene_id not in gt_data: continue
            
        baseline_path, ours_path = os.path.join(baseline_dir, filename), os.path.join(ours_dir, filename)
        if not os.path.exists(ours_path): continue
            
        with open(baseline_path, 'r') as f: baseline_preds = json.load(f)
        with open(ours_path, 'r') as f: ours_preds = json.load(f)
        
        pred_key = 'query' if baseline_preds and 'query' in baseline_preds[0] else 'utterance'

        baseline_map = {item.get(pred_key): item for item in baseline_preds if item.get(pred_key)}
        ours_map = {item.get(pred_key): item for item in ours_preds if item.get(pred_key)}
        scene_gt_list = gt_data[scene_id]
        
        for gt_entry in scene_gt_list:
            query_text_from_gt = gt_entry.get(gt_key)
            if not query_text_from_gt or query_text_from_gt not in baseline_map or query_text_from_gt not in ours_map: continue
                
            baseline_pred_entry = baseline_map[query_text_from_gt]
            ours_pred_entry = ours_map[query_text_from_gt]
            
            gt_id = gt_entry.get(gt_id_key)
            baseline_pred_id = baseline_pred_entry.get('predicted_id')
            ours_pred_id = ours_pred_entry.get('predicted_id')
            
            gt_id = str(gt_id) if gt_id is not None else None
            baseline_pred_id = str(baseline_pred_id) if baseline_pred_id is not None else None
            ours_pred_id = str(ours_pred_id) if ours_pred_id is not None else None
            
            if gt_id is None: continue

            is_success = baseline_pred_id != gt_id and ours_pred_id == gt_id

            if is_success:
                all_success_cases.append({'scene_id': scene_id, 'gt_entry': gt_entry})

    print(f"Found {len(all_success_cases)} potential success cases.")
    return all_success_cases

def process_selected_cases(cases, case_type, dataset, olt_data, pcd_dir, num_cases):
    """
    [Phase 2: 생성] 수집된 전체 사례에서 무작위로 샘플링하고, 해당 샘플에 대해서만 시각 자료 생성.
    """
    if not cases: return []
    
    print(f"\n[Phase 2] Randomly sampling and generating visuals for {num_cases} {case_type} cases...")
    
    random.shuffle(cases)
    selected_cases = cases[:num_cases]
    
    processed_cases = []
    
    gt_key = 'caption'
    gt_id_key = 'target_id'

    for i, case in enumerate(tqdm(selected_cases, desc=f"Generating visuals")):
        scene_id, gt_entry = case['scene_id'], case['gt_entry']
        scene_olt = olt_data[scene_id]
        
        baseline_preds = load_json(os.path.join(f"outputs/{dataset}/val/pred", f"{scene_id}.json"))
        ours_preds = load_json(os.path.join(f"outputs/DepthV2_{dataset}/val/pred", f"{scene_id}.json"))

        pred_key = 'query' if baseline_preds and 'query' in baseline_preds[0] else 'utterance'
        
        query_text = gt_entry[gt_key]
        baseline_pred_entry = next((p for p in baseline_preds if p.get(pred_key) == query_text), None)
        ours_pred_entry = next((p for p in ours_preds if p.get(pred_key) == query_text), None)

        if not baseline_pred_entry or not ours_pred_entry: continue

        object_names = [obj["target"] for obj in scene_olt.values()]
        parsed_query = gt_entry.get("parsed_query", {})
        target_name, anchor_name = parsed_query.get("Target", ""), parsed_query.get("Anchor", "")
        
        matched_targets = fuzzy_match(target_name, object_names).union(stem_match(target_name, object_names))
        matched_anchors = fuzzy_match(anchor_name, object_names).union(stem_match(anchor_name, object_names))
        
        targets = [obj for obj in scene_olt.values() if obj["target"] in matched_targets] or list(scene_olt.values())
        if dataset == 'nr3d':
            anchors = [obj for obj in scene_olt.values() if obj["target"] in matched_anchors] or targets
        else:
            anchors = [obj for obj in scene_olt.values() if obj["target"] in matched_anchors]

        scan_pc, center = load_scene_pcd(scene_id, pcd_dir)
        
        # 경로 단순화: outputs/paper_case_data/{dataset}/success/CASE_{i+1}
        base_save_dir = f"outputs/paper_case_data/{dataset}/{case_type}/CASE_{i+1}"
        
        # 1. Depth Map 생성을 위한 마커 없는 이미지(I) 렌더링
        raw_save_dir = os.path.join(base_save_dir, "raw")
        raw_image_path, _, alpha_mask = render_point_cloud_with_pytorch3d_with_objects(
            scene_olt.values(), targets, anchors, center, scan_pc,
            save_dir=raw_save_dir,
            draw_id=False, draw_img=True, return_marker_coords=True
        )

        # 2. VLM 입력을 위한 마커 있는 이미지(Im) 렌더링 (시각화용)
        # prompted_save_dir = os.path.join(base_save_dir, "prompted")
        image_path, marker_coords_2d, _ = render_point_cloud_with_pytorch3d_with_objects(
            scene_olt.values(), targets, anchors, center, scan_pc,
            save_dir=base_save_dir, # save_dir=prompted_save_dir, # 폴더 단순화
            draw_id=True, draw_img=True, return_marker_coords=True
        )

        depth_text = "N/A"
        depth_map_path = "N/A"

        if raw_image_path and alpha_mask is not None:
            # 마커 없는 raw_image_path를 입력으로 사용
            dirty_depth_map = get_depth_map(raw_image_path)
            
            clean_depth_map = None
            if dirty_depth_map is not None:
                target_size = (alpha_mask.shape[1], alpha_mask.shape[0])
                resized_dirty_depth_map = cv2.resize(dirty_depth_map, target_size, interpolation=cv2.INTER_NEAREST)
                clean_depth_map = resized_dirty_depth_map.copy()
                background_pixels = alpha_mask == 0
                clean_depth_map[background_pixels] = np.inf
            
            depth_map_path = os.path.join(base_save_dir, "DepthMap_rendered.png")
            save_depth_map_visualization(clean_depth_map, depth_map_path)
            depth_text = generate_depth_text_from_coords(marker_coords_2d, clean_depth_map, targets, anchors, query_text)

        processed_cases.append({
            "scene_id": scene_id, "query": query_text, "gt_id": str(gt_entry.get(gt_id_key)),
            "baseline_pred_id": str(baseline_pred_entry.get('predicted_id')),
            "ours_pred_id": str(ours_pred_entry.get('predicted_id')),
            "depth_text": depth_text, "image_path": image_path, "depth_map_path": depth_map_path,
        })
        
    return processed_cases

def print_cases(title, cases):
    print("\n" + "="*40)
    print(f"--- {title} ---")
    print("="*40)
    
    if not cases:
        print("No cases found.")
        return
    
    for i, case in enumerate(cases):
        ours_correct = case['ours_pred_id'] == case['gt_id']
        baseline_correct = case['baseline_pred_id'] == case['gt_id']
        
        status = "(Correct)" if ours_correct else "(Incorrect)"
        baseline_status = "(Correct)" if baseline_correct else "(Incorrect)"
        
        print(f"\n[CASE {i+1}]")
        print(f"  - Scene ID: {case['scene_id']}")
        print(f"  - Query: '{case['query']}'")
        print(f"  - Ground Truth ID: {case['gt_id']}")
        print(f"  - Baseline Predicted ID: {case['baseline_pred_id']} {baseline_status}")
        print(f"  - Ours Predicted ID:      {case['ours_pred_id']} {status}")
        print(f"  - Generated Depth Text: '{case['depth_text']}'")
        print(f"  - Rendered Image Path: {case['image_path']}")
        print(f"  - Depth Map Path:       {case['depth_map_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and display qualitative analysis cases by comparing two model predictions.")
    parser.add_argument('--dataset', type=str, required=True, choices=['nr3d', 'scanrefer'], help="Dataset to analyze.")
    parser.add_argument('--num_cases', type=int, default=5, help="Number of cases to display for each category.")
    args = parser.parse_args()

    pcd_dir = 'referit3d/scan_data/pcd_with_global_alignment/'
    baseline_pred_dir = f"outputs/{args.dataset}/val/pred"
    ours_pred_dir = f"outputs/DepthV2_{args.dataset}/val/pred"
    gt_dir = f"data/{args.dataset}/query"
    olt_dir = f"data/seeground_object_lookup_table/{args.dataset}/pred"
    
    load_depth_model()
    
    gt_data, olt_data = {}, {}
    for f in os.listdir(gt_dir):
        if f.endswith('.json'): gt_data[f.replace('.json', '')] = load_json(os.path.join(gt_dir, f))
    for f in os.listdir(olt_dir):
        if f.endswith('.json'):
            objects = load_json(os.path.join(olt_dir, f))
            olt_data[f.replace('.json', '')] = {obj['bbox_id']: obj for obj in objects}

    all_successes = collect_discrepancy_cases(baseline_pred_dir, ours_pred_dir, args.dataset, gt_data)
    
    sampled_successes = process_selected_cases(all_successes, "success", args.dataset, olt_data, pcd_dir, args.num_cases)

    print_cases(f"Success Cases (Baseline Fail -> Ours Success) for {args.dataset.upper()}", sampled_successes)
    
    print("\n" + "="*40)
    print("Analysis complete. Use the image paths to find the corresponding visualizations in 'paper_case_data/'.")
    print("="*40)
