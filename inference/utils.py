import re
from difflib import get_close_matches
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import base64
import json
import os
import numpy as np
import open3d as o3d
import torch

# nltk.download("punkt_tab") # 매번 실행 시 다운로드할 필요 없으므로 주석 처리
stemmer = PorterStemmer()


def fuzzy_match(names, object_names, threshold=0.8):
    matched_names = set()
    for name in names:
        matches = get_close_matches(name, object_names, n=1, cutoff=threshold)
        if matches:
            matched_names.add(matches[0])
    return matched_names


def stem_match(names, object_names):
    matched_names = set()
    if isinstance(names, str):
        names = [names]
    for name in names:
        name_stems = [stemmer.stem(word) for word in word_tokenize(name)]
        for obj_name in object_names:
            obj_name_stems = [stemmer.stem(word) for word in word_tokenize(obj_name)]
            if set(name_stems) & set(obj_name_stems):
                matched_names.add(obj_name)
    return matched_names


def load_json(file_path):
    """Load data from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)


def save_to_file(file_path, content):
    """Save content to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as file:
        file.write(content)


def encode_img(image_path):
    """Encode image to Base64 format."""
    with open(image_path, "rb") as file:
        encoded_image = base64.b64encode(file.read())
    return f"data:image;base64,{encoded_image.decode('utf-8')}"


def read_file_to_list(file_path):
    """
    Read a file and convert its contents to a list.
    """
    with open(file_path, "r") as file:
        lines = file.read().splitlines()
    return sorted(lines)


def calc_iou(box_a, box_b):
    """
    예측된 BBox가 없는 경우(None), IoU를 0으로 처리하여 에러 방지
    """
    if box_a is None or box_b is None:
        return 0.0
        
    box_a = np.array(box_a)
    box_b = np.array(box_b)

    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union


def parse_response(response):
    """
    JSON, 텍스트, 숫자 Fallback을 모두 처리하는 가장 안정적인 파서.
    """
    # 1. JSON 형식 우선 시도
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            pred_id = data.get("id")
            explanation = data.get("reason", "")
            # ID가 존재하면 반드시 int로 변환
            return int(pred_id) if pred_id is not None else None, explanation
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # 2. "Predicted ID: " 텍스트 형식 시도
    predicted_id = None
    explanation = None
    for line in response.split("\n"):
        if line.startswith("Predicted ID:"):
            predicted_id_str = line.split(":")[1].strip()
            if predicted_id_str.isdigit():
                predicted_id = int(predicted_id_str)
        elif line.startswith("Explanation:"):
            explanation = line.split(":", 1)[1].strip()
    
    if predicted_id is not None:
        return predicted_id, explanation

    # 3. 모든 방식 실패 시, 응답에서 첫 번째 숫자라도 추출 (Fallback)
    match = re.search(r'\d+', response)
    if match:
        return int(match.group(0)), f"Fallback extraction: {response}"
    
    return None, response


# Data Loading and Processing
def load_bboxes(room, bbox_dir, file_type="pred"):
    """Load bounding boxes (GT or predicted)."""
    bbox_file = os.path.join(bbox_dir, f"{room}.json")
    bboxes = load_json(bbox_file)
    # KeyError 방지를 위해 key를 다시 int로 변경 (원본과 동일)
    return {int(bbox["bbox_id"]): bbox for bbox in bboxes}


def generate_objects_info(pred_bbox_list):
    """Generate a formatted string of object information."""
    return "\n".join(
        [
            f"Object ID: {bbox['bbox_id']}, Type: {bbox['target']}, Dimensions: Width {bbox['bbox_3d'][3]:.2f}, Length {bbox['bbox_3d'][4]:.2f}, Height {bbox['bbox_3d'][5]:.2f}, Center Coordinates: X {bbox['bbox_3d'][0]:.2f}, Y {bbox['bbox_3d'][1]:.2f}, Z {bbox['bbox_3d'][2]:.2f}"
            for bbox in pred_bbox_list
            if bbox["target"] not in ["wall", "floor", "ceiling", "object", "objects"]
        ]
    )

# Rendering
def load_scene_pcd(room, scan_dir='referit3d/scan_data/pcd_with_global_alignment/'):
    """Load and process point cloud data."""
    pcds, colors, _, instance_labels = torch.load(
        os.path.join(scan_dir, '%s.pth' % room), weights_only=False)
    
    scan_pc = np.concatenate((pcds, colors/255), axis=1).astype("float32")
    center = np.mean(scan_pc[:, :3], axis=0)
    return scan_pc, center

