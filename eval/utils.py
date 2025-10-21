import json
import numpy as np    

def load_json(pred_file):
    with open(pred_file, 'r') as f:
        return json.load(f)

def calc_iou(box_a, box_b):
    """
    box_a 또는 box_b가 None인 경우를 모두 처리하여 충돌을 완벽히 방지.
    """
    # 주석: pred_bbox (box_a) 또는 gt_bbox (box_b)가 None일 경우, IoU는 0.
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

