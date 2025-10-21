#SeeGround/eval/eval_scanrefer.py
import json
import os
import random
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.utils import load_json, calc_iou

def main(pred_dir):
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.json')]
    print(f'Found {len(pred_files)} JSON files in {pred_dir}')
    assert len(pred_files) > 0, 'No JSON files found in the specified directory'

    unique_total = 0
    correct_25 = 0
    unique_25 = 0
    correct_50 = 0
    unique_50 = 0
    total_predictions = 0

    for pred_file in sorted(pred_files):
        pred_file_path = os.path.join(pred_dir, pred_file)
        if not os.path.exists(pred_file_path): continue
        preds = load_json(pred_file_path)

        for pred_entry in preds:
            gt_bbox = pred_entry['gt_bbox']
            pred_bbox = pred_entry['pred_bbox']

            if gt_bbox is None or pred_bbox is None:
                iou = 0
            else:
                iou = calc_iou(gt_bbox, pred_bbox)

            if pred_entry['unique']:
                unique_total += 1

            if iou >= 0.25:
                correct_25 += 1
                if pred_entry['unique']:
                    unique_25 += 1

            if iou >= 0.5:
                correct_50 += 1
                if pred_entry['unique']:
                    unique_50 += 1
        total_predictions += len(preds)

    print()
    print('Unique@25         {:.2%}    {} / {}'.format(unique_25 / unique_total if unique_total > 0 else 0, unique_25, unique_total))
    print('Multiple@25       {:.2%}     {} / {}'.format((correct_25 - unique_25) / (total_predictions - unique_total) if (total_predictions - unique_total) > 0 else 0, correct_25 - unique_25, total_predictions - unique_total))
    print('Unique@50         {:.2%}    {} / {}'.format(unique_50 / unique_total if unique_total > 0 else 0, unique_50, unique_total))
    print('Multiple@50       {:.2%}     {} / {}'.format((correct_50 - unique_50) / (total_predictions - unique_total) if (total_predictions - unique_total) > 0 else 0, correct_50 - unique_50, total_predictions - unique_total))
    print()
    print('Acc@25           {:.2%}    {} / {}'.format(correct_25 / total_predictions if total_predictions > 0 else 0, correct_25, total_predictions))
    print('Acc@50           {:.2%}    {} / {}'.format(correct_50 / total_predictions if total_predictions > 0 else 0, correct_50, total_predictions))


if __name__ == '__main__':
    pred_dir = 'outputs/scanrefer/val/pred'
    main(pred_dir)