import json
import os
import argparse
import sys
from tqdm import tqdm

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.utils import load_json, calc_iou

def main(pred_dir):
    """
    주어진 예측 디렉토리의 JSON 파일들을 평가.
    이 함수는 원본 eval_nr3d.py의 핵심 평가 로직을 그대로 사용.
    """
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.json')])
    print(f'\nFound {len(pred_files)} JSON files in {pred_dir}')
    assert len(pred_files) > 0, 'No JSON files found.'

    # 원본 스크립트와 동일한 통계 변수 초기화
    total_predictions = 0
    easy_total, dep_total = 0, 0
    correct_easy, correct_dep = 0, 0
    correct_25, correct_50 = 0, 0

    # 각 예측 파일을 순회하며 평가
    for pred_file in tqdm(pred_files, desc="Evaluating Scenes"):
        preds = load_json(os.path.join(pred_dir, pred_file))
        
        for pred_entry in preds:
            # ===> 핵심 로직: 예측 파일에 이미 포함된 gt_bbox와 pred_bbox를 직접 사용.
            # 외부 파일 조회 없이, 원본 스크립트와 동일한 방식으로 작동.
            if 'gt_bbox' not in pred_entry or 'pred_bbox' not in pred_entry:
                continue

            gt_bbox = pred_entry['gt_bbox']
            pred_bbox = pred_entry['pred_bbox']

            # 예측 실패로 pred_bbox가 None인 경우를 처리
            if pred_bbox is None:
                total_predictions += 1
                if pred_entry.get('easy'): easy_total += 1
                if pred_entry.get('view_dep'): dep_total += 1
                continue

            iou = calc_iou(gt_bbox, pred_bbox)

            # 원본 스크립트와 완벽히 동일한 통계 업데이트 로직
            total_predictions += 1
            if pred_entry.get('easy'):
                easy_total += 1
            if pred_entry.get('view_dep'):
                dep_total += 1

            if iou >= 0.25:
                correct_25 += 1
                if pred_entry.get('easy'):
                    correct_easy += 1
                if pred_entry.get('view_dep'):
                    correct_dep += 1
            
            if iou >= 0.5:
                correct_50 += 1

    # ===> 최종 결과 출력 (원본 스크립트와 완벽히 동일한 계산식 및 포맷)
    print("\n--- Evaluation Results ---")
    
    hard_total = total_predictions - easy_total
    correct_hard = correct_25 - correct_easy
    indep_total = total_predictions - dep_total
    correct_indep = correct_25 - correct_dep

    # 0으로 나누는 오류 방지
    if easy_total > 0: print('Easy      {:.2%}    {} / {}'.format(correct_easy / easy_total, correct_easy, easy_total))
    if hard_total > 0: print('Hard      {:.2%}    {} / {}'.format(correct_hard / hard_total, correct_hard, hard_total))
    if dep_total > 0: print('Dep       {:.2%}    {} / {}'.format(correct_dep / dep_total, correct_dep, dep_total))
    if indep_total > 0: print('Indep     {:.2%}    {} / {}'.format(correct_indep / indep_total, correct_indep, indep_total))
    print()
    if total_predictions > 0:
        print('Acc@25         {:.2%}    {} / {}'.format(correct_25 / total_predictions, correct_25, total_predictions))
        print('Acc@50         {:.2%}    {} / {}'.format(correct_50 / total_predictions, correct_50, total_predictions))
    print("------------------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True, help="Directory containing prediction JSON files.")
    args = parser.parse_args()
    main(args.pred_dir)