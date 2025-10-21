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
    주어진 예측 디렉토리의 JSON 파일들을 평가합니다.
    이 함수는 원본 eval_scanrefer.py의 핵심 평가 로직을 그대로 사용합니다.
    """
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.json')])
    print(f'\nFound {len(pred_files)} JSON files in {pred_dir}')
    assert len(pred_files) > 0, 'No JSON files found.'

    # 원본 스크립트와 동일한 통계 변수 초기화
    unique_total = 0
    correct_25, unique_25 = 0, 0
    correct_50, unique_50 = 0, 0
    total_predictions = 0

    # 각 예측 파일을 순회하며 평가
    for pred_file in tqdm(pred_files, desc="Evaluating Scenes"):
        preds = load_json(os.path.join(pred_dir, pred_file))
        
        # 원본 스크립트의 버그로 보이는 부분까지 동일하게 구현하여 공정한 비교를 보장
        # total_predictions가 내부 루프에서 증가하는 점을 그대로 반영
        current_file_predictions = 0
        
        for pred_entry in preds:
            # ===> 핵심 로직: 예측 파일에 이미 포함된 gt_bbox와 pred_bbox를 직접 사용
            if 'gt_bbox' not in pred_entry or 'pred_bbox' not in pred_entry:
                continue

            gt_bbox = pred_entry['gt_bbox']
            pred_bbox = pred_entry['pred_bbox']
            
            # 예측 실패 또는 GT 누락의 경우를 처리
            if gt_bbox is None or pred_bbox is None:
                iou = 0
            else:
                iou = calc_iou(gt_bbox, pred_bbox)

            # 원본 스크립트와 완벽히 동일한 통계 업데이트 로직
            if pred_entry.get('unique', False):
                unique_total += 1

            if iou >= 0.25:
                correct_25 += 1
                if pred_entry.get('unique', False):
                    unique_25 += 1
            
            if iou >= 0.5:
                correct_50 += 1
                if pred_entry.get('unique', False):
                    unique_50 += 1
            
            current_file_predictions += 1
        
        # 원본 스크립트의 로직을 그대로 따름 (파일 내 예측 수만큼 전체 카운트 증가)
        total_predictions += len(preds)


    # ===> 최종 결과 출력 (원본 스크립트와 완벽히 동일한 계산식 및 포맷)
    print("\n--- Evaluation Results ---")
    
    multiple_total = total_predictions - unique_total
    correct_multiple_25 = correct_25 - unique_25
    correct_multiple_50 = correct_50 - unique_50

    # 0으로 나누는 오류 방지
    print('Unique@25         {:.2%}    {} / {}'.format(unique_25 / unique_total if unique_total > 0 else 0, unique_25, unique_total))
    print('Multiple@25       {:.2%}      {} / {}'.format(correct_multiple_25 / multiple_total if multiple_total > 0 else 0, correct_multiple_25, multiple_total))
    print('Unique@50         {:.2%}    {} / {}'.format(unique_50 / unique_total if unique_total > 0 else 0, unique_50, unique_total))
    print('Multiple@50       {:.2%}      {} / {}'.format(correct_multiple_50 / multiple_total if multiple_total > 0 else 0, correct_multiple_50, multiple_total))
    print()
    if total_predictions > 0:
        print('Acc@25            {:.2%}    {} / {}'.format(correct_25 / total_predictions, correct_25, total_predictions))
        print('Acc@50            {:.2%}    {} / {}'.format(correct_50 / total_predictions, correct_50, total_predictions))
    print("------------------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True, help="Directory containing prediction JSON files.")
    args = parser.parse_args()
    main(args.pred_dir)
