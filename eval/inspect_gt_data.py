import json
import os

def inspect_data(gt_dir):
    """
    Ground Truth 데이터 파일의 구조를 검사하여 실제 키 이름을 확인합니다.
    """
    print("="*50)
    print("Starting Ground Truth data inspection...")
    
    # 샘플 파일 하나를 선택
    sample_file_name = "scene0011_00.json"
    sample_file_path = os.path.join(gt_dir, sample_file_name)
    
    if not os.path.exists(sample_file_path):
        print(f"[FATAL ERROR] Sample GT file not found at: {sample_file_path}")
        return

    print(f"Inspecting file: {sample_file_path}")
    
    with open(sample_file_path, 'r') as f:
        data = json.load(f)
    
    if not data or not isinstance(data, list) or not isinstance(data[0], dict):
        print("[FATAL ERROR] GT file format is unexpected.")
        return
        
    # 첫 번째 항목의 모든 키를 출력
    first_item_keys = data[0].keys()
    
    print("\n--- Inspection Result ---")
    print(f"Keys found in the first data entry: {list(first_item_keys)}")
    print("="*50)

    # 'utterance' 또는 'description'이 아닌 다른 유력한 키를 찾음
    possible_keys = ['utterance', 'description', 'sentence', 'caption', 'query']
    found_key = None
    for key in first_item_keys:
        if key in possible_keys:
            found_key = key
            break
            
    if not found_key:
         for key in first_item_keys:
            if isinstance(data[0][key], str) and len(data[0][key].split()) > 2:
                 found_key = key
                 break

    if found_key:
        print(f"\n[ACTION REQUIRED]")
        print(f"The correct key for the query text appears to be: '{found_key}'")
        print("Please use this key in the 'DepthV2_eval_nr3d.py' script on the line marked with '===>'.")
    else:
        print("\n[ACTION FAILED] Could not automatically determine the correct key.")

if __name__ == '__main__':
    # 기본 GT 디렉토리 경로
    gt_directory = "data/nr3d/query"
    inspect_data(gt_directory)
