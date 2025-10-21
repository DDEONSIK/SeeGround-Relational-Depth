import json
import os

def inspect_data(pred_dir):
    """
    Prediction 데이터 파일의 구조를 검사하여 실제 키 이름을 확인합니다.
    """
    print("="*50)
    print("Starting Prediction data inspection...")
    
    # 샘플 파일 하나를 선택
    sample_file_name = "scene0011_00.json"
    sample_file_path = os.path.join(pred_dir, sample_file_name)
    
    if not os.path.exists(sample_file_path):
        print(f"[FATAL ERROR] Sample Prediction file not found at: {sample_file_path}")
        return

    print(f"Inspecting file: {sample_file_path}")
    
    with open(sample_file_path, 'r') as f:
        data = json.load(f)
    
    if not data or not isinstance(data, list) or not isinstance(data[0], dict):
        print("[FATAL ERROR] Prediction file format is unexpected.")
        return
        
    # 첫 번째 항목의 모든 키를 출력
    first_item_keys = data[0].keys()
    
    print("\n--- Inspection Result ---")
    print(f"Keys found in the first prediction entry: {list(first_item_keys)}")
    print("="*50)

if __name__ == '__main__':
    pred_directory = "outputs/DepthV2_nr3d/val/pred"
    inspect_data(pred_directory)
