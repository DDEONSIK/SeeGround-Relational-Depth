# SeeGround 프로젝트 최종 설정 및 실행 가이드 (24GB VRAM 환경)

## Phase 0: Host Machine - 사전 준비

### 1. 프로젝트 저장소 복제 및 폴더 생성
# SeeGround 코드를 다운로드하고, 데이터와 모델을 저장할 폴더를 미리 생성함.
git clone https://github.com/iris0329/SeeGround.git
cd SeeGround
mkdir -p data models

### 2. 데이터셋 다운로드 및 압축 해제
# 2.1. 사전 처리된 Object Lookup Table (OLT)
wget "https://github.com/user-attachments/files/18056532/seeground_object_lookup_table.zip" -O data/seeground_object_lookup_table.zip
unzip data/seeground_object_lookup_table.zip -d data/

# 2.2. ScanRefer 쿼리 데이터
wget "@@@ scanrefer.zip" -O data/scanrefer.zip
unzip data/scanrefer.zip -d data/
# 생성된 ScanRefer 폴더로 관련 파일 이동
mv data/ScanRefer_filtered*.json data/ScanRefer/
mv data/ScanRefer_filtered*.txt data/ScanRefer/

# 2.3. Nr3D 쿼리 데이터 (TSV 형식)
# gdown 설치가 필요할 수 있음: pip install gdown
gdown '1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC' -O data/nr3d.tsv

# 2.4. 3D 장면 원본 데이터 (ScanNet)
wget "https://www.dropbox.com/s/n0m5bpfvea1fg7w/referit3d.tar.gz?dl=1" -O referit3d.tar.gz
tar -xzvf referit3d.tar.gz

# 2.5. 임시 다운로드 파일 정리
rm data/seeground_object_lookup_table.zip data/scanrefer.zip referit3d.tar.gz

---

## Phase 1: Host Machine - Docker 환경 준비 및 서버 실행

### 1. Docker 이미지 다운로드
docker pull qwenllm/qwenvl

### 2. (선택사항) Docker 시스템 정리
# 이전 작업으로 불필요한 데이터가 쌓인 경우, 아래 명령어로 디스크 공간을 확보함.
# "Are you sure...?" 질문에 y를 입력함.
docker system prune -a

### 3. VLM 서버 실행 (터미널 1)
# 아래 명령어로 Docker 컨테이너를 실행함과 동시에 VLM 서버를 시작함.
# --name seeground_env 로 컨테이너 이름을 명시적으로 지정함.
# 24GB VRAM 환경에 맞춰 7B 모델과 메모리/컨텍스트 길이 최적화 옵션을 사용함.
# 이 터미널은 추론이 끝날 때까지 계속 실행 상태로 두어야 함.
docker run -it --rm --name seeground_env --gpus all --shm-size=8g -v "$(pwd)":/workspace qwenllm/qwenvl python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --served-model-name Qwen2-VL-72B-Instruct \
  --tensor_parallel_size=1 \
  --download-dir /workspace/models/.cache \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --enforce-eager

---

## Phase 2: Container - 스크립트 오류 수정 및 전처리

### 1. 실행 중인 컨테이너에 접속 (터미널 2)
# Host PC에서 새 터미널을 열고, 아래 명령어로 실행 중인 컨테이너에 접속함.
docker exec -it seeground_env bash

### 2. 필수 라이브러리 설치
# 컨테이너에 접속된 터미널 2에서 실행함.
pip install jsonlines pandas open3d

### 3. Nr3D 데이터 전처리 (TSV -> JSONL)
# 원본 Nr3D 데이터는 TSV 형식이므로, 프로젝트에서 사용 가능한 JSONL 형식으로 변환함.
# 이 과정은 최초 1회만 필요함.
cat > /workspace/convert_nr3d.py << EOL
import pandas as pd
import json

# TSV 파일 로드. 에러 발생 시 'quoting=3' (QUOTE_NONE) 옵션 사용
df = pd.read_csv('data/nr3d.tsv', sep='\\t', quoting=3) 

# JSONL 파일로 저장
with open('data/Nr3D/nr3d.jsonl', 'w', encoding='utf-8') as file:
    for index, row in df.iterrows():
        # 각 행을 dictionary로 변환 후 JSON 문자열로 변환하여 파일에 쓰기
        json.dump(row.to_dict(), file, ensure_ascii=False)
        file.write('\\n')

print("TSV to JSONL conversion complete.")
EOL

python /workspace/convert_nr3d.py

### 4. 전체 스크립트 수정 (최초 1회만 실행)
# 프로젝트의 모든 파이썬 스크립트들이 참조하는 경로와 키 값을 현재 환경에 맞게 일괄 수정함.
sed -i "s|default='SeeGround/prompts/parsing_query.txt'|default='prompts/parsing_query.txt'|g" parse_query/generate_query_data_*.py
sed -i "s|'/remote-home/share/vg_datasets/referit3d/official_data/annotations/meta_data/scannetv2-labels.combined.tsv'|'referit3d/annotations/meta_data/scannetv2-labels.combined.tsv'|g" parse_query/generate_query_data_scanrefer.py
sed -i "s|if item\\['scan_id'\\]|if item['scene_id']|g" parse_query/generate_query_data_nr3d.py
sed -i "s|'nr3d.jsonl'|'Nr3D/nr3d.jsonl'|g" parse_query/generate_query_data_nr3d.py
sed -i "s|f'/remote-home/rongli/projection_img/{dataset}/{room}/{i}'|f'outputs/projection_img/{dataset}/{room}/{i}'|g" inference/inference_nr3d.py
sed -i "s|default='http://10.10.10.14:8000/v1'|default='http://localhost:8000/v1'|g" inference/*.py
sed -i "s|pred_dir = '.*'|pred_dir = 'outputs/nr3d/val/pred'|" eval/eval_nr3d.py
sed -i "s|pred_dir = '.*'|pred_dir = 'outputs/scanrefer/val/pred'|" eval/eval_scanrefer.py

---

## Phase 3: Container - 추론 파이프라인 실행

# 모든 명령어는 컨테이너에 접속된 터미널 2에서 실행함.
# VLM 서버는 터미널 1에서 계속 실행 중이어야 함.

### 1. 앵커 및 타겟 생성 (쿼리 전처리)
python parse_query/generate_query_data_nr3d.py
python parse_query/generate_query_data_scanrefer.py

### 2. 최종 예측 수행
# 이 과정은 GPU 성능에 따라 수 시간이 소요될 수 있음.
python inference/inference_nr3d.py
python inference/inference_scanrefer.py

### 3. 평가 진행
python eval/eval_nr3d.py
python eval/eval_scanrefer.py

---

## Phase 4: 재접속 및 작업 재개 가이드

### 상황: 컴퓨터를 재부팅했거나, 터미널 연결이 끊겼거나, 컨테이너가 중지된 경우

#### Step 1: 컨테이너 상태 확인
# Host PC에서 터미널을 열고 아래 명령어로 현재 실행 중인 모든 Docker 컨테이너를 확인함.
# 이 명령을 통해 우리의 컨테이너(seeground_env)가 실행 중인지, 중지 상태인지 알 수 있음.
docker ps -a

#### Step 2: 상황에 따른 조치

##### 시나리오 A: 컨테이너가 실행 중일 때 (STATUS가 'Up'일 경우)
# 이 경우는 컨테이너는 살아있지만, VLM 서버 프로세스가 종료되었거나 사용자가 터미널에서 나온 상태입니다.
# 아래 절차에 따라 서버를 재시작하고 작업 환경에 접속함.

# 1. 컨테이너 내부 셸(Shell)로 접속 (터미널 1)
# 서버를 실행시킬 첫 번째 터미널을 엽니다.
docker exec -it seeground_env bash

# 2. VLM 서버 재시작 (방금 접속한 터미널 1 내부에서 실행)
# 이 터미널은 서버 역할을 하므로, 추론이 끝날 때까지 켜두어야 함.
# (만약 Address in use 에러가 발생하면, 이전 서버가 완전히 종료되지 않은 것이므로 잠시 후 다시 시도함.)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --served-model-name Qwen2-VL-72B-Instruct \
  --tensor_parallel_size=1 \
  --download-dir /workspace/models/.cache \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --enforce-eager

# 3. 작업용 터미널 추가 접속 (Host PC에서 새 터미널 2를 열고 실행)
# 이제 추론 명령어를 입력할 두 번째 터미널을 엽니다.
docker exec -it seeground_env bash

##### 시나리오 B: 컨테이너가 중지되었을 때 (STATUS가 'Exited'일 경우)
# 컨테이너 전원이 꺼진 상태이므로, 컨테이너를 켜는 것부터 시작함.

# 1. 컨테이너 시작 (Host PC에서 실행)
# 이 명령어는 컨테이너를 백그라운드에서 실행 상태로만 만들며, 터미널에 특별한 출력은 없음.
docker start seeground_env

# 2. VLM 서버 실행 및 작업용 터미널 접속
# 컨테이너가 켜졌으므로, 이제 위 '시나리오 A'의 1~3번 절차를 그대로 따라 서버를 켜고 작업 환경에 접속함.

#### Step 3: 작업 재개
# 위 절차를 통해 VLM 서버(터미널 1)와 작업용 셸(터미널 2)이 모두 준비되었음.
# 터미널 2에서 이전에 중단했던 Phase 3의 단계부터 이어서 작업을 수행하면 됩니다.
# 예를 들어, 쿼리 전처리까지 완료했다면 Phase 3의 '최종 예측 수행'부터 다시 시작함.