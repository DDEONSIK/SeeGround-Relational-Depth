import openai
import base64
import json
import re

# --- 1. 사용자 입력 정보 ---
# 이미지 경로를 컨테이너 내부의 올바른 절대 경로로 변경
IMAGE_PATH = "/workspace/SGTEST.png" 

OBJECT_LIST_TEXT = """
1: mug A
2: mug B
3: wet tissues
4: battery
5: keyboard
6: mouse
7: scissors
8: monitor
9: chair
"""
QUERY_TEXT = "What is the distance between mugA and mugB in centimeters?"


# 한글 쿼리 및 객체 목록
# OBJECT_LIST_TEXT = """
# 1: 머그컵A
# 2: 머그컵B
# 3: 물티슈
# 4: 건전지
# 5: 키보드
# 6: 마우스
# 7: 가위
# 8: 모니터
# 9: 의자
# """
# QUERY_TEXT = "물티슈는 어디에 있는가?"

# --- 2. VLM 서버 정보 ---
VLM_API_BASE = "http://localhost:8000/v1"
VLM_API_KEY = "not-needed" # API 키는 vLLM 구동시 따로 설정하지 않았다면 아무 값이나 넣어도 무방.
MODEL_NAME = "Qwen2-VL-7B-Instruct"

# --- 3. API 요청 및 결과 처리 (이 아래는 수정할 필요 없음) ---

def encode_image_to_base64(image_path):
    """이미지 파일을 Base64로 인코딩하는 함수"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_json_response(response: str):
    """VLM의 JSON 응답을 파싱하는 함수"""
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            predicted_id = data.get("id")
            explanation = data.get("reason", "No explanation provided.")
            return predicted_id, explanation
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    numbers = re.findall(r'\d+', response)
    if numbers:
        return int(numbers[0]), f"Explanation extracted via fallback (found numbers): {response}"
    return None, response

def parse_query_with_vlm(client, query):
    """쿼리에서 Target과 Anchor를 추출하기 위해 VLM을 호출하는 함수"""
    system_prompt = (
        "You are a query analyzer. Your task is to parse the user's description to identify "
        "the main 'target' object and any 'anchor' objects used as reference points. "
        "Respond ONLY in this JSON format: {\"target\": \"<target_object>\", \"anchor\": \"<anchor_object>\"}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
        )
        parsed_data = json.loads(response.choices[0].message.content)
        return parsed_data.get("target", ""), parsed_data.get("anchor", "")
    except Exception:
        words = query.replace(" 찾아줘", "").replace(" 찾아", "").split()
        return words[-1], " ".join(words[:-1])

# --- 스크립트 실행 시작 ---

# 1. 입력 확인 및 쿼리 분석
print("=" * 40)
print(f"[입력 쿼리] {QUERY_TEXT}")
client = openai.OpenAI(api_key=VLM_API_KEY, base_url=VLM_API_BASE)
parsed_target, parsed_anchor = parse_query_with_vlm(client, QUERY_TEXT)
print(f"[쿼리 분석] Target: {parsed_target}, Anchor: {parsed_anchor}")

# 2. 최종 예측
print("-" * 40)
print("[최종 추론] VLM에 이미지와 프롬프트 전송 중...")

# 프롬프트 구성
SYSTEM_INFO = (
    "You are an expert system. Your task is to identify a single object ID from a provided list based on a description. "
    "You MUST respond ONLY in the following JSON format. Do NOT add any other text before or after the JSON."
)
RESPONSE_FORMAT = '{"id": <the_object_id>, "reason": "<your_brief_explanation>"}'
user_prompt = (
    f"Object list:\n{OBJECT_LIST_TEXT.strip()}\n\n"
    f"User's description: '{QUERY_TEXT}'\n\n"
    f"Identify the object that best matches the user's description and respond ONLY in this JSON format: {RESPONSE_FORMAT}"
)

# 이미지 인코딩 및 메시지 생성
base64_image = encode_image_to_base64(IMAGE_PATH)
image_url = f"data:image/png;base64,{base64_image}"
messages = [
    {"role": "system", "content": SYSTEM_INFO},
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": user_prompt},
        ],
    },
]

# VLM에 API 요청
chat_response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages
)
raw_response = chat_response.choices[0].message.content
predicted_id, explanation = parse_json_response(raw_response)

# 3. 결과 출력
print("[추론 완료] 결과:")
print("-" * 40)
print(f"  - 예측 ID: {predicted_id}")
print(f"  - VLM 답변: {explanation}")
print(f"  - Raw JSON: {raw_response}")
print("=" * 40)