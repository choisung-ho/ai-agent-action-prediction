# src/data.py
import json
import csv
import os

def load_jsonl(file_path):
    """
    JSONL 파일을 안전하게 로드하고 스키마를 검증합니다.
    동시에 모델링 및 검증에 필수적인 'session_id'를 id 필드에서 추출하여 추가합니다.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"데이터 파일 누락: {file_path}")

    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{file_path} 라인 {line_no} 파싱 에러: {e}")
            
            if "id" not in data:
                raise KeyError(f"{file_path} 라인 {line_no}: 식별자 'id' 누락은 허용되지 않습니다.")
            
            # [엔지니어링 고도화] id(sess_xxxx-step_NN)에서 세션 고유 키 분리 추출 (Group Split용)
            try:
                raw_id = data["id"]
                if "-step_" in raw_id:
                    session_id = raw_id.split("-step_")[0]
                else:
                    session_id = raw_id
                data["_session_id"] = session_id
            except Exception as e:
                raise ValueError(f"id 필드에서 세션 키 추출 실패 (라인 {line_no}): {e}")
                
            # 기본 결측 방어 규격 유지
            data.setdefault("current_prompt", "")
            data.setdefault("history", [])
            data.setdefault("session_meta", {})
            
            samples.append(data)
            
    return samples

def load_labels(file_path):
    """
    학습 라벨(train_labels.csv)을 로드하여 id-action 매핑 딕셔너리를 반환합니다.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"라벨 파일 누락: {file_path}")

    labels = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or reader.fieldnames[:2] != ["id", "action"]:
            raise ValueError(f"train_labels.csv 컬럼 구조 오류: {reader.fieldnames}")
            
        for row in reader:
            labels[row["id"]] = row["action"]
            
    return labels

def load_sample_submission(file_path):
    """
    sample_submission.csv를 로드하며, id 순서 유지를 보장하기 위해 리스트 형태로 반환합니다.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"제출 템플릿 파일 누락: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None or fieldnames[:2] != ["id", "action"]:
            raise ValueError(f"sample_submission.csv 컬럼 규격 위반: {fieldnames}")
            
        rows = list(reader)

    return fieldnames, rows