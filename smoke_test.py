# smoke_test.py
import os
import subprocess
import shutil
import sys
import json
import csv

def is_nonempty_file(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

def write_fallback_test_files(base_dir, sandbox_dir, n=20):
    """Create tiny test/sample files from train.jsonl when official test files are absent."""
    train_path = os.path.join(base_dir, "data", "train.jsonl")
    if not is_nonempty_file(train_path):
        raise FileNotFoundError("공식 test 파일도 없고 fallback 생성용 train.jsonl도 없습니다.")

    test_path = os.path.join(sandbox_dir, "data", "test.jsonl")
    sample_path = os.path.join(sandbox_dir, "data", "sample_submission.csv")
    ids = []
    with open(train_path, "r", encoding="utf-8") as src, open(test_path, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids.append(obj["id"])
            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
            if len(ids) >= n:
                break
    with open(sample_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "action"])
        writer.writeheader()
        for sample_id in ids:
            writer.writerow({"id": sample_id, "action": "respond_only"})
    print(f"[안내] 공식 test/sample 파일이 비어 있어 train.jsonl에서 {len(ids)}건의 임시 smoke test 데이터를 생성했습니다.")

def run_smoke_test():
    print("="*60)
    print("[엔지니어링 내부 검증] Smoke Test를 실행합니다.")
    print("="*60)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SANDBOX_DIR = os.path.join(BASE_DIR, "sandbox_test")
    
    # 1. 기존 가상 샌드박스 청소 및 신규 격리 폴더 생성
    if os.path.exists(SANDBOX_DIR):
        shutil.rmtree(SANDBOX_DIR)
    os.makedirs(SANDBOX_DIR)
    
    try:
        # 2. self-contained 검증: 오직 script.py 하나만 복사
        shutil.copy(os.path.join(BASE_DIR, "script.py"), os.path.join(SANDBOX_DIR, "script.py"))
        
        # 3. 데이터 구조 모사 연결
        os.makedirs(os.path.join(SANDBOX_DIR, "data"), exist_ok=True)
        os.makedirs(os.path.join(SANDBOX_DIR, "model"), exist_ok=True)
        
        source_test = os.path.join(BASE_DIR, "data", "test.jsonl")
        source_sample = os.path.join(BASE_DIR, "data", "sample_submission.csv")
        if is_nonempty_file(source_test) and is_nonempty_file(source_sample):
            shutil.copy(source_test, os.path.join(SANDBOX_DIR, "data", "test.jsonl"))
            shutil.copy(source_sample, os.path.join(SANDBOX_DIR, "data", "sample_submission.csv"))
        else:
            write_fallback_test_files(BASE_DIR, SANDBOX_DIR)
        
        if os.path.exists(os.path.join(BASE_DIR, "model", "best_model.pkl")):
            shutil.copy(os.path.join(BASE_DIR, "model", "best_model.pkl"), os.path.join(SANDBOX_DIR, "model", "best_model.pkl"))
        else:
            print("[경고] 테스트를 수행하기 위한 'model/best_model.pkl' 파일이 확인되지 않습니다.")
            print("모델링 파트에게 실제 모델을 수령한 후 진행하거나, 더미 모델을 배치하십시오.")
            sys.exit(1)

        metadata_path = os.path.join(BASE_DIR, "model", "best_model_metadata.json")
        if os.path.exists(metadata_path):
            shutil.copy(metadata_path, os.path.join(SANDBOX_DIR, "model", "best_model_metadata.json"))
        else:
            print("[경고] model/best_model_metadata.json이 없습니다. script.py는 Modeling A prompt-only 방식으로 기본 실행됩니다.")

        # 4. 고립 환경 내부에서 프로세스 격리 독립 실행
        print("-> 격리 샌드박스 내 환경에서 script.py 실행 중...")
        result = subprocess.run(
            [sys.executable, "script.py"],
            cwd=SANDBOX_DIR,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        
        # 5. 최종 산출물 물리 검증
        output_csv = os.path.join(SANDBOX_DIR, "output", "submission.csv")
        if os.path.exists(output_csv):
            print("\n" + "="*60)
            print("[결과: PASS] script.py의 완벽한 self-contained 독립 구동을 확인했습니다.")
            print("= submission.csv 파일 정상 추출 완료.")
            print("="*60)
        else:
            print("\n[결과: FAIL] 런타임 에러는 없었으나 submission.csv 파일이 소실되었습니다.")
            
    except subprocess.CalledProcessError as e:
        print("\n" + "="*60)
        print("[결과: FAIL] 격리 환경 실행 중 검증 단계에서 크래시가 발생했습니다.")
        print("오류 내용:\n", e.stderr)
        print("="*60)
    finally:
        # 가상 검증 디렉토리 클리닝
        if os.path.exists(SANDBOX_DIR):
            shutil.rmtree(SANDBOX_DIR)

if __name__ == "__main__":
    run_smoke_test()
