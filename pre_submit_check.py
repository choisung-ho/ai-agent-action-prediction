import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile


ALL_CLASSES = {
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash",
    "run_tests", "lint_or_typecheck", "ask_user", "plan_task",
    "web_search", "respond_only",
}

REQUIRED_MEMBERS = (
    "script.py",
    "requirements.txt",
    "model/best_model.pkl",
    "model/best_model_metadata.json",
)


def fail(message):
    print(f"[FAIL] {message}")
    sys.exit(1)


def pass_step(message):
    print(f"[PASS] {message}")


def validate_zip_name(zip_path):
    name = os.path.basename(zip_path)
    if any(ord(ch) > 127 for ch in name):
        fail(f"ZIP 파일명에 한글/비ASCII 문자가 포함되어 있습니다: {name}")
    if any(ch.isspace() for ch in name):
        fail(f"ZIP 파일명에 공백이 포함되어 있습니다: {name}")
    pass_step(f"ZIP 파일명 규칙 통과: {name}")


def validate_zip_members(zip_path):
    with zipfile.ZipFile(zip_path) as zipf:
        members = zipf.namelist()
        corrupt_member = zipf.testzip()

    if corrupt_member is not None:
        fail(f"ZIP CRC 검증 실패: {corrupt_member}")
    if len(members) != len(set(members)):
        fail(f"ZIP 내부 중복 파일이 있습니다: {members}")
    if members != list(REQUIRED_MEMBERS):
        fail(
            "ZIP 내부 파일 순서 또는 구성 오류: "
            f"expected={list(REQUIRED_MEMBERS)}, actual={members}"
        )
    if set(members) != set(REQUIRED_MEMBERS):
        fail(f"ZIP 내부 필수 파일 누락 또는 추가 파일: {members}")
    forbidden = [m for m in members if m.startswith("__MACOSX/") or os.path.basename(m) == ".DS_Store"]
    if forbidden:
        fail(f"불필요한 macOS 메타 파일 포함: {forbidden}")
    nonportable = [m for m in members if "\\" in m or m.startswith("/") or ".." in m.split("/")]
    if nonportable:
        fail(f"ZIP 내부 경로가 이식 가능하지 않습니다: {nonportable}")
    pass_step("ZIP 내부 필수 구조 통과")


def prepare_eval_like_data(base_dir, sandbox_dir, max_rows):
    source_test = os.path.join(base_dir, "data", "test.jsonl")
    source_sample = os.path.join(base_dir, "data", "sample_submission.csv")
    train_path = os.path.join(base_dir, "data", "train.jsonl")

    data_dir = os.path.join(sandbox_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(source_test) and os.path.getsize(source_test) > 0 and os.path.exists(source_sample) and os.path.getsize(source_sample) > 0:
        shutil.copy(source_test, os.path.join(data_dir, "test.jsonl"))
        shutil.copy(source_sample, os.path.join(data_dir, "sample_submission.csv"))
        with open(os.path.join(data_dir, "sample_submission.csv"), encoding="utf-8") as f:
            row_count = sum(1 for _ in csv.DictReader(f))
        return row_count, "official_local_test"

    if not os.path.exists(train_path) or os.path.getsize(train_path) == 0:
        fail("공식 test/sample 파일이 없고, 대체 검증용 data/train.jsonl도 없습니다.")

    row_count = 0
    with open(train_path, encoding="utf-8") as src, \
            open(os.path.join(data_dir, "test.jsonl"), "w", encoding="utf-8") as test_out, \
            open(os.path.join(data_dir, "sample_submission.csv"), "w", newline="", encoding="utf-8") as sample_out:
        writer = csv.DictWriter(sample_out, fieldnames=["id", "action"])
        writer.writeheader()
        for line in src:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sample_id = obj["id"]
            test_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            writer.writerow({"id": sample_id, "action": "respond_only"})
            row_count += 1
            if max_rows and row_count >= max_rows:
                break
    return row_count, "train_fallback"


def validate_submission(output_path, expected_rows):
    if not os.path.exists(output_path):
        fail("script.py 실행 후 output/submission.csv가 생성되지 않았습니다.")

    with open(output_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames[:2] != ["id", "action"]:
            fail(f"submission.csv 컬럼 오류: {reader.fieldnames}")
        rows = list(reader)

    if len(rows) != expected_rows:
        fail(f"submission.csv 행 수 불일치: expected={expected_rows}, actual={len(rows)}")

    invalid = [(row.get("id"), row.get("action")) for row in rows if row.get("action") not in ALL_CLASSES]
    if invalid:
        fail(f"규정 외 action 값 발견: {invalid[:5]}")

    pass_step(f"submission.csv 생성/형식/action 검증 통과: {len(rows)} rows")


def main():
    parser = argparse.ArgumentParser(description="제출 전 ZIP 구조, 추론 실행, 결과 파일, 제한시간을 한 번에 검증합니다.")
    parser.add_argument("--zip", default="baseline_submit.zip", help="검증할 제출 ZIP 파일 경로")
    parser.add_argument("--timeout-sec", type=int, default=120, help="script.py 실행 제한 시간")
    parser.add_argument("--max-rows", type=int, default=0, help="0이면 전체 train fallback 데이터로 검증")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = args.zip if os.path.isabs(args.zip) else os.path.join(base_dir, args.zip)

    if not os.path.exists(zip_path):
        fail(f"검증할 ZIP 파일이 없습니다: {zip_path}")

    print("=" * 70)
    print("[제출 전 최종 검증] ZIP 구조 / script.py 실행 / submission.csv / 시간 제한")
    print("=" * 70)

    validate_zip_name(zip_path)
    validate_zip_members(zip_path)

    with tempfile.TemporaryDirectory(prefix="submit_precheck_") as sandbox_dir:
        with zipfile.ZipFile(zip_path) as zipf:
            zipf.extractall(sandbox_dir)

        expected_rows, data_source = prepare_eval_like_data(base_dir, sandbox_dir, args.max_rows)
        print(f"[INFO] 검증 데이터: {data_source}, rows={expected_rows}")

        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, "script.py"],
                cwd=sandbox_dir,
                text=True,
                capture_output=True,
                timeout=args.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            fail(f"script.py 실행 시간이 {args.timeout_sec}초를 초과했습니다.")
        elapsed = time.time() - start

        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            fail(f"script.py 실행 중 에러 발생: returncode={result.returncode}")

        validate_submission(os.path.join(sandbox_dir, "output", "submission.csv"), expected_rows)
        pass_step(f"추론 시간 제한 통과: {elapsed:.2f}s <= {args.timeout_sec}s")

    print("=" * 70)
    print("[최종 결과: PASS] 제출 전 핵심 오류 조건을 모두 통과했습니다.")
    print("=" * 70)


if __name__ == "__main__":
    main()
