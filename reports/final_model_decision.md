# 최종 후보 모델 결정 기록

## 결정

현재 검증 기준에서는 **B3: `current_prompt + history 3개 + session_meta`**를 최종 기본 후보로 둔다.

## 근거

공통 세션 Group Split 기준:

| 모델 | 구성 | Macro-F1 |
|---|---|---:|
| A3 | prompt only + word/char TF-IDF | 0.428382 |
| B0 | prompt + 직전 action 1개 | 0.507247 |
| B1 | prompt + 최근 history 3개 | 0.526102 |
| B2 | prompt + 최근 history 5개 | 0.485693 |
| B3 | prompt + history 3개 + session_meta | 0.542014 |

B3는 A3 대비 **+0.114 Macro-F1** 개선되었다.

## 적용할 feature 설정

```json
{
  "feature_builder": "modeling_b_v1",
  "feature_config": {
    "history_n": 3,
    "include_meta": true
  }
}
```

이 설정은 `model/best_model_metadata.json`에 반영되어 있으며, `script.py`는 해당 메타데이터를 읽어 추론 feature builder를 자동 선택한다.

## 남은 작업

1. 공식 원본 `train.jsonl`, `train_labels.csv`를 확보한다.
2. 같은 `splits/common_split.csv` 기준으로 B3를 재현 학습한다.
3. 최종 제출용 모델을 전체 train 데이터로 재학습한다.
4. `model/best_model.pkl`을 생성한다.
5. `python build_zip.py`로 제출 ZIP을 만든다.
6. `python smoke_test.py`로 격리 실행을 확인한다.

## 주의

`EDA_오류분석_데이터셋`에는 실험 결과와 split 파일만 있고, 실제 `.pkl` 모델 파일은 포함되어 있지 않다. 따라서 현재 `best_model_metadata.json`은 최종 후보 설정을 고정하기 위한 파일이며, 제출 가능한 상태가 되려면 반드시 `model/best_model.pkl`이 추가로 필요하다.
