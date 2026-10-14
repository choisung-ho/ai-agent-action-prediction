# AI Coding Agent Action Prediction

코딩 에이전트의 현재 요청, 최근 행동과 세션 정보를 이용해 14개 행동 클래스를 예측하는 대회 실험 프로젝트입니다. 피처 비교, 세션 단위 검증, 클래스별 오류 분석과 제출 코드 구성을 다룹니다.

## 해결하려던 문제

같은 요청 문장도 세션의 진행 상태와 직전 행동에 따라 다음 행동이 달라질 수 있습니다. 현재 요청만 사용하는 기준 모델과 행동 이력·세션 정보를 추가한 모델을 같은 분할에서 비교했습니다.

## 실험과 구현

- **공통 검증 기준:** 세션 단위 Group Split로 후보 피처 구성을 비교
- **피처 비교:** prompt-only, 최근 history 길이, session_meta 포함 여부
- **오류 분석:** Macro-F1, 클래스별 F1과 혼동 행렬을 이용한 비교
- **추론 구성:** 모델 메타데이터에 따라 피처 생성 방식 선택
- **제출 구성:** 추론 코드·의존성·모델·메타데이터를 별도 후보 폴더에서 검사해 ZIP 생성

## 기록된 비교 결과

아래 수치는 [초기 공통 분할 실험 기록](reports/final_model_decision.md)에 해당합니다.

| 구성 | Macro-F1 |
| --- | ---: |
| A3: 현재 요청 + TF-IDF | 0.428382 |
| B1: 현재 요청 + 최근 history 3개 | 0.526102 |
| B3: 현재 요청 + 최근 history 3개 + session_meta | 0.542014 |

B3와 A3의 차이는 **+0.113632**입니다. 이 비교는 해당 실험 분할에 한정되며 최신 추론 모델의 성능이나 공식 대회 점수·순위를 의미하지 않습니다.

## 추론 조건과 평가 한계

현재 `script.py`에는 같은 세션의 **이후 행 history에 노출된 이전 행동을 복원**하는 배치 처리 로직이 포함되어 있습니다. 이 로직은 예측 시점 이전의 정보만 사용하는 온라인 다음 행동 예측과 조건이 다릅니다. 따라서 초기 모델 비교 결과와 배치 입력 복원 동작을 구분해야 하며, 대회에서 허용하는 입력 조건도 확인해야 합니다.

모델 가중치와 공식 데이터는 제공하지 않습니다. 저장소에 있는 메타데이터는 설정 참고 자료이며 실행 모델을 대신하지 않습니다.

## 구조

| 경로 | 역할 |
| --- | --- |
| `src/` | 피처 구성, 학습·앙상블·검증·후보 비교 |
| `script.py` | 제출 추론 진입점 |
| `reports/` | 초기 피처 비교·오류 분석 기록 |
| `artifacts/model_metadata.reference.json` | 모델 설정 참고 자료 |
| `build_zip.py` | 후보 파일 검사와 ZIP 생성 |

## 실행 준비

```bash
git clone https://github.com/choisung-ho/ai-agent-action-prediction.git
cd ai-agent-action-prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python build_zip.py --help
```

Python 3.11을 사용합니다. Windows에서는 `.venv\Scripts\activate`로 가상환경을 활성화합니다. `requirements.txt`는 기존 제출 추론 환경의 목록이며, 개별 실험의 pandas·PyTorch·임베딩 의존성은 추가 준비가 필요합니다. 데이터 경로와 일부 실험의 작업 경로도 실행 환경에 맞춰 설정합니다.

```bash
python build_zip.py --source-dir path/to/verified_candidate --output submission.zip
```

제출 후보 폴더에는 `script.py`, `requirements.txt`, `model/best_model.pkl`, `model/best_model_metadata.json`이 필요합니다. 원본 대회 데이터·모델 가중치·대량 실험 산출물·ZIP 백업은 소스에서 제외합니다.
