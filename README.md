# AI Coding Agent Action Prediction

코딩 에이전트의 현재 요청, 행동 이력과 세션 정보를 이용해 14개 행동 클래스를 예측하는 대회 프로젝트입니다. 피처 비교·공통 검증 분할·오류 분석·제출 패키지 구성 코드를 보관합니다.

## 구성

| 경로 | 역할 |
|---|---|
| `src/` | 피처 구성, 학습·앙상블·검증·후보 비교 스크립트 |
| `script.py` | 현재 로컬 제출 추론 코드 |
| `reports/` | 초기 피처 비교 및 오류 분석 기록 |
| `artifacts/model_metadata.reference.json` | 로컬 모델의 설정 참고 자료, 실행 모델 아님 |
| `build_zip.py` | 명시한 후보 폴더의 4개 파일을 검증해 제출 ZIP 생성 |

## 실험 기록과 해석

초기 공통 세션 Group Split의 기록은 prompt-only A3 Macro-F1 **0.428382**, history 3개와 session_meta를 추가한 B3 **0.542014**입니다. 개선폭은 **+0.113632**이며 `reports/`의 해당 실험에 한정합니다. 최신 추론 모델의 성능이나 공식 대회 점수·순위를 뜻하지 않습니다. 과거 후보 결정 문서와 최신 코드의 상태도 구분해야 합니다.

현재 `script.py`에는 동일 세션의 이후 행에 있는 history에서 이전 행동을 복원해 예측에 반영하는 로직이 있습니다. 배치 입력에서의 정보 복원과, 그 시점에 이용 가능한 정보만으로 하는 순수 다음 행동 예측은 별도 평가로 설명해야 합니다. 대회 규정 및 입력 제공 조건과의 관계를 확인한 뒤 성과를 기재하세요.

## 재현 준비

Python 3.11 환경에서 실행 의존성을 준비합니다.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train_modeling_b.py --help
```

`requirements.txt`는 기존 제출 환경의 의존성 목록입니다. 개별 학습·실험 스크립트는 pandas·PyTorch·임베딩 모델 등 추가 패키지가 필요할 수 있으므로 해당 스크립트 import와 설정을 확인하세요.

공식 데이터와 `model/best_model.pkl`은 제공하지 않습니다. 대회 데이터 사용 권한을 확보하고 데이터 경로를 준비한 뒤 학습 또는 적법하게 확보한 모델을 사용해야 합니다. 일부 실험에는 원래 작업 환경의 경로가 있어 실행 전 인자·설정을 조정해야 합니다. 소스 업로드와 전체 실험 재현 성공은 별개입니다.

제출 ZIP은 `script.py`, `requirements.txt`, `model/best_model.pkl`, `model/best_model_metadata.json`만 있는 후보 폴더를 대상으로 만듭니다.

```bash
python build_zip.py --source-dir path/to/verified_candidate --output submission.zip
```

## 포트폴리오에서 설명할 내용

세션 단위 데이터 분할, history 길이 비교, 클래스별 혼동과 F1 분석, 후보 선택 근거, 학습·추론 피처 일치와 제출 격리 검증을 중심으로 설명하세요. 개인 담당 역할과 공식 결과는 확인된 증빙을 추가해야 합니다. 모델 가중치·원본 대회 데이터·대량 실험 결과와 ZIP 백업은 이 저장소에 포함하지 않습니다.
