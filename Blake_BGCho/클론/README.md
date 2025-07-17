# model-playground
모델 테스트 레포
# AI Theory Lab 🧪

이 저장소는 수강 예정인 AI 강의의 내용을 실전에 적용하고, 다양한 상용 모델과 이론을 테스트하기 위한 실험실입니다.  
직접 모델을 불러오고, 튜닝하고, 실험하며 이론을 체화하는 것을 목표로 합니다.

---

## 📁 프로젝트 구조
ai-theory-lab/
├── runner/          # 실험 실행 스크립트 (학습, 검증 등)
├── models/          # 상용 모델 및 이론 적용 모델
├── datasets/        # 실험용 공개 데이터셋
├── results/         # 실험 결과 및 로그 저장소
├── notebooks/       # 실험 기록 및 시각화
├── requirements.txt # 필요 라이브러리 목록
└── README.md

## 🛠️ 목표 및 활용 계획

- 수강 중 배운 개념을 실제 코드로 실험하고 정리
- 상용 모델(YOLO 등)을 커스터마이징하여 이론 적용
- 다양한 데이터셋으로 검증 및 결과 비교
- 반복 실험을 위한 config 기반 runner 구축

---

## 📌 TODO (초기 계획)

- [ ] YOLO 모델(혹은 적절한 상용모델) 불러오기 및 runner로 실험
- [ ] Titanic 데이터셋(혹은 적절한 데이터셋)으로 학습 테스트
- [ ] 실험 config 시스템 구성
- [ ] 모델 튜닝 기록 및 성능 비교 툴 추가

---

## 🧹 `.gitignore` (예정)

- 과정 중 발생하는 찌꺼기 파일 꾸준히 업데이트

--

## 🧾 기록 방식

모델 또는 실험별로 다음과 같은 방식으로 버전 관리합니다:

models/
├── yolo_base.py
└── yolo_theory_20250707.py  # YYYYMMDD 형식으로 이론 적용 버전 기록

results/
└── yolo_20250707/
├── logs/
└── config_used.yaml

---

## 📚 라이선스

> 본 프로젝트는 개인 학습 및 연구 목적으로만 사용됩니다.


---

## 🧪 개발자 명령어 메모

```bash
# 가상환경 만들기
conda env create -f environment.yml

# 환경 업데이트
conda env update --file environment.yml --prune

# 현재 환경 저장
conda env export > environment.yml

# 환경 활성화
conda activate model-playground-mac

# 환경 삭제
conda remove --name model-playground-mac --all