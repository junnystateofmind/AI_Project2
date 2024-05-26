# AI_Project2
Topic: Design and Analysis Report of Action Recognition AI Model for UCF101 dataset.

Submission: Please submit the project code (including comments) and the modeling analysis report as a single compressed ZIP file with the following naming format: Name_StudentID_pr2.zip

e.g., "박재문_2018-12967_pr2.zip"

Due Date: 6/19 23:59 PM

Report:  (1) Designing a Suitable Model for UCF101 Action Recognition and the Reasoning Behind It.

(2) Model Performance and Result Analysis.

You are allowed to use chatGPT, but model modifications must be made with the goal of improving the performance of the generated model, and comparative analysis must also be included.

Dataset: UCF101: https://www.crcv.ucf.edu/data/UCF101.php#Results_on_UCF101 (외부 사이트로 연결합니다.)

Grading Criteria:

Difficulty level of implementing the model
Depth and quality of analysis presented in the report (maximum 3 pages)
Model's performance on the selected dataset
Note:

While there are no programming language restrictions, Python is recommended.
Transfer learning is not allowed.
PLAGIARISM WILL NOT BE TOLERATED: Please be aware that copying assignments will result in a score of 0, so ensure to submit your work thoughtfully.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

모델 기본 구조는 Vision Transformer(ViT)를 사용할 예정
-> 05.26 ViT는 inductive bias가 없어서 작은 데이터셋에서는 성능이 좋지 않음, 따라서 LSTM으로 접근

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

디렉토리 구조는 다음과 같다

```
AI_Project2/
│
├── data/
│   ├── raw/
│   │   ├── UCF101/  # 원본 데이터셋 폴더
│   │   └── annotations/  # 주석 파일 (필요한 경우)
│   ├── processed/
│   │   └── train/  # 훈련 데이터
│   │   └── val/  # 검증 데이터
│   │   └── test/  # 테스트 데이터
│   └── download_dataset.py  # 데이터셋 다운로드 및 전처리 스크립트
│
├── models/
│   ├── vit.py  # Vision Transformer 모델 정의
│   ├── custom_layers.py  # 추가적인 커스텀 레이어 정의 (필요한 경우)
│   └── utils.py  # 모델 관련 유틸리티 함수
│
├── training/
│   ├── train.py  # 모델 훈련 스크립트
│   ├── train_utils.py  # 훈련 관련 유틸리티 함수
│   ├── callbacks.py  # 훈련 중 사용할 콜백 함수들
│   └── configs.py  # 훈련 설정 파일
│
├── evaluation/
│   ├── evaluate.py  # 모델 평가 스크립트
│   ├── metrics.py  # 평가 지표 계산 스크립트
│   └── visualization.py  # 평가 결과 시각화 스크립트
│
├── reports/
│   ├── performance_report.md  # 모델 성능 및 결과 분석 보고서
│   └── figures/  # 그래프, 혼동 행렬 등 시각 자료
│
├── notebooks/
│   ├── data_exploration.ipynb  # 데이터셋 탐색 및 시각화
│   ├── model_training.ipynb  # 모델 훈련 실험 노트북
│   └── results_analysis.ipynb  # 결과 분석 노트북
│
├── scripts/
│   ├── preprocess_data.sh  # 데이터 전처리 스크립트
│   └── train_model.sh  # 모델 훈련 스크립트
│
├── requirements.txt  # 필요한 패키지 목록
└── README.md  # 프로젝트 설명 및 실행 방법


```
