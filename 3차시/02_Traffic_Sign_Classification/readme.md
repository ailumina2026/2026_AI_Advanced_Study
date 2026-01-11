# 🚦 Traffic Sign Classification

이 프로젝트는 Hugging Face의 사전학습된 CNN 모델을 사용하여 교통 표지판을 분류하는 딥러닝 모델을 학습하고 테스트하는 교육 자료입니다.<br>
Transfer Learning을 통해 적은 데이터로도 높은 정확도를 달성하는 과정을 실습합니다.

## 📂 Directory Structure

- **code/**
  - `01_train.ipynb`: Hugging Face 모델 Fine-tuning 및 학습
  - `02_test.ipynb`: 학습된 모델 평가 및 성능 지표 확인
  - `03_visualize.ipynb`: 예측 결과 시각화
- **data/**
  - `readme.md`: 데이터셋 상세 정보
  - `images/`: 학습/검증/테스트 이미지 (자동 생성)
- **runs/**
  - 학습 결과 및 평가 결과가 저장되는 디렉토리

## 🚀 Getting Started

### Step 0️⃣: 사전 준비

#### 필요한 것
- Google 계정 (Google Drive 및 Colab 사용)
- 인터넷 연결

#### 환경 요구사항
- Google Colab (Python, PyTorch 자동 제공)
- GPU 런타임 (T4 GPU 권장)

### Step 1️⃣: Google Drive에 프로젝트 업로드

1. GitHub에서 `02_Traffic_Sign_Classification` 폴더 다운로드
2. [Google Drive](https://drive.google.com)에 접속
3. 본인의 Drive에 업로드
   - 예: `내 드라이브/2026_AI_Advanced_Study/3차시/02_Traffic_Sign_Classification/`

### Step 2️⃣: Google Colab에서 노트북 열기

1. Google Drive에서 `code/01_train.ipynb` 파일 우클릭
2. `연결 앱` → `Google Colaboratory` 선택
3. GPU 런타임 설정: `런타임` → `런타임 유형 변경` → `T4 GPU`

### Step 3️⃣: 경로 설정 수정

노트북의 `WORK_DIR` 변수를 본인의 Drive 경로로 수정:

```python
WORK_DIR = '/content/drive/MyDrive/2026_AI_Advanced_Study/3차시/02_Traffic_Sign_Classification'
```

### Step 4️⃣: 노트북 실행하기

#### 4-1. 01_train.ipynb (모델 학습)
1. 셀을 순서대로 실행 (Shift + Enter)
2. 로컬에 저장된 데이터셋을 자동으로 로드 (별도 다운로드 불필요)
3. 학습 완료까지 대기 (약 10-15분)
4. 모델이 `runs/classification/final_model/`에 저장됨

#### 4-2. 02_test.ipynb (모델 평가)
1. `code/02_test.ipynb` 열기
2. `WORK_DIR`만 수정 (모델 경로는 자동 설정)
3. 셀을 순서대로 실행
4. Accuracy, Precision, Recall, F1-Score 확인

#### 4-3. 03_visualize.ipynb (결과 시각화)
1. `code/03_visualize.ipynb` 열기
2. `WORK_DIR`만 수정
3. 셀을 순서대로 실행
4. 예측 결과 이미지 확인

### Step 5️⃣: 결과 확인

```
02_Traffic_Sign_Classification/
└── runs/
    ├── classification/           # 학습 결과
    │   ├── final_model/         # 학습된 모델
    │   └── training_curves.png  # 학습 곡선
    ├── classification_val/       # 평가 결과
    │   ├── confusion_matrix.png
    │   └── test_results.txt
    └── classification_pred/      # 시각화 결과
        └── predictions.png
```

## 📚 Dataset Information

### 데이터셋 구성
- **학습 이미지**: 320장 (80%)
- **검증 이미지**: 40장 (10%)
- **테스트 이미지**: 40장 (10%)
- **클래스**: 5개 (GTSRB에서 선택된 교통 표지판)

### 데이터 소스
- **원본**: German Traffic Sign Recognition Benchmark (GTSRB)
- **Hugging Face**: `tanganke/gtsrb`
- **저장 위치**: `data/images/` (미리 준비되어 포함됨)
- **데이터 형식**: JPG 이미지 (class_0 ~ class_4 폴더에 분류됨)

## 🎓 Curriculum Goals

1. **Transfer Learning**: 사전학습된 CNN 모델 활용법 이해
2. **Hugging Face Transformers**: Hugging Face 라이브러리 사용법 습득
3. **Fine-tuning**: 적은 데이터로 모델을 효과적으로 학습하는 방법
4. **Model Evaluation**: 분류 모델의 성능 지표 분석
5. **Practical Application**: 실제 교통 표지판 인식 시스템 구축

## 📊 Performance Metrics

모델 성능 평가 시 확인하는 지표:

- **Accuracy**: 전체 예측 중 정확한 예측의 비율
- **Precision**: 모델이 특정 클래스라고 예측한 것 중 실제로 그 클래스인 비율
- **Recall**: 실제 특정 클래스 중 모델이 올바르게 예측한 비율
- **F1-Score**: Precision과 Recall의 조화평균
- **Confusion Matrix**: 클래스별 예측 결과 매트릭스

## 💡 Use Cases

- **자율주행**: 교통 표지판 인식 시스템
- **운전 보조**: 실시간 표지판 인식 및 경고
- **교통 안전**: 표지판 모니터링 및 분석
- **교육**: 딥러닝 기반 이미지 분류 학습

## 🔧 Technology Stack

- **Framework**: PyTorch
- **Library**: Hugging Face Transformers, Datasets
- **Model**: MobileNetV2 (사전학습 모델)
- **Platform**: Google Colab
- **Visualization**: Matplotlib, Seaborn

## ⚠️ 주의사항

- **GPU 필수**: GPU 런타임을 활성화하지 않으면 학습이 느려집니다
- **경로 수정**: `WORK_DIR`을 본인의 Drive 경로로 수정 필수
- **학습 시간**: GPU 성능에 따라 10-15분 소요
- **데이터 포함**: 데이터셋이 미리 포함되어 있으므로 별도 다운로드 불필요
- **Drive 업로드**: GitHub에서 다운로드한 전체 폴더(data 포함)를 Drive에 업로드해야 함

## 🔗 References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [GTSRB Dataset](http://benchmark.ini.rub.de/)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)

## 📄 License

이 프로젝트는 교육 목적으로 제공됩니다.
