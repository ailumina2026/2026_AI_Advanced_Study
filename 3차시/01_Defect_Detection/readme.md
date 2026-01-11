# 🔍 Industrial Defect Detection

이 프로젝트는 산업용 제품의 결함을 검출하는 딥러닝 모델을 학습하고 테스트하는 교육 자료입니다.<br>
YOLOv11을 사용하여 데이터 준비부터 모델 학습, 평가, 그리고 시각화까지의 전체 과정을 실습합니다.

## 📂 Directory Structure

- **code/**
  - `01_train.ipynb`: 데이터셋 로드 및 YOLOv11 모델 학습
  - `02_test.ipynb`: 학습된 모델 평가 및 성능 지표 확인
  - `03_overlay.ipynb`: 결함 검출 결과 시각화 및 응용
- **data/**
  - `config.yaml`: YOLOv11 학습을 위한 데이터셋 설정 파일
  - `readme.md`: 데이터셋 상세 정보 및 사용 가이드
  - `images/`: 학습(train), 검증(val), 테스트(test) 이미지
  - `labels/`: YOLO 형식의 annotation 파일
- **runs/**
  - 학습 결과 및 추론 결과가 저장되는 디렉토리

## 🚀 Getting Started

### Step 0️⃣: 사전 준비

#### 필요한 것
- Google 계정 (Google Drive 및 Colab 사용)
- 인터넷 연결
- 웹 브라우저

#### 환경 요구사항
- Google Colab (자동으로 Python, PyTorch 등 제공)
- GPU 런타임 (Colab에서 무료로 제공)

### Step 1️⃣: Google Drive에 프로젝트 업로드

#### 1-1. 프로젝트 다운로드
```
GitHub에서 이 repository를 다운로드하거나 clone합니다.
또는 01_Defect_Detection 폴더 전체를 다운로드받습니다.
```

#### 1-2. Google Drive에 업로드
1. [Google Drive](https://drive.google.com)에 접속
2. 본인의 Drive에서 업로드할 위치로 이동
   - 예: `내 드라이브` → `새 폴더` → `2026_AI_Advanced_Study` → `3차시`
3. `01_Defect_Detection` 폴더 전체를 드래그 앤 드롭으로 업로드
4. 업로드가 완료될 때까지 대기 (약 2-3분 소요)

#### 1-3. 경로 확인
업로드 후 본인의 폴더 경로를 확인하세요:
```
예시 경로: MyDrive/2026_AI_Advanced_Study/3차시/01_Defect_Detection/
```

### Step 2️⃣: Google Colab에서 노트북 열기

#### 2-1. 첫 번째 노트북 열기
1. Google Drive에서 `01_Defect_Detection/code/01_train.ipynb` 파일 찾기
2. 파일 우클릭 → `연결 앱` → `Google Colaboratory` 선택
3. Colab 창이 새로 열립니다

#### 2-2. GPU 런타임 설정 (중요!)
1. Colab 상단 메뉴: `런타임` → `런타임 유형 변경`
2. 하드웨어 가속기: `T4 GPU` 선택
3. `저장` 클릭

### Step 3️⃣: 경로 설정 수정

#### 3-1. WORK_DIR 경로 수정
노트북의 **"3. 작업 디렉토리 설정"** 섹션에서 경로를 수정합니다:

```python
# 수정 전 (예시)
WORK_DIR = '/content/drive/MyDrive/2026_AI_Advanced_Study/3차시/01_Defect_Detection'

# 수정 후 (본인의 Drive 구조에 맞게)
WORK_DIR = '/content/drive/MyDrive/여기에/본인의/경로/01_Defect_Detection'
```

**경로 찾는 방법:**
1. Google Drive에서 `01_Defect_Detection` 폴더 우클릭
2. `위치 정보 가져오기` 클릭
3. 경로를 복사하여 `/content/drive/MyDrive/` 뒤에 붙이기

### Step 4️⃣: 노트북 실행하기

#### 4-1. 01_train.ipynb (모델 학습)
1. 첫 번째 셀부터 순서대로 실행 (Shift + Enter)
2. Google Drive 마운트 권한 요청 시 `계정 선택` → `허용` 클릭
3. 학습 완료까지 대기 (약 5~20분)
4. 학습 완료 후 모델이 `runs/defect/weights/best.pt`에 저장됨

#### 4-2. 02_test.ipynb (모델 평가)
1. Google Drive에서 `code/02_test.ipynb` 열기
2. `WORK_DIR` 경로만 본인의 Drive 경로로 수정
   - `MODEL_PATH`는 **이미 설정되어 있어** 별도 수정 불필요!
3. 셀을 순서대로 실행
4. 성능 지표 확인 (mAP, Precision, Recall 등)

#### 4-3. 03_overlay.ipynb (결함 시각화)
1. Google Drive에서 `code/03_overlay.ipynb` 열기
2. `WORK_DIR` 경로만 본인의 Drive 경로로 수정
   - `MODEL_PATH`는 **이미 설정되어 있어** 별도 수정 불필요!
3. 셀을 순서대로 실행
4. 결함 검출 결과 이미지 확인

💡 **Tip**: 모델 경로가 고정되어 있어서 각 노트북에서 `WORK_DIR`만 수정하면 바로 실행할 수 있습니다!

### Step 5️⃣: 결과 확인

학습이 완료되면 Google Drive의 다음 위치에서 결과를 확인할 수 있습니다:

```
01_Defect_Detection/
└── runs/
    ├── defect/                # 01_train.ipynb 학습 결과
    │   ├── weights/
    │   │   └── best.pt        # 학습된 모델 (중요!)
    │   ├── results.png        # 학습 그래프
    │   └── confusion_matrix.png
    ├── defect_val/            # 02_test.ipynb 평가 결과
    │   ├── confusion_matrix.png
    │   ├── PR_curve.png
    │   └── test_results.txt
    └── defect_predict/        # 03_overlay.ipynb 시각화 결과
        └── *.png              # 결함 검출 결과 이미지들
```

**폴더 설명:**
- `runs/defect/`: 모델 학습 결과 및 학습된 모델 저장
- `runs/defect_val/`: 모델 평가 결과 및 성능 지표
- `runs/defect_predict/`: 테스트 이미지에 대한 결함 검출 결과

## 📚 Dataset Information

본 실습에서는 산업용 제품 결함 검출 데이터셋을 사용합니다.

### 데이터셋 구성
- **학습 이미지**: 320장 (80%)
- **검증 이미지**: 40장 (10%)
- **테스트 이미지**: 40장 (10%)
- **클래스**: 1개 (defect - 결함)

### 데이터 형식
- 이미지 형식: JPG
- Annotation 형식: YOLO (txt)
- 이미지 크기: 다양 (학습 시 640x640으로 자동 조정)

데이터셋은 `data/` 폴더에 포함되어 있으며, 별도로 다운로드할 필요가 없습니다.

## 🎓 Curriculum Goals

1. **Object Detection Basics**: YOLO 기반 객체 검출 모델의 원리 이해
2. **Transfer Learning**: 사전 학습된 YOLOv11 모델을 활용한 전이학습
3. **Model Evaluation**: mAP, Precision, Recall, F1-Score 등 성능 지표 분석
4. **Visualization**: 검출된 결함 영역을 이미지 위에 시각화하는 방법 습득
5. **Industrial Application**: 실제 산업 현장에서의 품질 관리 자동화 활용

## 📊 Performance Metrics

모델 성능 평가 시 다음 지표들을 확인합니다:

- **mAP50**: IoU 임계값 0.5에서의 평균 정밀도
- **mAP50-95**: IoU 임계값 0.5~0.95 범위에서의 평균 정밀도
- **Precision**: 모델이 결함이라고 예측한 것 중 실제 결함의 비율
- **Recall**: 실제 결함 중 모델이 올바르게 검출한 비율
- **F1-Score**: Precision과 Recall의 조화평균

## 💡 Use Cases

- **품질 관리 자동화**: 제조 라인에서 실시간 결함 검출
- **불량률 감소**: 조기 결함 발견을 통한 생산성 향상
- **비용 절감**: 수동 검사 인력 및 시간 절감
- **데이터 분석**: 결함 패턴 분석을 통한 공정 개선

## ⚠️ 주의사항

- **GPU 설정 필수**: Google Colab에서 GPU 런타임을 활성화하지 않으면 학습이 매우 느려집니다.
- **경로 수정**: 각 노트북에서 `WORK_DIR`을 본인의 Google Drive 경로로 수정해야 합니다.
- **학습 시간**: GPU 성능에 따라 5~20분 소요됩니다. Colab 무료 버전 사용 시 연결이 끊어지지 않도록 주의하세요.
- **재학습**: 같은 폴더(`runs/defect/`)에 결과가 저장되므로, 재학습 시 이전 결과가 덮어씌워집니다.

## 🔗 References

- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)
- [YOLO Object Detection](https://github.com/ultralytics/ultralytics)
- [Industrial Defect Detection Papers](https://paperswithcode.com/task/defect-detection)

## 📄 License

이 프로젝트는 교육 목적으로 제공됩니다.
