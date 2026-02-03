# 03. 식물 질병 진단 분류 실습

## 개요
YOLO Classification 모델을 활용하여 **토마토 잎 이미지**로부터 질병을 자동 진단하는 AI 시스템을 구축합니다.

---

## 학습 목표
1. YOLO Classification 모델의 학습 방법 이해
2. 이미지 분류(Classification) 태스크의 개념 학습
3. 학습된 모델을 활용한 실제 진단 시스템 구현

---

## 데이터셋

### 출처
- PlantVillage Dataset (https://github.com/spMohanty/PlantVillage-Dataset)

### 클래스 (5개)
| 클래스명 | 한글명 | 설명 |
|---------|--------|------|
| Tomato_healthy | 건강 | 정상적인 토마토 잎 |
| Tomato_Bacterial_spot | 세균성 점무늬병 | 세균(Xanthomonas)에 의한 갈색 반점 |
| Tomato_Early_blight | 잎마름병 | 곰팡이에 의한 동심원 무늬 반점 |
| Tomato_Leaf_Mold | 잎곰팡이병 | 잎 뒷면 올리브색 곰팡이 |
| Tomato_Yellow_Leaf_Curl_Virus | 황화잎말림바이러스 | 잎이 노랗게 변하고 말림 |

### 데이터 분할
| 분할 | 이미지 수 | 클래스당 |
|------|----------|---------|
| train | 600장 | 120장 |
| val | 100장 | 20장 |
| test | 50장 | 10장 |
| **총** | **750장** | - |

### 폴더 구조
```
data/
├── train/
│   ├── Tomato_healthy/        (120장)
│   ├── Tomato_Bacterial_spot/ (120장)
│   ├── Tomato_Early_blight/   (120장)
│   ├── Tomato_Leaf_Mold/      (120장)
│   └── Tomato_Yellow_Leaf_Curl_Virus/ (120장)
├── val/   (클래스당 20장)
└── test/  (클래스당 10장)
```

---

## 노트북 구성

### 01_train.ipynb - 모델 학습
YOLO Classification 모델을 학습시킵니다.

**주요 내용:**
- ultralytics 라이브러리 사용
- `yolo11n-cls.pt` 모델 (Nano Classification)
- 학습 결과: `runs/classify/train/`에 저장

**빈칸 (5개):**
| # | 위치 | 정답 | 힌트 |
|---|------|------|------|
| 1 | `DATA_PATH = _______` | `'../data'` | 상위 폴더의 data 디렉토리 |
| 2 | `'yolo11n-_____.pt'` | `cls` | Classification 접미사 |
| 3 | `model = YOLO(_______)` | `MODEL_SIZE` | 위에서 정의한 변수 |
| 4 | `model._______()` | `train` | 학습 메서드 |
| 5 | `"./runs/_______/train/"` | `classify` | Classification 결과 폴더 |

---

### 02_test.ipynb - 모델 평가
학습된 모델의 성능을 테스트 데이터로 평가합니다.

**주요 내용:**
- Top-1, Top-5 Accuracy 측정
- 혼동 행렬(Confusion Matrix) 시각화

**빈칸 (3개):**
| # | 위치 | 정답 | 힌트 |
|---|------|------|------|
| 1 | `from ultralytics import _______` | `YOLO` | 핵심 모듈 |
| 2 | `MODEL_PATH = _______` | `'./runs/classify/train/weights/best.pt'` | 학습된 모델 경로 |
| 3 | `model._______()` | `val` | 검증 메서드 |

---

### 03_inference.ipynb - 진단 시스템
실제 이미지를 입력받아 질병을 진단하는 시스템을 구현합니다.

**주요 내용:**
- 진단 함수 구현
- 질병별 정보 및 대처 방법 출력
- 진단 결과 시각화

**빈칸 (3개):**
| # | 위치 | 정답 | 힌트 |
|---|------|------|------|
| 1 | `model._______()` | `predict` | 예측 메서드 |
| 2 | `results[0]._______` | `probs` | 확률 객체 |
| 3 | `probs._______` | `top1` | 최고 확률 클래스 인덱스 |

---

## Detection vs Classification 비교

| 항목 | Detection/Pose | Classification |
|------|---------------|----------------|
| 모델 파일 | `yolo11n.pt`, `yolo11n-pose.pt` | `yolo11n-cls.pt` |
| 데이터 설정 | config.yaml 필요 | 폴더 구조만으로 충분 |
| 결과 저장 | `runs/detect/`, `runs/pose/` | `runs/classify/` |
| 출력 형태 | 바운딩 박스, 키포인트 | 클래스 확률 |
| 평가 지표 | mAP, Precision, Recall | Top-1, Top-5 Accuracy |

---

## 주요 코드 스니펫

### 모델 학습
```python
from ultralytics import YOLO

model = YOLO('yolo11n-cls.pt')
results = model.train(
    data='../data',
    epochs=50,
    imgsz=224,
    batch=32
)
```

### 모델 검증
```python
model = YOLO('./runs/classify/train/weights/best.pt')
metrics = model.val(data='../data', split='test')

print(f"Top-1 Accuracy: {metrics.top1:.4f}")
print(f"Top-5 Accuracy: {metrics.top5:.4f}")
```

### 단일 이미지 예측
```python
results = model.predict(image_path)
probs = results[0].probs

predicted_class = results[0].names[probs.top1]
confidence = probs.top1conf.item()
```

---

## 질병 정보 (PPT용)

### 1. 건강 (Tomato_healthy)
- **상태**: 정상
- **특징**: 균일한 녹색, 반점 없음
- **조치**: 현재 관리 유지

### 2. 세균성 점무늬병 (Tomato_Bacterial_spot)
- **원인**: 세균 (Xanthomonas)
- **특징**: 잎에 작은 갈색/검은색 반점
- **조치**: 구리 기반 살균제 살포, 감염 잎 제거

### 3. 잎마름병 (Tomato_Early_blight)
- **원인**: 곰팡이 (Alternaria solani)
- **특징**: 동심원 무늬의 갈색 반점 (과녁 모양)
- **조치**: 살균제 살포, 통풍 개선

### 4. 잎곰팡이병 (Tomato_Leaf_Mold)
- **원인**: 곰팡이 (Passalora fulva)
- **특징**: 잎 뒷면에 올리브색/갈색 곰팡이
- **조치**: 환기 개선, 습도 낮추기

### 5. 황화잎말림바이러스 (TYLCV)
- **원인**: 바이러스 (담배가루이가 전파)
- **특징**: 잎이 노랗게 변하고 위로 말림
- **조치**: 감염 식물 격리/제거, 해충 방제

---

## 실습 흐름

```
1. 라이브러리 설치 (pip install ultralytics)
       ↓
2. Google Drive 마운트 & 경로 설정
       ↓
3. 데이터셋 확인 (750장, 5개 클래스)
       ↓
4. 모델 학습 (01_train.ipynb)
   - yolo11n-cls.pt 로드
   - 50 epochs 학습
       ↓
5. 모델 평가 (02_test.ipynb)
   - Top-1/Top-5 Accuracy
   - 혼동 행렬 분석
       ↓
6. 진단 시스템 (03_inference.ipynb)
   - 이미지 입력 → 질병 진단
   - 대처 방법 안내
```

---

## 예상 학습 시간
- 모델 학습 (Colab GPU): 약 10-15분
- 전체 실습: 약 1-1.5시간

---

## 참고 자료
- [Ultralytics YOLO Classification 문서](https://docs.ultralytics.com/tasks/classify/)
- [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
