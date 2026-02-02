# 04. 공사현장 안전모 착용 탐지 실습

## 개요
YOLO Detection 모델을 활용하여 **공사현장 이미지**에서 안전모 착용 여부를 자동 탐지하고 실시간 경고를 제공하는 AI 시스템을 구축합니다.

---

## 학습 목표
1. YOLO Detection 모델의 학습 방법 이해
2. 객체 탐지(Object Detection) 태스크의 개념 학습
3. 학습된 모델을 활용한 실시간 안전 경고 시스템 구현

---

## 데이터셋

### 출처
- Roboflow Hard Hat Workers Dataset (https://universe.roboflow.com/joseph-nelson/hard-hat-workers)

### 클래스 (2개)
| 클래스 ID | 클래스명 | 한글명 | 설명 |
|:---------:|:--------:|:------:|:-----|
| 0 | head | 미착용 | 안전모를 착용하지 않은 머리 |
| 1 | helmet | 착용 | 안전모를 착용한 머리 |

### 데이터 분할
| 분할 | 이미지 수 |
|:----:|:---------:|
| train | 700장 |
| val | 100장 |
| test | 100장 |
| **총** | **900장** |

### 폴더 구조
```
data/
├── config.yaml           # 데이터셋 설정 파일 (Detection 필수!)
├── train/
│   ├── images/           (700장)
│   └── labels/           (바운딩 박스 좌표)
├── val/
│   ├── images/           (100장)
│   └── labels/
├── test/
│   ├── images/           (100장)
│   └── labels/
└── demo/                 (테스트용 이미지)
```

---

## 노트북 구성

### 01_train.ipynb - 모델 학습
YOLO Detection 모델을 학습시킵니다.

**주요 내용:**
- ultralytics 라이브러리 사용
- `yolo11n.pt` 모델 (Nano Detection)
- 학습 결과: `runs/detect/train/`에 저장

**빈칸 (4개):**
| # | 위치 | 정답 | 힌트 |
|---|------|------|------|
| 1 | `DATA_CONFIG = _______` | `'../data/config.yaml'` | config.yaml 파일 경로 |
| 2 | `model = YOLO(_______)` | `MODEL_SIZE` | 위에서 정의한 변수 |
| 3 | `model._______()` | `train` | 학습 메서드 |
| 4 | `"./runs/_______/train/"` | `detect` | Detection 결과 폴더 |

---

### 02_test.ipynb - 모델 평가
학습된 모델의 성능을 테스트 데이터로 평가합니다.

**주요 내용:**
- mAP50, mAP50-95 측정
- 샘플 이미지 예측 테스트
- 바운딩 박스 시각화

**빈칸 (3개):**
| # | 위치 | 정답 | 힌트 |
|---|------|------|------|
| 1 | `MODEL_PATH = _______` | `'./runs/detect/train/weights/best.pt'` | 학습된 모델 경로 |
| 2 | `model._______()` | `val` | 검증 메서드 |
| 3 | `results[0]._______` | `boxes` | Detection 결과 (바운딩 박스) |

---

### 03_safety_system.ipynb - 안전 경고 시스템
실제 이미지를 입력받아 안전 상태를 분석하고 경고하는 시스템을 구현합니다.

**주요 내용:**
- 안전 상태 분석 함수 구현
- 위험 수준 계산 (미착용 비율)
- 경고 메시지 및 시각화

**빈칸 (4개):**
| # | 위치 | 정답 | 힌트 |
|---|------|------|------|
| 1 | `boxes = result._______` | `boxes` | Detection 결과 (바운딩 박스) |
| 2 | `cls_id = int(box._______[0])` | `cls` | 바운딩 박스의 클래스 ID |
| 3 | `danger_level = _______ / _______` | `head_count / total_count` | 미착용 비율 계산 |
| 4 | `model._______(...)` | `predict` | 예측 메서드 |

---

## Detection vs Classification 비교

| 항목 | Detection (본 실습) | Classification |
|------|---------------------|----------------|
| 모델 파일 | `yolo11n.pt` | `yolo11n-cls.pt` |
| 데이터 설정 | config.yaml 필요 | 폴더 구조만으로 충분 |
| 라벨 형식 | txt 파일 (좌표) | 폴더명 = 클래스 |
| 결과 저장 | `runs/detect/` | `runs/classify/` |
| 출력 형태 | `boxes` (바운딩 박스) | `probs` (클래스 확률) |
| 평가 지표 | mAP, Precision, Recall | Top-1, Top-5 Accuracy |
| 질문 | "어디에 무엇이 있는가?" | "이 이미지는 무엇?" |

---

## 주요 코드 스니펫

### 모델 학습
```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.train(
    data='../data/config.yaml',
    epochs=50,
    imgsz=640,
    batch=16
)
```

### 모델 검증
```python
model = YOLO('./runs/detect/train/weights/best.pt')
metrics = model.val(data='../data/config.yaml', split='test')

print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

### 단일 이미지 예측
```python
results = model.predict(image_path)
boxes = results[0].boxes

for box in boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])
```

---

## 안전 경고 시스템 로직

### 위험 수준 계산
```
위험 수준 = 안전모 미착용자 수 / 전체 인원 수
```

### 상태 판정 (기본값, 자유롭게 조절 가능)
| 위험 수준 | 상태 | 설명 |
|:---------:|:----:|:-----|
| 0% | SAFE | 모든 작업자 안전모 착용 |
| 1% ~ 30% | WARNING | 일부 미착용자 발견 |
| 30% 이상 | DANGER | 작업 중지 필요 |

### 시각화 색상
- **초록색 박스**: 안전모 착용 (helmet)
- **빨간색 박스**: 안전모 미착용 (head)

---

## 실습 흐름

```
1. 라이브러리 설치 (pip install ultralytics)
       ↓
2. Google Drive 마운트 & 경로 설정
       ↓
3. 데이터셋 확인 (900장, 2개 클래스)
       ↓
4. 모델 학습 (01_train.ipynb)
   - yolo11n.pt 로드
   - config.yaml로 데이터 설정
   - 50 epochs 학습
       ↓
5. 모델 평가 (02_test.ipynb)
   - mAP50 / mAP50-95
   - 샘플 이미지 테스트
       ↓
6. 안전 시스템 (03_safety_system.ipynb)
   - 이미지 입력 → 안전모 탐지
   - 위험 수준 계산 → 경고 표시
```

---

## 예상 학습 시간
- 모델 학습 (Colab GPU): 약 15-20분
- 전체 실습: 약 1-1.5시간

---

## 참고 자료
- [Ultralytics YOLO Detection 문서](https://docs.ultralytics.com/tasks/detect/)
- [Roboflow Hard Hat Workers Dataset](https://universe.roboflow.com/joseph-nelson/hard-hat-workers)
