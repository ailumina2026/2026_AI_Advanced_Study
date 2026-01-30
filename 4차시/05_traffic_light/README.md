# 05. 교통 신호 탐지 기반 자율주행 신호 해석 시스템

## 개요
YOLO Detection 모델을 활용하여 **교통 신호등 이미지**에서 신호를 자동 탐지하고 자율주행 동작을 결정하는 AI 시스템을 구축합니다.

---

## 학습 목표
1. YOLO Detection 모델의 학습 방법 이해
2. 다중 클래스 객체 탐지(Object Detection) 태스크 학습
3. 학습된 모델을 활용한 자율주행 신호 해석 시스템 구현

---

## 데이터셋

### 출처
- Traffic Human Detection by DA150X Pro Gamer (https://universe.roboflow.com/da150x-pro-gamer/traffic-human-detection)
  - 원본 11개 클래스에서 교통 신호등 클래스만 추출하여 사용

### 클래스 (6개)
| 클래스 ID | 영문명 | 한글명 | 자율주행 동작 |
|:---------:|:------:|:------:|:-------------|
| 0 | green | 초록불 | GO (진행) |
| 1 | green_left | 초록 좌회전 | LEFT (좌회전 가능) |
| 2 | red | 빨간불 | STOP (정지) |
| 3 | red_left | 빨간 좌회전 | STOP_LEFT (좌회전 금지) |
| 4 | yellow | 노란불 | CAUTION (서행) |
| 5 | yellow_left | 노란 좌회전 | CAUTION_LEFT (좌회전 주의) |

### 데이터 분할
| 분할 | 이미지 수 | 비율 |
|:----:|:---------:|:----:|
| train | 272장 | 80% |
| val | 34장 | 10% |
| test | 34장 | 10% |
| **총** | **340장** | 100% |

### 폴더 구조
```
data/
├── config.yaml           # 데이터셋 설정 파일 (Detection 필수!)
├── train/
│   ├── images/           (272장)
│   └── labels/           (바운딩 박스 좌표)
├── val/
│   ├── images/           (34장)
│   └── labels/
├── test/
│   ├── images/           (34장)
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
| 4 | `"../runs/_______/train/"` | `detect` | Detection 결과 폴더 |

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
| 1 | `MODEL_PATH = _______` | `'../runs/detect/train/weights/best.pt'` | 학습된 모델 경로 |
| 2 | `model._______()` | `val` | 검증 메서드 |
| 3 | `results[0]._______` | `boxes` | Detection 결과 (바운딩 박스) |

---

### 03_signal_system.ipynb - 신호 해석 시스템
교통 신호를 탐지하고 자율주행 동작을 결정하는 시스템을 구현합니다.

**주요 내용:**
- 신호 분석 함수 구현
- 자율주행 동작 매핑 (GO/STOP/CAUTION/LEFT/STOP_LEFT/CAUTION_LEFT)
- 결과 시각화

**빈칸 (4개):**
| # | 위치 | 정답 | 힌트 |
|---|------|------|------|
| 1 | `boxes = result._______` | `boxes` | Detection 결과 (바운딩 박스) |
| 2 | `cls_id = int(box._______[0])` | `cls` | 바운딩 박스의 클래스 ID |
| 3 | `action, message, color = _______` | `get_driving_action(cls_id)` | 클래스별 동작 매핑 함수 |
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
model = YOLO('../runs/detect/train/weights/best.pt')
metrics = model.val(data='../data/config.yaml', split='test')

print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

### 신호 탐지 및 동작 결정
```python
results = model.predict(image_path)
boxes = results[0].boxes

for box in boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])

    # 동작 결정
    action, message, color = get_driving_action(cls_id)
    print(f"신호: {model.names[cls_id]}, 동작: {action}")
```

---

## 자율주행 신호 해석 로직

### 동작 매핑
```python
def get_driving_action(signal_class):
    actions = {
        0: ('GO', '🟢 진행하세요', (0, 255, 0)),                    # green
        1: ('LEFT', '🟢⬅️ 좌회전 가능', (0, 200, 255)),            # green_left
        2: ('STOP', '🔴 정지하세요', (255, 0, 0)),                  # red
        3: ('STOP_LEFT', '🔴⬅️ 좌회전 금지', (255, 100, 100)),     # red_left
        4: ('CAUTION', '🟡 서행하세요', (255, 255, 0)),             # yellow
        5: ('CAUTION_LEFT', '🟡⬅️ 좌회전 주의', (255, 200, 0))     # yellow_left
    }
    return actions.get(signal_class, ('UNKNOWN', '❓ 신호 불명', (128, 128, 128)))
```

### 우선순위 로직
- **안전 우선**: 빨간불(STOP, STOP_LEFT)이 탐지되면 무조건 정지
- **신뢰도 기준**: 같은 우선순위 내에서는 가장 높은 confidence 선택
- **좌회전 신호**: 색상별로 좌회전 허용/금지/주의 표시

---

## 실습 흐름

```
1. 라이브러리 설치 (pip install ultralytics)
       ↓
2. Google Drive 마운트 & 경로 설정
       ↓
3. 데이터셋 확인 (340장, 6개 클래스, 80:10:10 분할)
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
6. 신호 해석 시스템 (03_signal_system.ipynb)
   - 이미지 입력 → 신호 탐지
   - 신호 → 자율주행 동작 결정
   - 결과 시각화
```

---

## 참고 자료
- [Ultralytics YOLO Detection 문서](https://docs.ultralytics.com/tasks/detect/)
- [Traffic Human Detection - Roboflow](https://universe.roboflow.com/da150x-pro-gamer/traffic-human-detection)
