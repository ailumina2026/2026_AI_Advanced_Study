# 🖐️ Hand Pose Estimation

이 프로젝트는 2D Hand Pose Estimation(손 관절 추정)을 위한 딥러닝 모델을 학습하고 테스트하는 교육 자료입니다.<br>
PyTorch를 사용하여 데이터 로드부터 모델 학습, 그리고 시각화까지의 전체 과정을 실습합니다.

## 📂 Directory Structure

- **code/**
  - `01_train.ipynb`: 데이터셋 로드 및 CNN 모델 학습
  - `02_test.ipynb`: 학습된 모델 평가 및 지표 확인
  - `03_overlay.ipynb`: 손 관절 스켈레톤 시각화 및 응용
- **data/**
  - 학습 및 테스트를 위한 손 이미지 데이터셋

## 🚀 Getting Started

### 1. Prerequisites (환경 설정)
- Python 3.8+
- PyTorch, Torchvision
- OpenCV, Matplotlib, Numpy

### 2. Dataset
본 실습에서는 [FreiHAND](https://lmb.informatik.uni-freiburg.de/resources/datasets/Freihand.en.html) 데이터셋의 일부를 가공하여 사용합니다. `data/` 폴더 내의 가이드를 따라주세요.

## 🎓 Curriculum Goals
1. **Data Loading**: 이미지와 좌표(Keypoints) 데이터를 쌍으로 불러오는 방법 이해
2. **Transfer Learning**: Pre-trained 모델(ResNet)을 활용한 미세 조정(Fine-tuning)
3. **Visualization**: 모델의 예측값을 이미지 위에 시각화하는 방법 습득