"""
GTSRB 데이터셋 다운로드 및 저장 스크립트

Hugging Face에서 GTSRB 데이터셋을 다운로드하고
train/val/test로 분할하여 로컬에 저장합니다.
"""

import random
from pathlib import Path
from collections import Counter
from datasets import load_dataset
from PIL import Image

# 시드 설정
random.seed(42)

# 경로 설정
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'images'
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
TEST_DIR = DATA_DIR / 'test'

# 디렉토리 생성
for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print("📥 GTSRB 데이터셋 다운로드 중...")
print("(첫 실행 시 다운로드에 시간이 걸릴 수 있습니다)\n")

# GTSRB 데이터셋 로드
try:
    dataset = load_dataset("tanganke/gtsrb")
    print(f"✅ 데이터셋 로드 완료!")
    print(f"   - Train: {len(dataset['train'])}장")
    if 'test' in dataset:
        print(f"   - Test: {len(dataset['test'])}장")
except Exception as e:
    print(f"첫 번째 데이터셋 로드 실패: {e}")
    print("대체 데이터셋 시도...")
    try:
        dataset = load_dataset("bazyl/GTSRB")
        print(f"✅ 대체 데이터셋 로드 완료!")
    except Exception as e2:
        print(f"대체 데이터셋도 실패: {e2}")
        raise Exception("GTSRB 데이터셋을 로드할 수 없습니다. Hugging Face Hub 연결을 확인하세요.")

# 클래스 정보 확인 및 상위 5개 클래스 선택
print("\n📊 클래스 정보 분석 중...")
train_labels = [item['label'] for item in dataset['train']]
label_counts = Counter(train_labels)
selected_classes = [label for label, count in label_counts.most_common(5)]

print(f"\n선택된 5개 클래스:")
for idx, cls in enumerate(selected_classes):
    count = label_counts[cls]
    print(f"  - GTSRB 클래스 {cls}: {count}개 샘플 → Class {idx}로 저장")

# 레이블 매핑 (로컬에서만 사용)
label_mapping = {old_label: new_label for new_label, old_label in enumerate(selected_classes)}

# 선택된 클래스 필터링 및 클래스별 샘플 수집
print("\n🔍 데이터 필터링 및 수집 중...")
samples_by_class = {i: [] for i in range(5)}

for item in dataset['train']:
    if item['label'] in selected_classes:
        new_label = label_mapping[item['label']]
        samples_by_class[new_label].append({
            'image': item['image'],
            'label': new_label,
            'original_label': item['label']
        })

# 각 클래스당 240개씩 샘플링 및 분할
print("\n📦 데이터 분할 및 저장 중...")
train_data, val_data, test_data = [], [], []

for label in range(5):
    samples = samples_by_class[label]
    random.shuffle(samples)
    samples = samples[:240]  # 클래스당 240개

    train_data.extend(samples[:192])    # 192개
    val_data.extend(samples[192:216])   # 24개
    test_data.extend(samples[216:240])  # 24개

print(f"  - Train: {len(train_data)}장")
print(f"  - Val: {len(val_data)}장")
print(f"  - Test: {len(test_data)}장")

# 클래스별 디렉토리 생성
for label in range(5):
    (TRAIN_DIR / f'class_{label}').mkdir(exist_ok=True)
    (VAL_DIR / f'class_{label}').mkdir(exist_ok=True)
    (TEST_DIR / f'class_{label}').mkdir(exist_ok=True)

# 이미지 저장 함수
def save_images(data_list, base_dir, split_name):
    """이미지를 디렉토리에 저장"""
    print(f"\n💾 {split_name} 이미지 저장 중...")

    # 클래스별 카운터
    class_counters = {i: 0 for i in range(5)}

    for item in data_list:
        image = item['image'].convert('RGB')
        label = item['label']

        # 파일명 생성
        filename = f"{class_counters[label]:04d}.jpg"
        save_path = base_dir / f'class_{label}' / filename

        # 이미지 저장
        image.save(save_path)
        class_counters[label] += 1

    print(f"  ✅ {split_name} 저장 완료!")
    for label in range(5):
        print(f"     - class_{label}: {class_counters[label]}장")

# 이미지 저장
save_images(train_data, TRAIN_DIR, "Train")
save_images(val_data, VAL_DIR, "Val")
save_images(test_data, TEST_DIR, "Test")

# 데이터 준비 완료
print(f"\n✅ 데이터 준비 완료!")

# 최종 구조 출력
print("\n" + "="*60)
print("📂 최종 데이터 구조:")
print("="*60)
print("""
data/
├── images/
│   ├── train/
│   │   ├── class_0/  (192장)
│   │   ├── class_1/  (192장)
│   │   ├── class_2/  (192장)
│   │   ├── class_3/  (192장)
│   │   └── class_4/  (192장)
│   ├── val/
│   │   ├── class_0/  (24장)
│   │   ├── class_1/  (24장)
│   │   ├── class_2/  (24장)
│   │   ├── class_3/  (24장)
│   │   └── class_4/  (24장)
│   └── test/
│       ├── class_0/  (24장)
│       ├── class_1/  (24장)
│       ├── class_2/  (24장)
│       ├── class_3/  (24장)
│       └── class_4/  (24장)
""")

total_images = len(train_data) + len(val_data) + len(test_data)
print("\n✅ 데이터셋 준비 완료!")
print(f"   총 {total_images}장의 이미지가 저장되었습니다.")
print(f"\n다음 단계: 노트북(01_train.ipynb)에서 로컬 데이터를 사용하여 학습")