import os
from torchvision.datasets.ucf101 import UCF101
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader

# 데이터셋 경로 설정
root = "./UCF101"
annotation_path = "./UCF101/annotations"

# 전처리 변환 설정
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
])

# 데이터셋 초기화
ucf101_train = UCF101(
    root=root,
    annotation_path=annotation_path,
    frames_per_clip=16,
    step_between_clips=1,
    fold=1,
    train=True,
    transform=transform
)

ucf101_test = UCF101(
    root=root,
    annotation_path=annotation_path,
    frames_per_clip=16,
    step_between_clips=1,
    fold=1,
    train=False,
    transform=transform
)

# DataLoader 설정
train_loader = DataLoader(ucf101_train, batch_size=4, shuffle=True, num_workers=2)
test_loader = DataLoader(ucf101_test, batch_size=4, shuffle=False, num_workers=2)

# 데이터 로드 예제
for batch in train_loader:
    videos, audios, labels = batch
    print(f"비디오 크기: {videos.size()}")
    print(f"오디오 크기: {audios.size()}")
    print(f"레이블: {labels}")
    break

