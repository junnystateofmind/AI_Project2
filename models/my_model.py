import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary


class MyModel(nn.Module):
    def __init__(self, num_classes=101):
        super(MyModel, self).__init__()
        # EfficientNet-b4 모델 불러오기
        imagemodel = models.efficientnet_b4(weights=None)
        self.cnn = nn.Sequential(*list(imagemodel.children())[:-2])  # 마지막 두 레이어 제거
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling, 이후 차원은 (batch_size, 1792, 1, 1)
        self.lstm = nn.LSTM(1792, 512, 1, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.size()
        x = x.view(batch_size * num_frames, c, h, w)  # Reshape to (batch_size * num_frames, 3, 224, 224)

        # CNN 통과
        x = self.cnn(x)  # (batch_size * num_frames, 1792, 7, 7)

        # Global Average Pooling
        x = self.avgpool(x)  # (batch_size * num_frames, 1792, 1, 1)
        x = x.view(batch_size, num_frames, -1)  # (batch_size, num_frames, 1792)

        # LSTM 통과
        x, _ = self.lstm(x)  # (batch_size, num_frames, 512)

        # LSTM 출력에 대해 Global Average Pooling 적용
        x = x.mean(dim=1)  # (batch_size, 512)

        # Fully Connected Layer 통과
        x = self.fc(x)  # (batch_size, num_classes)

        return x

# 모델 인스턴스 생성
model = MyModel(num_classes=101)

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델 요약 정보 출력
summary(model, input_size=(8, 10, 3, 224, 224), device=device.type)  # (batch_size, num_frames, channels, height, width)