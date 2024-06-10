import torch
import torch.nn as nn
from torchvision import models
import timm

class MyModel(nn.Module):
    def __init__(self, num_classes=101):
        super(MyModel, self).__init__()
        # # EfficientNet-Lite0 모델 불러오기
        # self.image_model = timm.create_model('efficientnet_lite0', pretrained=False)
        # self.image_model.conv_stem = nn.Conv2d(240, 32, kernel_size=3, stride=2, padding=1, bias=False)  # Input 채널 변경
        # self.image_model.classifier = nn.Identity()  # 마지막 Fully Connected Layer 제거

        # GhostNet 모델 불러오기
        self.image_model = timm.create_model('ghostnet_100', pretrained=False)
        self.image_model.conv_stem = nn.Conv2d(240, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.image_model.classifier = nn.Identity()


        # Global Average Pooling 및 Fully Connected 레이어 정의
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1280, num_classes)

        # LSTM 및 최종 Fully Connected 레이어 정의
        self.lstm = nn.LSTM(num_classes, 128, batch_first=True)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()

        # (batch_size, num_frames, channels, height, width) -> (batch_size * num_frames, channels, height, width)
        x = x.view(-1, channels, height, width)

        # CNN 통과
        cnn_features = self.image_model(x)  # (batch_size * num_frames, 1280, 7, 7)

        # cnn_features가 4차원인지 확인하고 변환
        if cnn_features.dim() == 2:
            cnn_features = cnn_features.unsqueeze(-1).unsqueeze(-1)

        # Global Average Pooling 적용
        cnn_features = self.gap(cnn_features)  # (batch_size * num_frames, 1280, 1, 1)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)  # (batch_size * num_frames, 1280)

        # Fully Connected Layer 통과
        cnn_features = self.fc1(cnn_features)  # (batch_size * num_frames, 101)

        # (batch_size, num_frames, 101)로 변환
        cnn_features = cnn_features.view(batch_size, num_frames, -1)

        # LSTM 통과
        lstm_output, _ = self.lstm(cnn_features)  # (batch_size, num_frames, 128)

        # LSTM 출력에 대해 Global Average Pooling 적용
        lstm_output = lstm_output.mean(dim=1)  # (batch_size, 128)

        # 최종 Fully Connected Layer 통과
        output = self.fc2(lstm_output)  # (batch_size, num_classes)

        return output

# 모델 인스턴스 생성
model = MyModel(num_classes=101)

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델 요약 정보 출력
from torchinfo import summary
summary(model, input_size=(8, 16, 240, 112, 112), device=device.type)  # (batch_size, num_frames, channels, height, width)