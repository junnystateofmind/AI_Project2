import torch
import torch.nn as nn
from torchvision import models
import timm


class MyModel(nn.Module):
    def __init__(self, num_classes=101):
        super(MyModel, self).__init__()
        # EfficientNet-Lite0 모델 불러오기
        self.efficientnet = timm.create_model('efficientnet_lite0', pretrained=True)
        self.efficientnet.conv_stem = nn.Conv2d(240, 32, kernel_size=3, stride=2, padding=1, bias=False)  # Input 채널 변경
        self.efficientnet.classifier = nn.Identity()  # 마지막 Fully Connected Layer 제거

        # EfficientNet 레이어를 동결
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # 1x1 컨볼루션 레이어 추가 (차원 축소용)
        connected_dimension = 320
        self.conv1x1 = nn.Conv2d(1280, 640, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(connected_dimension)
        self.relu = nn.ReLU()

        # LSTM 및 Fully Connected 레이어 정의
        self.lstm = nn.LSTM(input_size=connected_dimension, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()

        # (batch_size, num_frames, channels, height, width) -> (batch_size * num_frames, channels, height, width)
        x = x.view(-1, channels, height, width)

        # CNN 통과
        cnn_features = self.efficientnet(x)  # (batch_size * num_frames, 1280, 7, 7)

        # 차원 축소 (1x1 컨볼루션 레이어 통과)
        cnn_features = self.relu(self.bn(self.conv1x1(cnn_features)))  # (batch_size * num_frames, 640, 7, 7)

        # Global Average Pooling
        cnn_features = torch.mean(cnn_features, dim=[2, 3])  # (batch_size * num_frames, 640)

        # (batch_size, num_frames, 640)로 변환
        cnn_features = cnn_features.view(batch_size, num_frames, -1)

        # LSTM 통과
        lstm_output, _ = self.lstm(cnn_features)  # (batch_size, num_frames, 128)

        # LSTM 출력에 대해 Global Average Pooling 적용
        lstm_output = lstm_output.mean(dim=1)  # (batch_size, 128)

        # Fully Connected Layer 통과
        output = self.fc(lstm_output)  # (batch_size, num_classes)

        return output