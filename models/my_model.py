import torch
import torch.nn as nn
import timm
from torchinfo import summary

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

        # LSTM 및 Fully Connected 레이어 정의
        self.lstm = nn.LSTM(1280, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()

        # (batch_size, num_frames, channels, height, width) -> (batch_size * num_frames, channels, height, width)
        x = x.view(-1, channels, height, width)

        # CNN 통과
        cnn_features = self.efficientnet(x)  # (batch_size * num_frames, 1280)

        # (batch_size, num_frames, 1280)로 변환
        cnn_features = cnn_features.view(batch_size, num_frames, -1)

        # LSTM 통과
        lstm_output, _ = self.lstm(cnn_features)  # (batch_size, num_frames, 128)

        # Global Average Pooling 적용
        lstm_output = lstm_output.mean(dim=1)  # (batch_size, 128)

        # Fully Connected Layer 통과
        output = self.fc(lstm_output)  # (batch_size, num_classes)

        return output

# 모델 인스턴스 생성
model = MyModel(num_classes=101)

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델 요약 정보 출력
summary(model, input_size=(8, 16, 240, 224, 224), device=device.type)  # (batch_size, num_frames, channels, height, width)