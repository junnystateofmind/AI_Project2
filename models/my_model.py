import torch
import torch.nn as nn
from torchvision import models

class MyModel(nn.Module):
    def __init__(self, num_classes=101):
        super(MyModel, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=None)
        self.efficientnet.classifier = nn.Identity()  # 최종 분류 레이어 제거
        self.reduce_channels = nn.Conv2d(1280, 512, kernel_size=1)  # 채널 수 줄이기
        self.lstm = nn.LSTM(512, 256, batch_first=True)  # LSTM 입력 크기 조정
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, num_segments, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)  # (batch_size * num_segments * num_frames, 3, height, width)

        cnn_features = self.efficientnet(x)  # (batch_size * num_segments * num_frames, 1280)
        cnn_features = self.reduce_channels(cnn_features)  # (batch_size * num_segments * num_frames, 512, H, W)
        cnn_features = cnn_features.mean([2, 3])  # Global Average Pooling (batch_size * num_segments * num_frames, 512)

        cnn_features = cnn_features.view(batch_size * num_segments, num_frames, -1)  # (batch_size * num_segments, num_frames, 512)
        aggregated_features = cnn_features.mean(dim=1)  # (batch_size * num_segments, 512)

        aggregated_features = aggregated_features.view(batch_size, num_segments, -1)  # (batch_size, num_segments, 512)
        lstm_output, _ = self.lstm(aggregated_features)  # (batch_size, num_segments, 256)
        lstm_output = lstm_output.mean(dim=1)  # (batch_size, 256)

        output = self.fc(lstm_output)  # (batch_size, num_classes)
        return output


from torchsummary import summary
# 모델 인스턴스 생성
model = MyModel(num_classes=101)

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델 요약 정보 출력
summary(model, input_size=(8, 10, 3, 224, 224), device=device.type)  # (batch_size, num_frames, channels, height, width)