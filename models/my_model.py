import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class MyModel(nn.Module):
    def __init__(self, num_classes=101):
        super(MyModel, self).__init__()
        # EfficientNet-Lite 모델 불러오기
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-lite0')
        self.efficientnet._fc = nn.Identity()  # 마지막 분류기 레이어 제거

        # LSTM 및 Fully Connected 레이어 정의
        self.lstm = nn.LSTM(1280, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)
        cnn_features = self.efficientnet(x)  # (batch_size * num_frames, 1280)
        cnn_features = cnn_features.view(batch_size, num_frames, -1)
        lstm_output, _ = self.lstm(cnn_features)  # (batch_size, num_frames, 128)
        lstm_output = lstm_output.mean(dim=1)  # (batch_size, 128)
        output = self.fc(lstm_output)  # (batch_size, num_classes)
        return output

# 모델 인스턴스 생성 및 요약 정보 출력
model = MyModel(num_classes=101)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
summary(model, input_size=(8, 10, 3, 224, 224), device=device.type)  # (batch_size, num_frames, channels, height, width)