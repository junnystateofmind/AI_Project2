import torch
import torch.nn as nn
from torchvision import models


class MyModel(nn.Module):
    def __init__(self, num_classes=101):
        super(MyModel, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.features[0][0] = nn.Conv2d(240, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        for param in self.efficientnet.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(1280, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)
        cnn_features = self.efficientnet(x)
        cnn_features = cnn_features.view(batch_size, num_frames, -1)
        lstm_output, _ = self.lstm(cnn_features)
        lstm_output = lstm_output.mean(dim=1)
        output = self.fc(lstm_output)
        return output

from torchinfo import summary

# 모델 인스턴스 생성
model = MyModel(num_classes=101)

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델 요약 정보 출력
summary(model, input_size=(8, 10, 3, 224, 224), device=device.type)  # (batch_size, num_frames, channels, height, width)