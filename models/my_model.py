import torch
import torch.nn as nn
from torchvision import models


class MyModel(nn.Module):
    def __init__(self, num_classes=101):
        super(MyModel, self).__init__()
        # EfficientNet-B0 모델 불러오기
        self.efficientnet = models.efficientnet_b0(weights=None)
        self.efficientnet.classifier = nn.Identity()  # Remove the final classifier layer
        self.lstm = nn.LSTM(1280, 512, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, num_segments, num_frames, channels, height, width = x.size()
        # Reshape to (batch_size * num_segments * num_frames * 80, 3, height, width)
        x = x.view(-1, 3, height, width)

        # Process each channel with EfficientNet-B0
        cnn_features = self.efficientnet(x)  # Output shape: (batch_size * num_segments * num_frames * 80, 1280)

        # Reshape to (batch_size * num_segments * num_frames, 80, 1280)
        cnn_features = cnn_features.view(batch_size * num_segments * num_frames, -1, 1280)

        # Aggregate the features from 80 channels using mean
        aggregated_features = torch.mean(cnn_features, dim=1)  # Shape: (batch_size * num_segments * num_frames, 1280)

        # Reshape to (batch_size, num_segments * num_frames, 1280)
        aggregated_features = aggregated_features.view(batch_size, num_segments * num_frames, 1280)

        # LSTM
        lstm_output, _ = self.lstm(aggregated_features)  # Shape: (batch_size, num_segments * num_frames, 512)

        # Average pooling over frames
        lstm_output = lstm_output.mean(dim=1)  # Shape: (batch_size, 512)


from torchsummary import summary
# 모델 인스턴스 생성
model = MyModel(num_classes=101)

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델 요약 정보 출력
summary(model, input_size=(8, 10, 3, 224, 224), device=device.type)  # (batch_size, num_frames, channels, height, width)