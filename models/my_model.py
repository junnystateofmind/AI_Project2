import torch
import torch.nn as nn
from torchvision import models


class MyModel(nn.Module):
    def __init__(self, num_classes=101):
        super(MyModel, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=None)
        self.efficientnet.classifier = nn.Identity()  # Remove the final classifier layer
        self.lstm = nn.LSTM(1280, 512, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, num_segments, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)  # Reshape to (batch_size * num_segments * num_frames, 3, height, width)

        cnn_features = self.efficientnet(x)  # (batch_size * num_segments * num_frames, 1280)

        cnn_features = cnn_features.view(batch_size * num_segments, num_frames,
                                         -1)  # (batch_size * num_segments, num_frames, 1280)

        aggregated_features = cnn_features.mean(dim=1)  # Aggregate features (batch_size * num_segments, 1280)

        aggregated_features = aggregated_features.view(batch_size, num_segments, -1)  # (batch_size, num_segments, 1280)

        lstm_output, _ = self.lstm(aggregated_features)  # (batch_size, num_segments, 512)

        lstm_output = lstm_output.mean(dim=1)  # (batch_size, 512)

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