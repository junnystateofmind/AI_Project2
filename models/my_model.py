import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, num_classes=101):
        super(MyModel, self).__init__()
        # EfficientNet-b4 모델 불러오기
        imagemodel = models.efficientnet_b4(pretrained=False) # (batch_size, num_frames, 3, 224, 224) -> (batch_size, num_frames, 1792, 7, 7)
        self.cnn = nn.Sequential(*list(imagemodel.children())[:-2]) # 마지막 두 레이어 제거
        self.avgpool = nn.AdaptiveAvgPool2d(1) # Global Average Pooling, 이후 차원은 (batch_size, num_frames, 1792, 7, 7) -> (batch_size, num_frames, 1792, 1, 1)
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

        # 마지막 타임스텝 출력 사용
        x = x[:, -1, :]  # (batch_size, 512)

        # Fully Connected Layer 통과
        x = self.fc(x)  # (batch_size, num_classes)

        return x