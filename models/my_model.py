import torch
import torch.nn as nn
from torchvision import models
import timm


# 모델 정의
class MyModel(nn.Module):
    def __init__(self, num_classes=101, top_k=5, height=240, width=320):
        super(MyModel, self).__init__()
        self.top_k = top_k
        self.height = height
        self.width = width

        # GhostNet 모델 불러오기
        self.image_model = timm.create_model('ghostnet_100', pretrained=False)
        self.image_model.conv_stem = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.image_model.classifier = nn.Identity()

        # Global Average Pooling 및 Fully Connected 레이어 정의
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1280, 101)

        # LSTM 및 최종 Fully Connected 레이어 정의
        self.lstm = nn.LSTM(101, 128, batch_first=True)
        self.fc2 = nn.Linear(128, num_classes)

        # MLP for frame selection
        self.mlp = nn.Sequential(
            nn.Linear(height * width * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()  # x는 (batch_size, num_frames, channels, height, width)

        assert height == self.height and width == self.width, \
            f"Input height and width must match the initialized values ({self.height}, {self.width})"

        # 각 프레임의 중요도를 계산합니다.
        importance_scores = []
        for i in range(num_frames):
            frame = x[:, i, :, :, :]  # 각 배치의 i번째 프레임을 가져옵니다.
            frame = frame.view(batch_size, -1)  # (batch_size, channels * height * width)로 변환
            score = self.mlp(frame)  # MLP를 통해 중요도 계산
            importance_scores.append(score)
        importance_scores = torch.stack(importance_scores, dim=1).squeeze(-1)  # (batch_size, num_frames)

        # top_k 프레임 선택
        _, selected_indices = torch.topk(importance_scores, self.top_k, dim=1)

        selected_frames = []
        for i in range(batch_size):
            selected_frames.append(x[i, selected_indices[i], :, :, :])  # 각 배치에서 선택된 프레임을 추가
        selected_frames = torch.stack(selected_frames)  # (batch_size, top_k, channels, height, width)

        # CNN 통과
        selected_frames = selected_frames.view(-1, channels, height, width)
        cnn_features = self.image_model(selected_frames)  # (batch_size * top_k, 1280, 7, 7)

        # Global Average Pooling 적용
        if cnn_features.dim() == 4:
            cnn_features = self.gap(cnn_features)  # (batch_size * top_k, 1280, 1, 1)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)  # (batch_size * top_k, 1280)

        # Fully Connected Layer 통과
        cnn_features = self.fc1(cnn_features)  # (batch_size * top_k, 101)

        # (batch_size, top_k, 101)로 변환
        cnn_features = cnn_features.view(batch_size, self.top_k, -1)

        # LSTM 통과
        lstm_output, _ = self.lstm(cnn_features)  # (batch_size, top_k, 128)

        # LSTM 출력에 대해 Global Average Pooling 적용
        lstm_output = lstm_output.mean(dim=1)  # (batch_size, 128)

        # 최종 Fully Connected Layer 통과
        output = self.fc2(lstm_output)  # (batch_size, num_classes)

        return output

# 모델 인스턴스 생성
model = MyModel(num_classes=101, top_k=5)

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델 요약 정보 출력
from torchinfo import summary

summary(model, input_size=(8, 16, 3, 240, 320))