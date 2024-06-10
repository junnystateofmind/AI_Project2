import torch
import torch.nn as nn
from torchvision import transforms
import timm

class MyModel(nn.Module):
    def __init__(self, num_classes=101):
        super(MyModel, self).__init__()

        # GhostNet 모델 불러오기
        self.image_model = timm.create_model('ghostnet_100', pretrained=False)
        self.image_model.conv_stem = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.image_model.classifier = nn.Identity()

        # MLP 기반 중요도 계산
        self.mlp = nn.Sequential(
            nn.Linear(112*112*3, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Global Average Pooling 및 Fully Connected 레이어 정의
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1280, num_classes)

        # LSTM 및 최종 Fully Connected 레이어 정의
        self.lstm = nn.LSTM(num_classes, 128, batch_first=True)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()

        # MLP로 중요도 계산
        x = x.view(batch_size * num_frames, -1)
        scores = self.mlp(x)  # (batch_size * num_frames, 1)
        scores = scores.view(batch_size, num_frames)

        # 상위 K개의 프레임 선택
        top_k = 5
        _, selected_indices = torch.topk(scores, top_k, dim=1, largest=True, sorted=False)

        # 선택된 프레임을 기반으로 이미지 모델 통과
        selected_features = []
        for i in range(batch_size):
            selected_frames = x[i, selected_indices[i], :].view(top_k, channels, height, width)
            cnn_features = self.image_model(selected_frames)
            cnn_features = self.gap(cnn_features)
            cnn_features = cnn_features.view(cnn_features.size(0), -1)
            selected_features.append(cnn_features)

        cnn_features = torch.stack(selected_features)
        cnn_features = self.fc1(cnn_features)

        # (batch_size, num_frames, 101)로 변환
        cnn_features = cnn_features.view(batch_size, top_k, -1)

        # LSTM 통과
        lstm_output, _ = self.lstm(cnn_features)

        # LSTM 출력에 대해 Global Average Pooling 적용
        lstm_output = lstm_output.mean(dim=1)

        # 최종 Fully Connected Layer 통과
        output = self.fc2(lstm_output)

        return output

# 모델 인스턴스 생성
model = MyModel(num_classes=101)

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델 요약 정보 출력
from torchinfo import summary
summary(model, input_size=(8, 16, 3, 112, 112), device=device.type)  # (batch_size, num_frames, channels, height, width)