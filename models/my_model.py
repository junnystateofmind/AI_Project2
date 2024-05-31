import torch
import torch.nn as nn
import torchvision.models as models


class MyModel(nn.Module):
    def __init__(self, num_classes=101):
        super(MyModel, self).__init__()
        # EfficientNet-b4 model for RGB
        rgb_model = models.efficientnet_b0(weights=None)
        self.rgb_cnn = nn.Sequential(*list(rgb_model.children())[:-2])

        # EfficientNet-b4 model for Optical Flow
        flow_model = models.efficientnet_b0(weights=None)
        self.flow_cnn = nn.Sequential(*list(flow_model.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.lstm = nn.LSTM(3584, 512, 1, batch_first=True)  # Concatenated features from both CNNs
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # Debugging print statement

        batch_size, num_segments, num_frames, c, h, w = x.size()

        # Split channels into RGB and Optical Flow
        rgb_data = x[:, :, :, :3, :, :].contiguous()  # (batch_size, num_segments, num_frames, 3, h, w)
        flow_data = x[:, :, :, 3:, :, :].contiguous()  # (batch_size, num_segments, num_frames, 237, h, w)

        # Reshape for CNN input
        rgb_data = rgb_data.view(-1, 3, h, w)  # (batch_size * num_segments * num_frames, 3, h, w)
        flow_data = flow_data.view(-1, 237, h, w)  # (batch_size * num_segments * num_frames, 237, h, w)

        # Process through respective CNNs
        rgb_features = self.rgb_cnn(rgb_data)  # (batch_size * num_segments * num_frames, 1792, 7, 7)
        flow_features = self.flow_cnn(flow_data)  # (batch_size * num_segments * num_frames, 1792, 7, 7)

        # Global Average Pooling
        rgb_features = self.avgpool(rgb_features).view(batch_size, num_segments * num_frames,
                                                       -1)  # (batch_size, num_segments * num_frames, 1792)
        flow_features = self.avgpool(flow_features).view(batch_size, num_segments * num_frames,
                                                         -1)  # (batch_size, num_segments * num_frames, 1792)

        # Concatenate RGB and Optical Flow features
        features = torch.cat((rgb_features, flow_features), dim=2)  # (batch_size, num_segments * num_frames, 3584)

        # LSTM
        x, _ = self.lstm(features)  # (batch_size, num_segments * num_frames, 512)

        # Global Average Pooling over the time dimension
        x = x.mean(dim=1)  # (batch_size, 512)

        # Fully Connected Layer
        x = self.fc(x)  # (batch_size, num_classes)

        return x

from torchsummary import summary
# 모델 인스턴스 생성
model = MyModel(num_classes=101)

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델 요약 정보 출력
summary(model, input_size=(8, 10, 3, 224, 224), device=device.type)  # (batch_size, num_frames, channels, height, width)