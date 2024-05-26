import torch
import torch.nn as nn
import torchvision.models as models


class MyModel(nn.Module):
    def __init__(self, num_classes=101):
        super(MyModel, self).__init__()
        imagemodel = models.efficientnet_b0(pretrained=False)
        self.cnn = nn.Sequential(*list(imagemodel.children())[:-2])  # 마지막 두 레이어 제거 (AdaptiveAvgPool2d와 Linear 레이어)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling 추가
        self.lstm = nn.LSTM(input_size=1280, hidden_size=512, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()
        cnn_out = []
        for t in range(seq_length):
            with torch.no_grad():
                c_out = self.cnn(x[:, t, :, :, :])
                c_out = self.avgpool(c_out)  # Global Average Pooling 적용
                c_out = c_out.view(batch_size, -1)
            cnn_out.append(c_out)

        cnn_out = torch.stack(cnn_out, dim=1)
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = lstm_out[:, -1, :]  # LSTM의 마지막 출력을 가져옴
        out = self.fc(lstm_out)
        return out