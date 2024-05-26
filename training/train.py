import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import UCF101
from argparse import ArgumentParser

from models.my_model import MyModel
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class VideoTransform:
    def __init__(self, resize, normalize):
        self.resize = resize
        self.normalize = normalize

    def __call__(self, video):
        transformed_frames = []
        for frame in video:
            frame = self.resize(frame).float()  # float 타입으로 변환
            frame = self.normalize(frame)
            transformed_frames.append(frame)
        return torch.stack(transformed_frames)

transform = VideoTransform(
    resize=Resize((224, 224)),
    normalize=Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)

parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10)

args = parser.parse_args()

model = MyModel(num_classes=101).cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

train_dataset = UCF101(root='./data/UCF-101', annotation_path='./data/annotations/ucfTrainTestlist', frames_per_clip=16, step_between_clips=1, fold=1, train=True, transform=transform)
test_dataset = UCF101(root='./data/UCF-101', annotation_path='./data/annotations/ucfTrainTestlist', frames_per_clip=16, step_between_clips=1, fold=1, train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total} %")