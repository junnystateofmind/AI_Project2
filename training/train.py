import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import UCF101
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, Normalize
from models.my_model import MyModel
from tqdm import tqdm


class FrameNormalize:
    def __init__(self, mean, std):
        self.normalize = Normalize(mean, std)

    def __call__(self, video):
        # 비디오의 각 프레임에 대해 정규화를 적용합니다.
        video = video.float()  # float 타입으로 변환
        return torch.stack([self.normalize(frame) for frame in video])


def train_one_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(data_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(data_loader)


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        Resize((224, 224)),
        FrameNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = UCF101(root='./data/UCF-101', annotation_path='./data/annotations/ucfTrainTestlist',
                           frames_per_clip=16, step_between_clips=1, fold=1, train=True, transform=transform)
    test_dataset = UCF101(root='./data/UCF-101', annotation_path='./data/annotations/ucfTrainTestlist',
                          frames_per_clip=16, step_between_clips=1, fold=1, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MyModel(num_classes=101).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_accuracy = 0.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")

        accuracy = evaluate(model, test_loader, device)
        print(f"Accuracy: {accuracy:.2f}%")

        # 가중치 저장
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved with accuracy: {:.2f}%".format(best_accuracy))

    print("Training complete. Best accuracy: {:.2f}%".format(best_accuracy))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    main(args)