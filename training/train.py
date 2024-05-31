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
        self.mean = mean
        self.std = std

    def __call__(self, video):
        video = video.float()  # Convert to float
        for t in video:  # 각 프레임을 순회하며 정규화
            for c, (mean, std) in enumerate(zip(self.mean, self.std)):
                t[c, :, :].sub_(mean).div_(std)
        return video


def custom_collate_fn(batch):
    # 비디오와 레이블만 남기고 나머지 요소는 무시
    videos = [item[0] for item in batch]
    labels = [item[2] for item in batch]
    videos = torch.stack(videos)
    labels = torch.tensor(labels)
    return videos, labels


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

    print("Loading datasets...")
    train_dataset = UCF101(root='./data/UCF-101', annotation_path='./data/annotations/ucfTrainTestlist',
                           frames_per_clip=16, step_between_clips=1, fold=1, train=True, transform=transform)
    test_dataset = UCF101(root='./data/UCF-101', annotation_path='./data/annotations/ucfTrainTestlist',
                          frames_per_clip=16, step_between_clips=1, fold=1, train=False, transform=transform)

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=custom_collate_fn, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             collate_fn=custom_collate_fn, pin_memory=args.pin_memory)

    print("Initializing model...")
    model = MyModel(num_classes=101).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 데이터셋의 반환 값 확인
    sample = train_dataset[0]
    print(f"Sample type: {type(sample)}")
    print(f"Sample length: {len(sample)}")

    if len(sample) == 2:
        sample_data, sample_label = sample
    elif len(sample) == 3:
        sample_data, _, sample_label = sample
        sample_data = sample_data[:, :3, :, :]  # RGB 채널만 사용
    else:
        print(f"Unexpected sample format: {sample}")
        return

    print("Sample data shape:", sample_data.shape)
    print("Sample label:", sample_label)

    best_accuracy = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1} starting...")
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
    parser.add_argument('--batch_size', type=int, default=4)  # 배치 크기를 줄임
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)  # 워커 수를 줄임
    parser.add_argument('--pin_memory', type=bool, default=False)  # pin_memory 비활성화
    args = parser.parse_args()

    main(args)