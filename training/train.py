import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import UCF101
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, Normalize
from models.my_model import MyModel
from tqdm import tqdm
from torchinfo import summary
from torch.cuda.amp import autocast, GradScaler

class FrameNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video):
        video = video.float()
        for t in video:
            for c, (mean, std) in enumerate(zip(self.mean, self.std)):
                t[c, :, :].sub_(mean).div_(std)
        return video

def custom_collate_fn(batch):
    videos = [item[0] for item in batch]
    labels = [item[2] for item in batch]
    videos = torch.stack(videos)
    labels = torch.tensor(labels)
    return videos, labels


def train_one_epoch(model, criterion, optimizer, data_loader, device, scaler):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(data_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
    return running_loss / len(data_loader)

def evaluate(model, data_loader, device, scaler):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = combine_optical_flow_channels(inputs)
            print(f"Input shape after combine_optical_flow_channels: {inputs.shape}")

            with autocast():
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
    full_dataset = UCF101(root='./data/UCF-101', annotation_path='./data/annotations/ucfTrainTestlist',
                          frames_per_clip=16, step_between_clips=1, fold=1, train=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=custom_collate_fn, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             collate_fn=custom_collate_fn, pin_memory=args.pin_memory)

    print("Initializing model...")
    model = MyModel(num_classes=101).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    best_accuracy = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1} starting...")
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device, scaler)
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")

        accuracy = evaluate(model, test_loader, device, scaler)
        print(f"Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved with accuracy: {:.2f}%".format(best_accuracy))

    print("Training complete. Best accuracy: {:.2f}%".format(best_accuracy))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--pin_memory', type=bool, default=False)
    args = parser.parse_args()

    main(args)