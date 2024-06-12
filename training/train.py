import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, Normalize, ToPILImage
from PIL import Image
import numpy as np
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, Normalize
from argparse import ArgumentParser
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch import optim
from models.my_model import MyModel
from torchvision.transforms import ToTensor


class UCF101Dataset(Dataset):
    def __init__(self, root_dir, clip_len, split='1', train=True, transforms_=None, test_sample_num=10):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.split = split
        self.train = train
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        self.toPIL = ToPILImage()
        class_idx_path = os.path.join(root_dir, 'annotations/ucfTrainTestlist', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        if self.train:
            train_split_path = os.path.join(root_dir, 'annotations/ucfTrainTestlist',
                                            'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'annotations/ucfTrainTestlist', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
        print('Use split' + self.split)

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        class_idx = self.class_label2idx[videoname[:videoname.find('/')]] - 1 # 0부터 시작하도록
        filename = os.path.join(self.root_dir, 'UCF-101', videoname)

        cap = cv2.VideoCapture(filename)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        videodata = np.array(frames)
        # print("videodata.shape : ", videodata.shape)
        length, height, width, channel = videodata.shape

        if self.train:
            clip_start = random.randint(0, length - self.clip_len)
            clip = videodata[clip_start: clip_start + self.clip_len]

            if self.transforms_:
                trans_clip = []
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = Image.fromarray(frame)
                    frame = self.transforms_(frame)
                    trans_clip.append(frame)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                assert clip.shape[2] == 240 and clip.shape[
                    3] == 320, f"Unexpected frame size: {clip.shape[2:]} after transforms"
            else:
                clip = torch.tensor(clip)
                clip = clip.permute(3, 0, 1, 2)  # Change shape from (T, H, W, C) to (C, T, H, W)

            return clip, int(class_idx)
        else:
            all_clips = []
            all_idx = []
            for i in np.linspace(self.clip_len / 2, length - self.clip_len / 2, self.test_sample_num):
                clip_start = int(i - self.clip_len / 2)
                clip = videodata[clip_start: clip_start + self.clip_len]
                if self.transforms_:
                    trans_clip = []
                    seed = random.random()
                    for frame in clip:
                        random.seed(seed)
                        frame = Image.fromarray(frame)
                        frame = self.transforms_(frame)
                        trans_clip.append(frame)
                    clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                    assert clip.shape[2] == 240 and clip.shape[
                        3] == 320, f"Unexpected frame size: {clip.shape[2:]} after transforms"
                else:
                    clip = torch.tensor(clip)
                    clip = clip.permute(3, 0, 1, 2)  # Change shape from (T, H, W, C) to (C, T, H, W)
                all_clips.append(clip)
                all_idx.append(int(class_idx))

            return torch.stack(all_clips), torch.tensor(all_idx)


class FrameNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video):
        if isinstance(video, torch.Tensor):
            video = video.float()
        else:
            video = torch.tensor(np.array(video), dtype=torch.float32)

        if video.dim() == 4:  # (B, C, H, W) 배치 형태일 때
            for t in video:
                for c, (mean, std) in enumerate(zip(self.mean, self.std)):
                    t[c, :, :].sub_(mean).div_(std)
        elif video.dim() == 3:  # (C, H, W) 단일 이미지일 때
            for c, (mean, std) in enumerate(zip(self.mean, self.std)):
                video[c, :, :].sub_(mean).div_(std)
        else:
            raise ValueError("Unexpected video dimensions: {}".format(video.dim()))
        return video


def custom_collate_fn(batch):
    videos = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # print(f"Labels: {labels}")
    # print(f"Labels type: {[type(label) for label in labels]}")
    # print(f"Labels length: {len(labels)}")
    #
    # print(f"Shape of one frame: {videos[0].shape}")  # 채널 포함
    try:
        labels = torch.tensor(labels, dtype=torch.int64)
    except Exception as e:
        print(f"Error converting labels to tensor: {e}")
        raise e

    videos = torch.stack(videos)
    return videos, labels


# 학습 함수
def train_one_epoch(model, criterion, optimizer, data_loader, device, scaler):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(data_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size, num_clip, channels, height, width = inputs.size()

        inputs = inputs.view(batch_size * num_clip, channels, height, width)
        inputs = inputs.view(batch_size, num_clip, channels, height, width)  # 올바른 형태로 변환

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
    return running_loss / len(data_loader)


# 평가 함수
def evaluate(model, data_loader, device, scaler):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# 메인 함수
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        Resize((240, 320)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    full_dataset = UCF101Dataset(root_dir='./data', clip_len=16, split='1', train=True, transforms_=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    print(f"Train dataset size: {len(train_dataset)}")

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=custom_collate_fn, pin_memory=args.pin_memory, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             collate_fn=custom_collate_fn, pin_memory=args.pin_memory, persistent_workers=True)
    # train_loader에서 어떠한 데이터를 출력하는지 하나만 출력해보기
    # for inputs, labels in train_loader:
    #     print(f"Train Loader Inputs shape: {inputs.shape}")
    #     print(f"Train Loader Labels shape: {labels.shape}")
    #     break

    print("Initializing model...")
    # best_model.pth 파일이 없는 경우에만 모델을 초기화합니다.
    if not os.path.exists("best_model.pth"):
        model = MyModel(num_classes=101, top_k=5).to(device)
        print("Model initialized.")
    else:
        model = MyModel(num_classes=101, top_k=5)
        model.load_state_dict(torch.load("best_model.pth"))
        model.to(device)
        print("Model loaded from best_model.pth.")
    criterion = nn.CrossEntropyLoss()
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
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', action='store_true')
    args = parser.parse_args()
    main(args)