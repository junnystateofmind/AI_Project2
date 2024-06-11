import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, Normalize, ToPILImage
from PIL import Image
import numpy as np
import cv2

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
        class_idx = self.class_label2idx[videoname[:videoname.find('/')]]
        filename = os.path.join(self.root_dir, 'UCF-101', videoname)

        # Use OpenCV to read the video
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
        length, height, width, channel = videodata.shape

        if self.train:
            clip_start = random.randint(0, length - self.clip_len)
            clip = videodata[clip_start: clip_start + self.clip_len]

            if self.transforms_:
                trans_clip = []
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame)
                    frame = self.transforms_(frame)
                    trans_clip.append(frame)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
            else:
                clip = torch.tensor(clip)

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
                        frame = self.toPIL(frame)
                        frame = self.transforms_(frame)
                        trans_clip.append(frame)
                    clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                else:
                    clip = torch.tensor(clip)
                all_clips.append(clip)
                all_idx.append(int(class_idx))

            return torch.stack(all_clips), torch.tensor(all_idx)


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
    labels = [item[1] for item in batch]

    print(f"Labels: {labels}")
    print(f"Labels type: {[type(label) for label in labels]}")
    print(f"Labels length: {len(labels)}")

    try:
        labels = torch.tensor(labels, dtype=torch.int64)
    except Exception as e:
        print(f"Error converting labels to tensor: {e}")
        raise e

    videos = torch.stack(videos)
    return videos, labels