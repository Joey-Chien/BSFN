import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2 as cv
import time
import os
from tqdm import tqdm
import pandas as pd



mean = [0.485, 0.456, 0.406]  # RGB
std = [0.229, 0.224, 0.225]



class BBDataset(Dataset):
    def __init__(self, file_dir='dataset', type='train', test=False, images_dir='/home/joey/from_local/AIAA_2/AIAA/BAID/images'):
        self.if_test = test
        self.train_transformer = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.test_transformer = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.images = []
        self.pic_paths = []
        self.labels = []
        self.q_scores = []

        if type == 'train':
            DATA = pd.read_csv(os.path.join(file_dir, 'BAID_train.csv'))
        elif type == 'validation':
            DATA = pd.read_csv(os.path.join(file_dir, 'BAID_val.csv'))
        elif type == 'test':
            DATA = pd.read_csv(os.path.join(file_dir, 'BAID_test.csv'))

        labels = DATA['score'].values.tolist()
        q_scores = DATA['qalign_point'].values.tolist()
        pic_paths = DATA['image'].values.tolist()
        for i in tqdm(range(len(pic_paths))):
            pic_path = os.path.join(images_dir, pic_paths[i])
            
            # pic_path = os.path.join('/local/joey/AIAA/BAID/images', pic_paths[i])
            # pic_path = os.path.join('/home/joey/from_local/AIAA_2/AIAA/BAID/images', pic_paths[i])
            label = float(labels[i] / 10)
            self.pic_paths.append(pic_path)
            self.labels.append(label)
            self.q_scores.append(q_scores[i])

    def __len__(self):
        return len(self.pic_paths)

    def __getitem__(self, index):
        pic_path = self.pic_paths[index]
        img = cv.imread(pic_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if self.if_test:
            img = self.test_transformer(img)
        else:
            img = self.train_transformer(img)

        return img, self.labels[index], self.q_scores[index]