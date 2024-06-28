import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from AVA_data import BBDataset
from torch.utils.data import DataLoader
from models.model import SAAN, QALIGN_1, QALIGN_2, QALIGN_3, QALIGN_4, QALIGN_5, CSQ_check_26_AVA
import torch.optim as optim
from common import *
import argparse


train_dataset = BBDataset(file_dir='qalign_score_dataset', type='train', test=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoint/AVA_check_1')
    parser.add_argument('--val_freq', type=int,
                        default=100)
    parser.add_argument('--save_freq', type=int,
                        default=1)

    return parser.parse_args()


def train(args):
    device = args.device

    model = BSFN_AVA(num_classes=1)

    for name, param in model.named_parameters():
        if 'GenAes' in name:
            param.requires_grad = False
    model = model.to(device)

    loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=5e-4)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)

    # for epoch in range(64, 64 + args.epoch):
    for epoch in range(args.epoch):
        if (epoch+1) % 5 == 0:
            args.lr *= 0.1
            lr = args.lr
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        print("Learning Rate: ", args.lr)
        model.train()
        epoch_loss = 0.0

        for step, train_data in enumerate(train_loader):
            optimizer.zero_grad()
            image = train_data[0].to(device)
            label = train_data[1].to(device).float()
            q_score = train_data[2].to(device)
            A_q_score = train_data[3].to(device)

            predicted_label = model(image, q_score, A_q_score).squeeze()
            train_loss = loss(predicted_label, label)

            train_loss.backward()
            optimizer.step()

            epoch_loss += train_loss.item()

            if step % 100 == 0:
                print("Epoch: %3d Step: %5d / %5d Train loss: %.8f" % (epoch, step, len(train_loader), train_loss.item()))

        adjust_learning_rate(args, optimizer, epoch)

        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(args, model, epoch)


if __name__ == '__main__':
    args = parse_args()
    train(args)
