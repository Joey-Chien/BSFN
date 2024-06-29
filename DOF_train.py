import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from DOF_data import BBDataset
from torch.utils.data import DataLoader
from models.model import BSFN, BSFN_AVA
import torch.optim as optim
import argparse


train_dataset = BBDataset(file_dir='dataset/DOF_dataset', type='train', test=False, images_dir='/home/joey/BAID/DOF/bokeh_image_add_blur')

def save_checkpoint(args, model, epoch):
    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoint/DOF/store_weight_1')
    parser.add_argument('--val_freq', type=int,
                        default=50)
    parser.add_argument('--save_freq', type=int,
                        default=1)

    return parser.parse_args()



def train(args):
    device = args.device

    model = BSFN(num_classes=1)
    for name, param in model.named_parameters():
        if 'GenAes' in name:
            param.requires_grad = False
    model = model.to(device)

    loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=5e-4)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)

    print(model)

    for epoch in range(args.epoch):
        model.train()
        epoch_loss = 0.0
        if (epoch+1) % 10 == 0:
            args.lr *= 0.1
            lr = args.lr
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        print("Learning Rate: ", args.lr)

        for step, train_data in enumerate(train_loader):
            optimizer.zero_grad()
            image = train_data[0].to(device)
            # print(image.shape)
            label = train_data[1].to(device).float()
            q_score = train_data[2].to(device)

            predicted_label = model(image, q_score).squeeze()
            # print('Label: ', label, 'Predicted_label: ', predicted_label)
            train_loss = loss(predicted_label, label)

            train_loss.backward()
            optimizer.step()

            epoch_loss += train_loss.item()

            if step%100 == 0:
                print("Epoch: %3d Step: %5d / %5d Train loss: %.8f" % (epoch, step, len(train_loader), train_loss.item()))

        # adjust_learning_rate(args, optimizer, epoch)

        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(args, model, epoch)


if __name__ == '__main__':
    args = parse_args()
    train(args)
