import os
import scipy.stats
from AVA_data import imageDataset
from torch.utils.data import DataLoader
from models.model import BSFN_AVA
import torch.nn as nn
import torch
import torch.optim as optim
import argparse
import torch.nn.functional as F
import pandas as pd
import scipy
from tqdm import tqdm

test_dataset = imageDataset(file_dir='dataset/AVA_dataset', type='test', test=True, images_dir='/local/joey/images')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/home/joey/paper/checkpoint/AVA')
    parser.add_argument('--checkpoint_name', type=str,
                        default='model_best.pth')
    parser.add_argument('--save_dir', type=str,
                        default='result')

    return parser.parse_args()

def test(args, test_epoch):
    device = args.device
    checkpoint_path = os.path.join(args.checkpoint_dir, 'epoch_' + str(test_epoch) + '.pth')
    df = pd.read_csv('dataset/AVA_dataset/AVA_test.csv')
    predictions = []

    model = BSFN_AVA(num_classes=1)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)

    with torch.no_grad():
        for step, test_data in tqdm(enumerate(test_loader)):
            image = test_data[0].to(device)
            q_score = test_data[2].to(device)
            A_q_score = test_data[3].to(device)

            predicted_label = model(image, q_score, A_q_score)
            prediction = predicted_label.squeeze().cpu().numpy()
            predictions.append(prediction * 10)

    scores = df['score'].values.tolist()

    print(scipy.stats.spearmanr(scores, predictions))
    print(scipy.stats.pearsonr(scores, predictions))

    acc = 0
    for i in range(len(scores)):
        cls1 = 1 if scores[i] > 5 else 0
        cls2 = 1 if predictions[i] > 5 else 0
        if cls1 == cls2:
            acc += 1
    print(acc/len(scores))
    df.insert(loc=2, column='prediction', value=predictions)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    save_path = os.path.join(args.save_dir, 'result.csv')
    df.to_csv(save_path, index=False)
    
    return scipy.stats.spearmanr(scores, predictions), scipy.stats.pearsonr(scores, predictions), acc/len(scores)

if __name__ == '__main__':
    args = parse_args()
    srcc = []
    prcc = []
    acc = []
    for test_epoch in range(99, 100):
        print('-' * 50, 'epoch: ', test_epoch)
        temp_srcc, temp_prcc, temp_acc = test(args,test_epoch)
        srcc.append(temp_srcc)
        prcc.append(temp_prcc)
        acc.append(temp_acc)
        result = {'srcc':srcc, 'prcc':prcc, 'acc':acc}
        result_csv = pd.DataFrame(result)
        result_csv.to_csv('result/AVA_check.csv')
