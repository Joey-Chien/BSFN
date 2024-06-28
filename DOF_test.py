import os
import scipy.stats
from DOF_data import BBDataset
from torch.utils.data import DataLoader
from models.model import BSFN, BSFN_AVA
import torch.nn as nn
import torch
import torch.optim as optim
import argparse
import torch.nn.functional as F
import pandas as pd
import scipy
from tqdm import tqdm

test_dataset = BBDataset(file_dir='dataset/DOF_dataset', type='test', test=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoint/DOF')
    parser.add_argument('--checkpoint_name', type=str,
                        default='model_best.pth')
    parser.add_argument('--save_dir', type=str,
                        default='result')

    return parser.parse_args()

def test(args, test_epoch):
    device = args.device
    checkpoint_path = os.path.join(args.checkpoint_dir, 'epoch_' + str(test_epoch) + '.pth')
    df = pd.read_csv('dataset/DOF_dataset/DOF_test.csv')
    predictions = []

    model = BSFN(num_classes=1)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)

    with torch.no_grad():
        for step, test_data in tqdm(enumerate(test_loader)):
            image = test_data[0].to(device)
            q_score = test_data[2].to(device)

            predicted_label = model(image, q_score)
            # print('predicted: ', predicted_label)
            prediction = predicted_label.squeeze().cpu().numpy()
            predictions.append(prediction * 10)

    scores = df['score'].values.tolist()

    print(scipy.stats.spearmanr(scores, predictions))
    print(scipy.stats.pearsonr(scores, predictions))

    acc = 0
    for i in range(len(scores)):
        if abs(scores[i] - predictions[i]) <= 0.318:
            acc += 1
    print(acc/len(scores))


    acc_correct = 0
    for i in range(len(scores)//3):
        temp_1 = i * 3 + 0
        temp_2 = i * 3 + 1
        temp_3 = i * 3 + 2
        indices = [temp_1, temp_2, temp_3]
        a = [scores[i] for i in indices]
        b = [predictions[i] for i in indices]
        a_dict = {'original':a[0], 'bokeh':a[1], 'blur':a[2]}
        b_dict = {'original':b[0], 'bokeh':b[1], 'blur':b[2]}
        a_sorted_keys = [k for k, v in sorted(a_dict.items(), key=lambda item: item[1])]
        b_sorted_keys = [k for k, v in sorted(b_dict.items(), key=lambda item: item[1])]


        if (a_sorted_keys[0] == b_sorted_keys[0] and a_sorted_keys[1] == b_sorted_keys[1] and a_sorted_keys[2] == b_sorted_keys[2]):
            acc_correct += 1
            # print(a, a_sorted_keys)
            # print(b, b_sorted_keys)
    print('Group_Acc: ', acc_correct / (len(scores)//3) )

        


    df.insert(loc=2, column='prediction', value=predictions)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    save_path = os.path.join(args.save_dir, 'result.csv')
    # df.to_csv(save_path, index=False)

    return scipy.stats.spearmanr(scores, predictions), scipy.stats.pearsonr(scores, predictions), acc/len(scores), acc_correct / (len(scores)//3)


if __name__ == '__main__':
    args = parse_args()
    srcc = []
    prcc = []
    acc = []
    acc_correct = []
    for test_epoch in range(99, 100):
        print('-' * 50, 'epoch: ', test_epoch)
        temp_srcc, temp_prcc, temp_acc, temp_acc_correct = test(args,test_epoch)
        srcc.append(temp_srcc)
        prcc.append(temp_prcc)
        acc.append(temp_acc)
        acc_correct.append(temp_acc_correct)
        result = {'srcc':srcc, 'prcc':prcc, 'acc':acc, 'Group_Acc':acc_correct}
        result_csv = pd.DataFrame(result)
        result_csv.to_csv('result/DOF_output.csv')
