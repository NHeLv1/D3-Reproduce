import random
import argparse
import yaml
import torch
import os
import util
from util import build_model, train_one_epoch, eval_model
from dataloader import generate_dataset_loader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, LambdaLR, MultiStepLR
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of detector in yaml format')
    args = parser.parse_args()

    return args


if __name__ == '__main__': 
    args = get_arguments()
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    print("******* Building models. *******")
    print(cfg)
    model = util.build_model(cfg['model'])
    model = model.cuda()

    if cfg['tuning_mode'] == 'lp':
        for param in model.encoder.parameters():
            param.requires_grad = False

    # model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=[20, 25], gamma=0.1)
    loss = nn.BCEWithLogitsLoss()
    
    trMaxEpoch = 1
    snapshot_path = cfg['save_dir']
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    max_epoch, max_acc = 0, 0

    for epochID in range(0, trMaxEpoch):
        print("******* Building datasets. *******")
        train_loader, val_loader = generate_dataset_loader(cfg)
        pred_accuracy, video_id, pred_labels, true_labels, outpred = eval_model(cfg, model, val_loader, loss, cfg['val_batch_size'])

        df_result = pd.DataFrame({
            'data_path': video_id,
            'predicted_label': pred_labels,
            'actual_label': true_labels,
            'predicted_prob':outpred
        })

        temp_result_txt = snapshot_path+'/Epoch_'+str(epochID)+'_accuracy.txt'
        with open(temp_result_txt, 'w') as file:
            true_labels = df_result['actual_label']
            pred_probs = df_result['predicted_prob'] 
            auc = roc_auc_score(true_labels, pred_probs)
            ap = average_precision_score(true_labels, pred_probs)
            file.write(f"总正确率: {pred_accuracy:.2%}\n")
            file.write(f"AUC是: {auc:.2%}\n")
            file.write(f"AP是: {ap:.2%}\n")

        prefixes = ["Fake/ModelScope", "Fake/MorphStudio", "Fake/MoonValley", 
                    "Fake/HotShot", "Fake/Show_1", 
                    "Fake/Sora", "Fake/WildScrape", "Fake/Crafter",
                   "Fake/Lavie", "Fake/Gen2"]

        video_nums = [700, 700, 626, 700, 700, 56, 926, 1400, 1400, 1380]

        # real 
        condition = df_result['data_path'].apply(lambda x: "/Real/" in x)
        temp_df_val = df_result[condition]
        temp_df_val['correct'] = temp_df_val['predicted_label'] == temp_df_val['actual_label']
        # print(temp_df_val['predicted_label'], temp_df_val['actual_label'])  # --- IGNORE ---
        accuracy = temp_df_val['correct'].mean()

        FP = int((1-accuracy) * 10000)

        for index, temp_prefixes in enumerate(prefixes):
            condition = df_result['data_path'].apply(lambda x: temp_prefixes in x)
            temp_df_val = df_result[condition]
            temp_df_val['correct'] = temp_df_val['predicted_label'] == temp_df_val['actual_label']
            accuracy = temp_df_val['correct'].mean()

            TP = int(accuracy * video_nums[index])
            FN = int((1-accuracy) * video_nums[index])
            P, R = TP / (TP + FP), TP / (TP + FN)
            F1 = 2 * P * R / (P + R)

            condition |= df_result['data_path'].str.startswith('real')
            temp_df_val = df_result[condition]
            true_labels = temp_df_val['actual_label']
            pred_probs = temp_df_val['predicted_prob']  # 假设这是模型预测的概率
            ap = average_precision_score(true_labels, pred_probs)
            with open(temp_result_txt, 'a') as file:
                name = temp_prefixes.split('/')[-1]
                file.write(f"文件名: {name}, Recall是: {accuracy}\n")
                file.write(f"文件名: {name}, F1是: {F1}\n")
                file.write(f"文件名: {name}, AP是: {ap}\n")

