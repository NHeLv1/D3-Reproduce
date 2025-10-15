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
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
import matplotlib.pyplot as plt


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
    th = 0.7

    for epochID in range(0, trMaxEpoch):
        print("******* Building datasets. *******")
        train_loader, val_loader = generate_dataset_loader(cfg)
        pred_accuracy, video_id, pred_labels, true_labels, outpred = eval_model(cfg, model, val_loader, loss, cfg['val_batch_size'], th=th)

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
            file.write(f"=============================阈值设为: {th}=============================\n")
            file.write(f"总正确率: {pred_accuracy:.2%}\n")
            file.write(f"AUC是: {auc:.2%}\n")
            file.write(f"AP是: {ap:.2%}\n")

        prefixes = ["Fake/ModelScope", "Fake/MorphStudio", "Fake/MoonValley", 
                    "Fake/HotShot", "Fake/Show_1", 
                    "Fake/Sora", "Fake/WildScrape", "Fake/Crafter",
                   "Fake/Lavie", "Fake/Gen2"]

        video_nums = [700, 700, 626, 700, 700, 56, 926, 1400, 1400, 1380]

        # real 
        real_condition = df_result['data_path'].apply(lambda x: "/Real/" in x or x.startswith('real') or x.startswith('Real'))
        real_df = df_result[real_condition].copy()


        for temp_prefix in prefixes:
            # Fake 子集
            fake_condition = df_result['data_path'].apply(lambda x: temp_prefix in x)
            fake_df = df_result[fake_condition].copy()

            # 若没有 fake 样本，跳过（或记录为 nan）
            if fake_df.empty:
                with open(temp_result_txt, 'a') as file:
                    name = temp_prefix.split('/')[-1]
                    file.write(f"文件名: {name}, Recall是: {float('nan')}\n")
                    file.write(f"文件名: {name}, F1是: {float('nan')}\n")
                    file.write(f"文件名: {name}, AP是: {float('nan')}\n")
                continue

            # 合并 fake 子集 + 所有 real 样本
            combined_df = pd.concat([fake_df, real_df], ignore_index=True)
            # y_true, y_pred, y_prob
            y_true = combined_df['actual_label'].astype(int)
            y_pred = combined_df['predicted_label'].astype(int)
            y_prob = combined_df['predicted_prob'].astype(float)

            # 计算 precision/recall/f1（把正类 pos_label 设为 1，表示 fake 为正类）
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', pos_label=1, zero_division=0
            )

            # AP（对 combined 计算）
            try:
                ap = average_precision_score(y_true, y_prob)
            except Exception:
                ap = float('nan')

            name = temp_prefix.split('/')[-1]
            with open(temp_result_txt, 'a') as file:
                file.write(f"文件名: {name}, Recall是: {recall}\n")
                file.write(f"文件名: {name}, F1是: {f1}\n")
                file.write(f"文件名: {name}, AP是: {ap}\n")
            

            # =============================== 绘制直方图 ===============================
            plt.figure(figsize=(6, 4))
            plt.hist(
                y_prob[y_true == 0],
                bins=30, alpha=0.6, color='skyblue', label='Real (label=0)'
            )
            plt.hist(
                y_prob[y_true == 1],
                bins=30, alpha=0.6, color='salmon', label='Fake (label=1)'
            )
            plt.title(f"Epoch {epochID} - {name} Probability Distribution")
            plt.xlabel("Predicted Probability")
            plt.ylabel("Count")
            plt.legend()
            plt.grid(alpha=0.3)

            # 保存直方图
            hist_path = os.path.join(snapshot_path, f"Epoch_{epochID}_{name}_hist.png")
            plt.tight_layout()
            plt.savefig(hist_path, dpi=200)
            plt.close()