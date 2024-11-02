# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import os
import time
# from pyskl.datasets import build_dataloader, build_dataset
from ..utils.datasets import load_dataset, load_npz_dataset_random_split
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchsummary import summary
import argparse
import random
import uuid
import torchmetrics
from ..models.stgcnpp import STGCN

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=r"C:\Users\ADMIN\PycharmProjects\Gait\pyskl\tools\data\drunk\crop.csv", help='Input the ann_file.')
# parser.add_argument('--data',type=str, default=r'tools/data/KOA_NM/splits/KOA_NM_split_1.pkl', help='Input the ann_file.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum.')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=300, help='Patience')
# parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--frame', type=int, default=200, help='seq_len(frames).')
parser.add_argument('--feature', type=str, default='j', help='seq_len(frames).')
parser.add_argument('--batch', type=int, default=16, help='batch size.')
parser.add_argument('--view', type=str, default='full', help='coronal/sagittal/full.')
parser.add_argument('--split', type=int, default=1, help='split num 1~5')
parser.add_argument('--run', type=int, default=1, help='run times 1~5')
parser.add_argument('--pose', type=str, default='yolo', help='yolo/hrnet/blaze')

class_names = ['Normal', 'Low', 'Medium', 'High']

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'

# 直接預設GPU
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=4).to(args.device)
loss_fn = torch.nn.CrossEntropyLoss()


def train_step(model, optimizer, training_loader, tb_writer, epoch_index, scheduler):
    model.train()
    accuracy_metric.reset()  # Reset accuracy metric at the beginning of the epoch
    running_loss = 0.0
    running_acc = 0.0

    for i, batch in enumerate(training_loader):
        inputs, labels = batch['keypoint'], batch['label']
        # print(f"inputs: {inputs.size(0)}, labels: {labels.size(0)}")
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        if labels.dim() > 1 and labels.size(-1) == 1:
            labels = labels.squeeze(-1)
        # 檢查 labels 是否為空
        if labels.size(0) == 0:
            print("Skipping empty labels batch")
            continue  # 跳過空的 batch

        optimizer.zero_grad()

        outputs, embedding = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        batch_acc = accuracy_metric(preds, labels)
        running_acc += batch_acc.item()

    avg_loss = running_loss / len(training_loader)
    avg_acc = running_acc / len(training_loader)

    if tb_writer:
        tb_x = epoch_index + 1
        tb_writer.add_scalar('Loss/train/epoch', avg_loss, tb_x)
        tb_writer.add_scalar('Accuracy/train/epoch', avg_acc, tb_x)
    return avg_loss, avg_acc

def validate_step(model, validation_loader, tb_writer, epoch_index):
    model.eval()
    accuracy_metric.reset()  # Reset accuracy metric at the beginning of the epoch
    running_loss = 0.0
    running_acc = 0.0

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, batch in enumerate(validation_loader):
            inputs, labels = batch['keypoint'], batch['label']
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            if labels.dim() > 1 and labels.size(-1) == 1:
                labels = labels.squeeze(-1)
            outputs, _ = model(inputs)
            loss = loss_fn(outputs, labels)
            batch_size = inputs.size(0)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            batch_acc = accuracy_metric(preds, labels)
            running_acc += batch_acc.item()


    avg_loss = running_loss / len(validation_loader)
    avg_acc = running_acc / len(validation_loader)

    if tb_writer:
        tb_x = epoch_index + 1
        tb_writer.add_scalar('Loss/valid/epoch', avg_loss, tb_x)
        tb_writer.add_scalar('Accuracy/valid/epoch', avg_acc, tb_x)

    return avg_loss, avg_acc

def test_step(model, test_loader, tb_writer, epoch_index, save_path):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    accuracy_metric.reset()  # Reset accuracy metric at the beginning of the epoch
    running_loss = 0.0
    running_acc = 0.0
    all_preds = []
    all_labels = []

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs, labels = batch['keypoint'], batch['label']
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            if labels.dim() > 1 and labels.size(-1) == 1:
                labels = labels.squeeze(-1)
            outputs, embedding = model(inputs)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            batch_acc = accuracy_metric(preds, labels)
            running_acc += batch_acc.item()


            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(test_loader)
    avg_acc = running_acc / len(test_loader)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    if tb_writer:
        tb_x = epoch_index + 1
        tb_writer.add_scalar('Loss/test/epoch', avg_loss, tb_x)
        tb_writer.add_scalar('Accuracy/test/epoch', avg_acc, tb_x)

        # Plot confusion matrix using seaborn
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

        # Log the confusion matrix as an image in TensorBoard
        tb_writer.add_figure(f'Confusion_Matrix/test/epoch_{tb_x}', plt.gcf(), tb_x)
        plt.close()

    return avg_loss, avg_acc


def train(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    csv_file =  args.data
    suffix = os.path.splitext(args.data)[-1].lower()
    if suffix == 'pkl':
        csv_file =  args.data
        training_loader, validation_loader, test_loader = load_dataset(csv_file, feats=args.feature, batch_size=16, frames=200, pose_extractor=args.pose, mode = args.view, workers_per_gpu=2, split=args.split)
    if suffix == 'npz':
        npz_file =  args.data
        training_loader, validation_loader, test_loader = load_npz_dataset_random_split(npz_file,  workers_per_gpu=2, batch_size=16)

    # # learning policy
    lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
    model = STGCN(graph_cfg=dict(layout='coco', mode='spatial'),
                  in_channels=3,
                  base_channels=64,
                  data_bn_type='VC',
                  ch_ratio=2,  # 每次放大兩倍通道數
                  num_person=1,  # * Only used when data_bn_type == 'MVC'
                  num_stages=4,  # 有幾層
                  inflate_stages=[2, 3],  # 通道數在這兩層增加
                  down_stages=[2, 3],  # 時間維度下採樣，壓縮時間維度
                  pretrained=None,
                  tcn_type='mstcn',
                  tcn_dropout = 0,
                  gcn_with_res = True,
                  gcn_adaptive='init',
                  head_dropout=0,
                  num_classes=4)

    model.to(args.device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay, momentum=args.momentum)
    
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(training_loader), eta_min=lr_config['min_lr'])
    
    bad_counter = 0
    best = 999999999

    for epoch in range(args.epochs):
        loss_tra, acc_tra = train_step(model, optimizer, training_loader, writer, epoch, scheduler)
        loss_val, acc_val = validate_step(model, validation_loader, writer, epoch)
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 1 == 0:
            print('Epoch:{:04d}'.format(epoch + 1),
                  'train',
                  'lr:{:.3f}'.format(current_lr),
                  'loss:{:.3f}'.format(loss_tra),
                  'acc:{:.2f}'.format(acc_tra * 100),
                  '| val',
                  'loss:{:.3f}'.format(loss_val),
                  'acc:{:.2f}'.format(acc_val * 100))
        if loss_val <= best:

            checkpt_dir = os.path.dirname(checkpt_file)
            if not os.path.exists(checkpt_dir):
                os.makedirs(checkpt_dir)

            best = loss_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'results/confusion_matrix_{timestamp}.png'

    acc = test_step(model, test_loader, writer, epoch, save_path=f'results/confusion_matrix_{timestamp}.png')[1]

    return acc * 100

def save_results(args, acc_list, filename='results.txt'):
    with open(filename, 'w') as f:
        #
        ## 寫入 args
        checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'
        f.write("Checkpt_file:\n")
        f.write(f"{checkpt_file}: {checkpt_file}\n")


        # 寫入 args
        f.write("\nArguments:\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

        # 寫入 acc_list
        f.write("\nAccuracy List:\n")
        for epoch, acc in enumerate(acc_list):
            f.write(f"Run {epoch + 1}: {acc}\n")

        f.write(f"\nTrain cost: {(time.time() - t_total)}s\n")
if __name__ == '__main__':

    t_total = time.time()
    acc_list = []

    # train 幾次
    for i in range(args.run):
        acc_list.append(train(args))
        print(i, ": {:.4f}".format(acc_list[-1]))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results(args, acc_list, filename = f'results/record_{timestamp}.txt')

    print("Train cost: {:.4f}s".format(time.time() - t_total))
    print(acc_list)
    print(f"Final Test Accuracy: {np.mean(acc_list)}")
    print("Test acc.:{:.4f}".format(np.mean(acc_list)))