# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import os, time, random, uuid, torch, torchmetrics, argparse
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchsummary import summary
from pytorch_metric_learning import losses
from ..utils.loss import LabelSmoothingCrossEntropy as LSCE
from ..utils.metrics import compute_precision_recall_f1, compute_accuracy, merge_cm_to_2_classes, merge_cm_to_3_classes
import multiprocessing as mp
import threading
from ..utils.gpu_monitor import monitor_gpu_usage
import torch

num_workers = 2

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=r'drunk', help='Input the ann_file.')

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=80, help='Number of epochs to train.')
parser.add_argument('--warm_up_epoch', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--model', type=str, default='stgcn', help='stgcn/ctrgcn/.....')

# NAS init stream
parser.add_argument('--base_channel', type=int, default=64, help='base channel.')
parser.add_argument('--num_init', type=int, default=4, help='number of layers about init stream.')
parser.add_argument('--tkernel_init', type=int, default=3, help='learning rate.')
parser.add_argument('--stride_init', type=int, default=1, help='the stride about the first tcn of input stream')

# NAS input stream
parser.add_argument('--num_in', type=int, default=3, help='number of layers about input stream.')
parser.add_argument('--tkernel_in', type=int, default=3, help='learning rate.')
parser.add_argument('--stride_in', type=int, default=2, help='the stride about the first tcn of input stream')
parser.add_argument('--oc_in', type=int, default=128, help='the stride about the first tcn of input stream')

# NAS main stream
parser.add_argument('--num_main', type=int, default=3, help='number of layers about main stream.')
parser.add_argument('--tkernel_main', type=int, default=3, help='learning rate.')
parser.add_argument('--stride_main', type=int, default=2, help='the stride about the first tcn of main stream')
parser.add_argument('--oc_main', type=int, default=256, help='the stride about the first tcn of main stream')

# NAS other
parser.add_argument('--act', type=int, default=0, help='act func.')
parser.add_argument('--opt', type=int, default=0, help='optimizor.')

# HPO
parser.add_argument('--scheduler', type=int, default=0, help='0:Cosine with warm up, 1:Cosine, 2:None')
parser.add_argument('--batch', type=int, default=16, help='batch size.')
parser.add_argument('--dropout_bk', type=float, default=0, help='Dropout rate.')
parser.add_argument('--dropout_fc', type=float, default=0, help='Dropout rate.')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay (L2 loss on parameters).')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum.')

# HPO - triplet loss
parser.add_argument('--lambda_val', type=float, default=0.5, help='the ratio about triplet loss')
parser.add_argument('--margin', type=float, default=0.6, help='the margin of triplet loss')
parser.add_argument('--mt', action='store_true', default=False, help='mt or not')

# preprocessing
parser.add_argument('--frame', type=int, default=90, help='seq_len(frames).')
parser.add_argument('--feature', type=str, default='j', help='j/b/jm/bm.')

# dataset
parser.add_argument('--view', type=str, default='sagittal', help='coronal/sagittal/full.')
parser.add_argument('--split', type=int, default=1, help='split num 1~5')
parser.add_argument('--run', type=int, default=1, help='run times 1~5')
parser.add_argument('--pose', type=str, default='yolo', help='yolo/hrnet/blaze')
parser.add_argument('--person', type=int, default=1, help='person or inter')
parser.add_argument('--cls', type=int, default= 4, help='class')
parser.add_argument('--test', action='store_true', default=False, help='T return test acc/ F return val acc.')
parser.add_argument('--partition', action='store_true', default=False, help='partition or not')
parser.add_argument('--loss', type=str, default='LSCE', help='LSCE or CE')
# 沒有使用到
parser.add_argument('--patience', type=int, default=10, help='early stop')
parser.add_argument('--input_channel', type=int, default=3, help='just PD DATASET is 2 ')
parser.add_argument('--avg', type=str, default='micro', help='micro or macro')
parser.add_argument('--pretrained', type=str, default=None,  help='micro or macro')

args = parser.parse_args()

if args.cls == 4:
    class_names = ['Normal', 'Low', 'Medium', 'High']
elif args.cls == 3:
    class_names = ['Normal', 'Low', 'High']
elif args.data == 'nw-ucla':
    class_names = ['1', '2', '3','4','5','6','7','8','9','10']

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

tri_loss, loss_fn = None, None

# 直接預設GPU
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'

global_step = 0
cm = None


def visulize(args):
    global cm, global_step, tri_loss, loss_fn
    global_step = 0

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f"runs/{args.data}_{args.feature}_{args.split}_{args.epochs}_{timestamp}")

    csv_file =  args.data
    suffix = os.path.splitext(args.data)[-1].lower()

    if csv_file == 'drunk':
        # from ..utils.datasets_npz_aug_2c_flip_fixed import load_npz_dataset
        from ..utils.datasets_npz_aug_2c_flip import load_npz_dataset
        # from ..utils.datasets_npz_ori import load_npz_dataset
        num_point = 17
        args.input_channel = 2

        train_file = 'data/' + 'drunk71_small/dataset_'+ args.pose + '_' + args.view + f'_split{args.split}_'+ str(args.frame) + 'frames_j_' + str(args.cls) + 'class_train.npz'
        validation_file ='data/' + 'drunk71_small/dataset_' + args.pose + '_' + args.view + f'_split{args.split}_'+ str(args.frame) + 'frames_j_' + str(args.cls) + 'class_val.npz'
        test_file ='data/' + 'drunk71_small/dataset_' + args.pose + '_' + args.view + f'_split{args.split}_'+ str(args.frame) + 'frames_j_' + str(args.cls) + 'class_test.npz'
    
        training_loader = load_npz_dataset(train_file,  workers_per_gpu=num_workers, batch_size=args.batch, pose=args.pose, feature=args.feature)
        validation_loader = load_npz_dataset(validation_file,  workers_per_gpu=num_workers, batch_size=args.batch, pose=args.pose, feature=args.feature)
        test_loader = load_npz_dataset(test_file,  workers_per_gpu=num_workers, batch_size=args.batch, pose=args.pose, feature=args.feature)
    
    if args.pretrained :
        if args.model == 'stgcn':
            from ..models.stgcnpp_sso import SSO_STGCN
            model = SSO_STGCN(graph_cfg=dict(layout=args.pose, mode='spatial'),
                    in_channels=args.input_channel,
                    data_bn_type='VC',
                    num_person=args.person,  # * Only used when data_bn_type == 'MVC'
                    pretrained=None,
                    tcn_type='mstcn',
                    tcn_dropout =args.dropout_bk,
                    gcn_with_res = True,
                    gcn_adaptive='init',
                    args=args, #直接把args送進去
                    )
            # 確保 args.pretrained 是一個有效的 checkpoint 路徑
            if isinstance(args.pretrained, str) and os.path.exists(args.pretrained):
                print(f"Loading pretrained model from {args.pretrained} ...")
                checkpoint = torch.load(args.pretrained, map_location=args.device)
                # 檢查 checkpoint 是否包含 `state_dict`
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)  # 直接載入模型權重
                
                print("Pretrained model loaded successfully!")
            else:
                print("Invalid pretrained model path. Skipping weight loading.")


    import numpy as np
    # import torch
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model.to(args.device)
    # 儲存所有數據和標籤
    all_inputs = []
    all_labels = []
    all_outputs = []
    all_embeddings = []
    # 讀取整個 training_loader
    for batch in test_loader:
        inputs, labels = batch['keypoint'], batch['label']
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        
        # 選擇 M=0，保留所有時間步與關鍵點
        # inputs = inputs[:, 0, :, :, :]  # (N, T, V, C)

        # 將 C 維展平 (x,y座標)
        N, M,  T, V, C = inputs.shape

        if args.pretrained:
            model.eval()
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                outputs, embedding = model(inputs)
                all_outputs.append(outputs.cpu().numpy())

        inputs = inputs.view(N, M * T * V * C)  # (N, T*V*C)
        
        # 轉換為 numpy
        all_inputs.append(inputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_embeddings.append(embedding.cpu().numpy())

    # 拼接所有 batch
    all_inputs_np = np.concatenate(all_inputs, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    
    if args.pretrained:
        all_outputs_np = np.concatenate(all_outputs, axis=0)  # (N, num_classes)
        all_preds_np = np.argmax(all_outputs_np, axis=1)      # (N,)
        all_embeddings_np = np.concatenate(all_embeddings, axis=0)  # 你需要在上面存 all_embeddings

    # # 使用 t-SNE 降維到 2D
    # tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    # inputs_2d = tsne.fit_transform(all_inputs_np)

    # # 定義顏色和標籤對應關係
    # label_names = ['Normal', 'Low', 'Medium', 'High']
    # colors = ['blue', 'green', 'orange', 'red']

    # # 畫圖
    # plt.figure(figsize=(8, 6))

    # # 針對每個類別分別繪製散點圖
    # for i, label in enumerate(label_names):
    #     indices = (all_labels_np == i)
    #     plt.scatter(inputs_2d[indices, 0], inputs_2d[indices, 1], c=colors[i], label=label, alpha=0.7)

    # # 添加圖例
    # plt.legend(title="Class Labels")
    # plt.title(f"t-SNE Visualization of Gait Keypoints (Entire Dataset) - Split {args.split} - testing")
    # plt.xlabel("t-SNE Dimension 1")
    # plt.ylabel("t-SNE Dimension 2")

    # # 設定存檔路徑
    # save_path = f"colab/t-sne/tsne_split{args.split}_testing_fixed.png"

    # # 儲存圖片
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi=300 提高圖片解析度
    # print(f"t-SNE 圖表已保存至: {save_path}")

    # # 顯示圖形
    # # plt.show()
    #############################################

    # 使用 t-SNE 降維到 2D：針對 embedding 做
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding_2d = tsne.fit_transform(all_embeddings_np)

    # 定義顏色和標籤對應關係
    label_names = ['Normal', 'Low', 'Medium', 'High']
    colors = ['blue', 'green', 'orange', 'red']

    plt.figure(figsize=(8, 6))

    # 畫正確分類點
    plt.figure(figsize=(8, 6))

    # 畫每個類別（正確與錯誤分開標示）
    for i, label in enumerate(label_names):
        # 正確分類：o 圓圈
        correct = (all_labels_np == i) & (all_preds_np == i)
        plt.scatter(embedding_2d[correct, 0], embedding_2d[correct, 1],
                    c=colors[i], label=f"{label} (Correct)", marker='o', alpha=0.7)

        # 錯誤分類：x 標記
        incorrect = (all_labels_np == i) & (all_preds_np != i)
        plt.scatter(embedding_2d[incorrect, 0], embedding_2d[incorrect, 1],
                    c='black', label=f"{label} (Wrong)", marker='x', alpha=0.9)

    # 添加圖例與標題
    plt.legend(title="Prediction Results")
    plt.title(f"t-SNE of Embedding Space (Split {args.split})")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    # 存檔
    save_path = f"colab/t-sne/tsne_split{args.split}_embedding.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE 嵌入圖已保存至: {save_path}")

    
    
if __name__ == '__main__':
   visulize(args)