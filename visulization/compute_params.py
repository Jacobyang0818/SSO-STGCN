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
from thop import profile
from fvcore.nn import FlopCountAnalysis
from torchinfo import summary



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

def load_model(args):
    global cm, global_step, tri_loss, loss_fn
    global_step = 0
    args.input_channel = 2
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

    elif args.model == 'stgcn-raw':
        from ..models.stgcnpp_sso import SSO_STGCN
        model = SSO_STGCN(graph_cfg=dict(layout=args.pose, mode='spatial'),
                in_channels=args.input_channel,
                data_bn_type='VC',
                num_person=args.person,  # * Only used when data_bn_type == 'MVC'
                pretrained=None,
                tcn_type='unit_tcn',
                tcn_dropout =args.dropout_bk,
                gcn_with_res = False,
                gcn_adaptive='importance',
                args=args, #直接把args送進去
                )
        
    # ok
    elif args.model == 'ctrgcn':
        from ..models.ctrgcn import CTRGCN
        model = CTRGCN(graph_cfg=dict(layout=args.pose, mode='spatial'),
                num_classes=args.cls, 
                num_point=17,
                num_person=args.person,
                in_channels=args.input_channel
                )
    
    # ok
    elif args.model == 'dgstgcn':
        from ..models.dgstgcn import DGSTGCN
        model = DGSTGCN(dict(layout=args.pose, mode='random', num_filter=8, init_off=.04, init_std=.02),
                gcn_ratio=0.125,
                gcn_ctr='T',
                gcn_ada='T',
                tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                num_classes=args.cls,
                num_person=args.person,
                in_channels=args.input_channel
                )
        
    # ok
    elif args.model == 'aagcn':
        from ..models.aagcn import AAGCN
        model = AAGCN(dict(layout=args.pose, mode='spatial'),
                num_classes=args.cls,
                num_person=args.person,
                in_channels=args.input_channel
                )
        
    elif args.model == 'lstm1':
        from ..models.lstm import SimpleLSTM
        model = SimpleLSTM(input_dim=17*2, hidden_dim=256, num_layers=1, num_classes=4, dropout=0.3)
        args.opt = 1
        args.scheduler = None
        args.lr = 0.001
        args.epoch=100

    elif args.model == 'lstm2':
        from ..models.lstm import SimpleLSTM
        model = SimpleLSTM(input_dim=17*2, hidden_dim=256, num_layers=2, num_classes=4, dropout=0.3)
        args.opt = 1
        args.scheduler = None
        args.lr = 0.001
        args.epoch=100

    elif args.model == 'plstm1':
        from ..models.lstm import PLSTM
        model = PLSTM(input_dim=17*2, hidden_dim=256, num_layers=1, num_classes=4, dropout=0.3)
        args.opt = 1
        args.scheduler = None
        args.lr = 0.001
        args.epoch=100

    elif args.model == 'plstm2':
        from ..models.lstm import PLSTM
        model = PLSTM(input_dim=17*2, hidden_dim=256, num_layers=2, num_classes=4, dropout=0.3)
        args.opt = 1
        args.scheduler = None
        args.lr = 0.001
        args.epoch=100

    elif args.model == 'stgcn-lw':
        from ..models.stgcnpp_sso import SSO_STGCN
        args.num_in = 0
        args.num_main = 0
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
    
    elif args.model == 'sgn':
        from ..models.sgn import SGN
        model = SGN(in_channels=2, base_channels=64, num_joints=17, T=64, bias=True)
    return model

def compute():

    model_list = ['stgcn', 'stgcn-raw', 'ctrgcn', 'dgstgcn', 'aagcn', 'lstm1', 'lstm2', 'plstm1', 'plstm2']

    for m in model_list:
        args.model = m
        model = load_model(args)

        print(args.model)
        input_tensor = torch.randn(1, 1, 64, 17, 2)  # 一筆資料

        macs, params = profile(model, inputs=(input_tensor,))

        print(f"FLOPs: {macs / 1e9:.7f} GFLOPs")
        print(f"Params: {params / 1e6:.7f} M")  # 轉換成百萬參數
        ############################################################

    import itertools
    import csv
    # 寫入 CSV 結果檔案
    output_file = 'SSO模型參數量.csv'
    
    #原始的
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'num_init', 'base_channel', 'stride_init', 'tkernel_init', 'FLOPs(G)', 'Params(M)'])

        m='dgstgcn'
        args.model = m
        model = load_model(args)

        input_tensor = torch.randn(1, 1, 64, 17, 2)  # 一筆資料

        macs, params = profile(model, inputs=(input_tensor,))

        writer.writerow([
                    m,
                    args.num_init,
                    args.base_channel,
                    args.stride_init,
                    args.tkernel_init,
                    f"{macs / 1e9:.7f}",
                    f"{params / 1e6:.7f}"
                ])
        
        print(f"[{m}] num_init={args.num_init}, base_channel={args.base_channel}, stride={args.stride_init}, tkernel={args.tkernel_init} → "
                      f"FLOPs: {macs / 1e9:.7f} GFLOPs, Params: {params / 1e6:.7f} M")
        
        ###############################################
        # m = 'stgcn-lw'
        
        # num_init_list = [1, 2, 3, 4]
        # base_channel_list = [64, 128, 159, 195,256]
        # stride_init_list = [1, 2, 3]
        # tkernel_init_list = [3, 5, 7]

        # combinations = itertools.product(num_init_list, base_channel_list, stride_init_list, tkernel_init_list)
        # records = []
        # for num_init, base_channel, stride_init, tkernel_init in combinations:
        #     args.model = m
        #     args.num_in = 0
        #     args.num_main = 0
        #     args.num_init = num_init
        #     args.base_channel = base_channel
        #     args.stride_init = stride_init
        #     args.tkernel_init = tkernel_init

        #     model = load_model(args)

        #     input_tensor = torch.randn(1, 1, 64, 17, 2)
        #     macs, params = profile(model, inputs=(input_tensor,))

        #     writer.writerow([
        #         m,
        #         num_init,
        #         base_channel,
        #         stride_init,
        #         tkernel_init,
        #         f"{macs / 1e9:.7f}",
        #         f"{params / 1e6:.7f}"
        #     ])

        #     print(f"[{m}] num_init={num_init}, base_channel={base_channel}, stride={stride_init}, tkernel={tkernel_init} → "
        #             f"FLOPs: {macs / 1e9:.7f} GFLOPs, Params: {params / 1e6:.7f} M")
            
        #     records.append([
        #         {'STGCN++'},
        #         num_init,
        #         base_channel,
        #         stride_init,
        #         tkernel_init,
        #         f"{macs / 1e9:.7f}",
        #         f"{params / 1e6:.7f}"
        #     ])

        ##########################################################################################
        # SSO-STGCN 使用
        # records = []

        # args.model = 'stgcn'
        # args.num_in = 0
        # args.num_main = 0

        # args.num_init = 3
        # args.base_channel = 206
        # args.stride_init = 1
        # args.tkernel_init = 3
        # args.act = 3

        # model = load_model(args)

        # input_tensor = torch.randn(1, 1, 64, 17, 2)
        # macs, params = profile(model, inputs=(input_tensor,))

        # writer.writerow([
        #     args.model,
        #     args.num_init,
        #     args.base_channel,
        #     args.stride_init,
        #     args.tkernel_init,
        #     f"{macs / 1e9:.7f}",
        #     f"{params / 1e6:.7f}"
        # ])

        # print(f"[{args.model}] num_init={args.num_init}, base_channel={args.base_channel}, stride={args.stride_init}, tkernel={args.tkernel_init} → "
        #         f"FLOPs: {macs / 1e9:.7f} GFLOPs, Params: {params / 1e6:.7f} M")
        
        # records.append([
        #     {'STGCN++'},
        #     args.num_init,
        #     args.base_channel,
        #     args.stride_init,
        #     args.tkernel_init,
        #     f"{macs / 1e9:.7f}",
        #     f"{params / 1e6:.7f}"
        # ])

        # import pandas as pd

        # df = pd.DataFrame(records, columns=[
        #     "Model", "num_init", "base_channel", "stride_init", "tkernel_init", "FLOPs (GFLOPs)", "Params (M)"
        # ])
        
        # save_path = "colab/visulization/params_and_flops.csv"
        # df.to_csv(save_path, index=False)
        # print(f"✅ 成功儲存至 {save_path}")

        # import pandas as pd

        # df = pd.DataFrame(records, columns=[
        #     "Model", "num_init", "base_channel", "stride_init", "tkernel_init", "FLOPs (GFLOPs)", "Params (M)"
        # ])
        
        # save_path = "colab/visulization/params_and_flops.csv"
        # df.to_csv(save_path, index=False)
        # print(f"✅ 成功儲存至 {save_path}")
            


if __name__ == '__main__':
   compute()