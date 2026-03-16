"""
trainer/train.py
================
酒醉步態辨識的主訓練腳本（Drunk Gait Classification Trainer）。

功能：
    - 支援多種模型（STGCN++, CTRGCN, DGSTGCN, AAGCN, SGN, 等）
    - 損失函數：CrossEntropy（含類別權重）或 LabelSmoothing CE
    - 可選加入 Triplet Loss（--mt 旗標）
    - 多粒度評估：4類 / 3類 / 2類 Accuracy、Precision、Recall、F1
    - 訓練超時保護（multiprocessing + GPU 監控）
    - 結果儲存至 results/ 目錄

主要函數：
    train(args)              → 執行單次訓練，回傳 (results_dict, updated_args)
    train_with_timeout(args) → 含超時與 GPU 監控保護的訓練包裝器

命令列使用（直接執行）：
    python -m trainer.train --data drunk --model stgcn --split 8 --epochs 80 --mt
"""

import os
import time
import random
import uuid

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchmetrics
import argparse
import multiprocessing as mp
import threading

from timm.scheduler.cosine_lr import CosineLRScheduler
from pytorch_metric_learning import losses
from datetime import datetime

from utils.loss    import LabelSmoothingCrossEntropy as LSCE
from utils.metrics import (
    compute_precision_recall_f1,
    compute_accuracy,
    merge_cm_to_2_classes,
    merge_cm_to_3_classes,
)
from utils.gpu_monitor import monitor_gpu_usage

# ─────────────────────────────────────────────────────────────────────────────
# 命令列參數定義
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Drunk Gait Classification Trainer")

# 資料集
parser.add_argument("--data",    type=str, default="drunk",     help="資料集名稱（drunk / nw-ucla / ...）")
parser.add_argument("--split",   type=int, default=1,           help="資料集 split 編號")
parser.add_argument("--cls",     type=int, default=4,           help="分類類別數（2/3/4）")
parser.add_argument("--view",    type=str, default="sagittal",  help="視角（sagittal / coronal）")
parser.add_argument("--pose",    type=str, default="yolo",      help="姿態估計器（yolo / hrnet）")
parser.add_argument("--frame",   type=int, default=90,          help="每個樣本的 frame 數")
parser.add_argument("--feature", type=str, default="j",         help="特徵類型（j / b / jm / bm）")
parser.add_argument("--person",  type=int, default=1,           help="人數（1 / 2）")

# 模型選擇
parser.add_argument("--model",   type=str, default="stgcn",     help="模型名稱（stgcn / ctrgcn / dgstgcn / aagcn / sgn / ...）")

# NAS 結構參數（STGCN++ stream 設定）
parser.add_argument("--base_channel", type=int,   default=64,   help="基礎通道數")
parser.add_argument("--num_init",     type=int,   default=4,    help="init stream 層數")
parser.add_argument("--tkernel_init", type=int,   default=3,    help="init stream TCN kernel size")
parser.add_argument("--stride_init",  type=int,   default=1,    help="init stream stride")
parser.add_argument("--num_in",       type=int,   default=3,    help="input stream 層數")
parser.add_argument("--tkernel_in",   type=int,   default=3,    help="input stream TCN kernel size")
parser.add_argument("--stride_in",    type=int,   default=2,    help="input stream stride")
parser.add_argument("--oc_in",        type=int,   default=128,  help="input stream output channels")
parser.add_argument("--num_main",     type=int,   default=3,    help="main stream 層數")
parser.add_argument("--tkernel_main", type=int,   default=3,    help="main stream TCN kernel size")
parser.add_argument("--stride_main",  type=int,   default=2,    help="main stream stride")
parser.add_argument("--oc_main",      type=int,   default=256,  help="main stream output channels")
parser.add_argument("--act",          type=int,   default=0,    help="激活函數（0:ReLU / 1:GELU）")

# 訓練超參數（HPO 搜索空間）
parser.add_argument("--epochs",       type=int,   default=80,   help="訓練 epoch 數")
parser.add_argument("--warm_up_epoch",type=int,   default=5,    help="Warm-up epoch 數")
parser.add_argument("--batch",        type=int,   default=16,   help="Batch size")
parser.add_argument("--lr",           type=float, default=0.1,  help="學習率")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="L2 正則化係數")
parser.add_argument("--momentum",     type=float, default=0.9,  help="SGD momentum")
parser.add_argument("--dropout_bk",  type=float, default=0.0,   help="Backbone dropout")
parser.add_argument("--dropout_fc",  type=float, default=0.0,   help="FC dropout")
parser.add_argument("--opt",          type=int,   default=0,    help="優化器（0:SGD / 1:Adam / 2:AdamW）")
parser.add_argument("--scheduler",    type=int,   default=0,    help="排程器（0:Cosine+Warmup / 1:Cosine / 2:None）")

# Triplet Loss
parser.add_argument("--mt",           action="store_true", default=False,
                    help="是否啟用 Metric Triplet Loss")
parser.add_argument("--lambda_val",   type=float, default=0.5,
                    help="Triplet Loss 比重（loss = (1-λ)*CE + λ*Triplet）")
parser.add_argument("--margin",       type=float, default=0.6,
                    help="Triplet Loss margin")

# 損失函數
parser.add_argument("--loss",         type=str,  default="CE",
                    help="損失函數（CE / LSCE）")
parser.add_argument("--avg",          type=str,  default="micro",
                    help="Accuracy 計算方式（micro / macro）")

# 執行控制
parser.add_argument("--seed",         type=int,   default=42,   help="隨機種子")
parser.add_argument("--run",          type=int,   default=1,    help="重複執行次數")
parser.add_argument("--patience",     type=int,   default=10,   help="Early stop patience")
parser.add_argument("--timeout",      type=int,   default=1000, help="單次訓練超時秒數")
parser.add_argument("--test",         action="store_true",      help="以 test acc 為優化目標（預設用 val acc）")
parser.add_argument("--partition",    action="store_true",      help="是否使用 partition 技巧")
parser.add_argument("--input_channel",type=int,   default=3,    help="輸入通道數（PD dataset 為 2）")

args, _ = parser.parse_known_args()

# ─────────────────────────────────────────────────────────────────────────────
# 全局變數（training 子進程共用）
# ─────────────────────────────────────────────────────────────────────────────

# 由 train() 內部設定，避免在 module 層級使用 args（方便從外部呼叫時傳入覆蓋後的 args）
tri_loss = None
loss_fn  = None
cm       = None          # torchmetrics.ConfusionMatrix

# 類別名稱
CLASS_NAMES = {
    4: ["Normal", "Low", "Medium", "High"],
    3: ["Normal", "Low", "High"],
    2: ["Normal", "Drunk"],
}

NUM_WORKERS = 2          # DataLoader worker 數
MAX_GPU_USAGE = 0.95     # GPU 記憶體使用上限（超過則強制終止）

# ─────────────────────────────────────────────────────────────────────────────
# 工具函數
# ─────────────────────────────────────────────────────────────────────────────

def init_seed(seed: int = 42):
    """設定全局隨機種子，確保實驗可重現。"""
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled      = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


def compute_class_weights(loader, num_classes: int, device) -> torch.Tensor:
    """
    根據 DataLoader 中各類別的樣本數計算反比例類別權重。
    
    公式：weight_c = total / (count_c + ε)，再正規化使平均權重為 1。
    """
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    for batch in loader:
        labels = batch["label"].view(-1)
        for c in range(num_classes):
            class_counts[c] += (labels == c).sum().item()
    total = class_counts.sum()
    weights = total / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights.to(device)


def update_metrics(preds, labels):
    """更新混淆矩陣（透過 global cm）。"""
    global cm
    cm.update(preds, labels)


def reset_metrics():
    """重置混淆矩陣。"""
    global cm
    cm.reset()


def compute_metrics(avg: str = "micro") -> dict:
    """
    從混淆矩陣計算多粒度指標。

    Returns
    -------
    dict with keys:
        acc_4c, acc_3c, acc_2c,
        precision_2c, recall_2c, f1_2c
    """
    global cm
    cm_4c = cm.compute().cpu().numpy()
    cm_3c = merge_cm_to_3_classes(cm_4c)
    cm_2c = merge_cm_to_2_classes(cm_4c)

    acc_4c = compute_accuracy(cm_4c, avg)
    acc_3c = compute_accuracy(cm_3c, avg)
    acc_2c = compute_accuracy(cm_2c, avg)

    # pos_label=1 → 酒醉類別為 Positive
    precision, recall, f1 = compute_precision_recall_f1(cm_2c, pos_label=1)
    return {
        "acc_4c": acc_4c, "acc_3c": acc_3c, "acc_2c": acc_2c,
        "precision_2c": precision, "recall_2c": recall, "f1_2c": f1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 資料集載入
# ─────────────────────────────────────────────────────────────────────────────

def load_drunk_dataset(args):
    """
    載入 drunk71_small 資料集的 train / val / test DataLoader。

    資料路徑格式：
        data/drunk71_small/dataset_{pose}_{view}_split{split}_{frame}frames_j_{cls}class_{split}.npz
    """
    from utils.datasets_npz_aug_2c_flip import load_npz_dataset

    base = (
        f"../data/drunk71_small/dataset_{args.pose}_{args.view}"
        f"_split{args.split}_{args.frame}frames_j_{args.cls}class"
    )
    train_loader = load_npz_dataset(
        f"{base}_train.npz", workers_per_gpu=NUM_WORKERS,
        batch_size=args.batch, pose=args.pose, feature=args.feature,
    )
    val_loader = load_npz_dataset(
        f"{base}_val.npz", workers_per_gpu=NUM_WORKERS,
        batch_size=args.batch, pose=args.pose, feature=args.feature,
    )
    test_loader = load_npz_dataset(
        f"{base}_test.npz", workers_per_gpu=NUM_WORKERS,
        batch_size=args.batch, pose=args.pose, feature=args.feature,
    )
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 模型建立工廠
# ─────────────────────────────────────────────────────────────────────────────

def build_model(args, num_point: int, device):
    """
    根據 args.model 建立對應的骨架動作辨識模型並移至 device。

    支援的模型（args.model 值）：
        stgcn       → STGCN++（NAS 可搜索結構，支援 Triplet Loss）
        stgcn-raw   → STGCN++（原始固定結構）
        ctrgcn      → CTRGCN（支援 Triplet Loss）
        ctrgcn-raw  → CTRGCN（原始固定結構）
        dgstgcn     → DG-STGCN
        aagcn       → AA-GCN
        sgn         → SGN
        skateformer → SkateFormer
        lstm1/2     → Simple LSTM（1/2 層）
        plstm1/2    → Parallel LSTM（1/2 層）
    """
    m = args.model.lower()

    if m in ("stgcn", "stgcn-raw"):
        from models.stgcnpp_sso import SSO_STGCN
        tcn_type   = "mstcn"    if m == "stgcn"     else "unit_tcn"
        gcn_ada    = "init"     if m == "stgcn"     else "importance"
        gcn_res    = True       if m == "stgcn"     else False
        model = SSO_STGCN(
            graph_cfg     = dict(layout=args.pose, mode="spatial"),
            in_channels   = args.input_channel,
            data_bn_type  = "VC",
            num_person    = args.person,
            pretrained    = None,
            tcn_type      = tcn_type,
            tcn_dropout   = args.dropout_bk,
            gcn_with_res  = gcn_res,
            gcn_adaptive  = gcn_ada,
            args          = args,
        )

    elif m in ("ctrgcn", "ctrgcn-raw"):
        from models.ctrgcn_sso import CTRGCN
        model = CTRGCN(
            graph_cfg  = dict(layout=args.pose, mode="spatial"),
            num_classes = args.cls,
            num_point  = num_point,
            num_person = args.person,
            in_channels = args.input_channel,
            drop_out   = args.dropout_fc if m == "ctrgcn" else 0,
        )

    elif m == "dgstgcn":
        from models.dgstgcn import DGSTGCN
        model = DGSTGCN(
            dict(layout=args.pose, mode="random", num_filter=8, init_off=.04, init_std=.02),
            gcn_ratio  = 0.125,
            gcn_ctr    = "T",
            gcn_ada    = "T",
            tcn_ms_cfg = [(3,1),(3,2),(3,3),(3,4),("max",3),"1x1"],
            num_classes = args.cls,
            num_person = args.person,
            in_channels = args.input_channel,
        )

    elif m == "aagcn":
        from models.aagcn import AAGCN
        model = AAGCN(
            dict(layout=args.pose, mode="spatial"),
            num_classes = args.cls,
            num_person = args.person,
            in_channels = args.input_channel,
        )

    elif m == "sgn":
        from models.sgn import SGN
        model = SGN(in_channels=2, base_channels=64, num_joints=17, T=64, bias=True)

    elif m == "skateformer":
        from models.skateFormer import SkateFormer
        model = SkateFormer(
            in_channels=2, num_people=1, num_classes=4, num_points=17,
            kernel_size=7, num_heads=32, attn_drop=0.5, head_drop=0.0,
            drop_path=0.2, type_1_size=[8,8], type_2_size=[8,12],
            type_3_size=[8,8], type_4_size=[8,12],
            mlp_ratio=4.0, index_t=True,
        )

    elif m in ("lstm1", "lstm2"):
        from models.lstm import SimpleLSTM
        model = SimpleLSTM(input_dim=17*2, hidden_dim=75,
                           num_layers=int(m[-1]), num_classes=4)
        args.opt, args.scheduler, args.lr = 1, None, 0.001

    elif m in ("plstm1", "plstm2"):
        from models.lstm import PLSTM
        model = PLSTM(input_dim=17*2, hidden_dim=75,
                      num_layers=int(m[-1]), num_classes=4)
        args.opt, args.scheduler, args.lr = 1, None, 0.001

    else:
        raise ValueError(f"不支援的模型名稱：{args.model}")

    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 訓練 / 驗證 / 測試步驟
# ─────────────────────────────────────────────────────────────────────────────

def train_step(model, optimizer, loader, tb_writer, epoch, scheduler, args):
    """
    單個 epoch 的訓練步驟。

    Returns
    -------
    dict : 包含 loss, acc_4c, acc_3c, acc_2c, precision_2c, recall_2c, f1_2c
    """
    global global_step, tri_loss, loss_fn
    model.train()
    running_loss = 0.0
    reset_metrics()

    for batch in loader:
        inputs, labels = batch["keypoint"].to(args.device), batch["label"].to(args.device)
        optimizer.zero_grad()

        outputs, embedding = model(inputs)

        # 損失計算：CE 或 CE + Triplet
        if args.mt:
            loss = (1 - args.lambda_val) * loss_fn(outputs, labels) \
                 + args.lambda_val * tri_loss(embedding, labels)
        else:
            loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        # Scheduler 更新（Cosine+Warmup 需 step 級別更新）
        if args.scheduler == 0:
            global_step += 1
            scheduler.step_update(global_step)
        elif args.scheduler == 1:
            scheduler.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        update_metrics(preds, labels)

    metrics = compute_metrics(args.avg)
    avg_loss = running_loss / len(loader)

    if tb_writer:
        e = epoch + 1
        tb_writer.add_scalar("Loss/train/epoch",         avg_loss,                e)
        tb_writer.add_scalar("Accuracy/train/epoch_4c",  metrics["acc_4c"],       e)
        tb_writer.add_scalar("Accuracy/train/epoch_2c",  metrics["acc_2c"],       e)
        tb_writer.add_scalar("F1-score/train/epoch_2c",  metrics["f1_2c"],        e)

    return {"loss": avg_loss, **metrics}


def validate_step(model, loader, tb_writer, epoch):
    """單個 epoch 的驗證步驟。"""
    global loss_fn
    model.eval()
    running_loss = 0.0
    reset_metrics()

    with torch.no_grad():
        for batch in loader:
            inputs = batch["keypoint"].to(args.device)
            labels = batch["label"].to(args.device)
            if labels.dim() > 1 and labels.size(-1) == 1:
                labels = labels.squeeze(-1)
            outputs, _ = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            update_metrics(preds, labels)

    metrics = compute_metrics(args.avg)
    avg_loss = running_loss / len(loader)

    if tb_writer:
        e = epoch + 1
        tb_writer.add_scalar("Loss/valid/epoch",        avg_loss,          e)
        tb_writer.add_scalar("Accuracy/valid/epoch_4c", metrics["acc_4c"], e)
        tb_writer.add_scalar("Accuracy/valid/epoch_2c", metrics["acc_2c"], e)
        tb_writer.add_scalar("F1-score/valid/epoch_2c", metrics["f1_2c"],  e)

    return {"loss": avg_loss, **metrics}


def test_step(model, loader, tb_writer, epoch, save_path: str, checkpt_file: str):
    """
    測試步驟：載入最佳 checkpoint，評估測試集，儲存預測結果 CSV。

    Returns
    -------
    (dict, args) : 測試指標字典 和 更新後的 args（含 final_ckpt 路徑）
    """
    global loss_fn
    import pandas as pd

    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    running_loss = 0.0
    reset_metrics()
    all_records = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["keypoint"].to(args.device)
            labels = batch["label"].to(args.device)
            names  = batch["name"]
            if labels.dim() > 1 and labels.size(-1) == 1:
                labels = labels.squeeze(-1)
            outputs, _ = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            update_metrics(preds, labels)
            for name, pred, label in zip(names, preds.cpu().tolist(), labels.cpu().tolist()):
                all_records.append({"name": name, "pred": pred, "label": label})

    metrics  = compute_metrics(args.avg)
    avg_loss = running_loss / len(loader)

    # 儲存逐樣本預測結果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("pred_results", exist_ok=True)
    csv_path = f"pred_results/pred_results_split{args.split}_{timestamp}.csv"
    pd.DataFrame(all_records).to_csv(csv_path, index=False)
    print(f"📊 預測結果已儲存至：{csv_path}")

    # 刪除暫存 checkpoint，儲存最終模型
    if os.path.exists(checkpt_file):
        os.remove(checkpt_file)
    args.final_ckpt = (
        f"pretrained/{uuid.uuid4().hex[:8]}"
        f"_data_{args.data}_{args.split}_{args.feature}_{args.model}"
        f"_acc{metrics['acc_2c']:.6f}.pt"
    )
    torch.save(model.state_dict(), args.final_ckpt)

    return {"loss": avg_loss, **metrics}, args


def save_results(args, results: dict, filename: str = "results/record.txt"):
    """
    將訓練與測試指標儲存為可讀的 .txt 記錄檔。

    Parameters
    ----------
    args : argparse.Namespace
        訓練參數（含 final_ckpt）
    results : dict
        包含 train_time, test_time, val_metrics, test_metrics
    filename : str
        輸出路徑（預設 results/record.txt）
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Checkpt_file:\n")
        f.write(f"  {args.final_ckpt}\n\n")

        f.write("Arguments:\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")

        f.write("\nResults Summary:\n")
        f.write(f"  - Train Time: {results['train_time']:.4f}s\n")
        f.write(f"  - Test  Time: {results['test_time']:.4f}s\n")

        for phase, key in [("Validation", "val_metrics"), ("Test", "test_metrics")]:
            m = results[key]
            f.write(f"\n  [{phase}]\n")
            f.write(f"    Acc 4C: {m['acc_4c']*100:.4f}%\n")
            f.write(f"    Acc 3C: {m['acc_3c']*100:.4f}%\n")
            f.write(f"    Acc 2C: {m['acc_2c']*100:.4f}%\n")
            f.write(f"    Precision 2C : {m['precision_2c']:.4f}\n")
            f.write(f"    Recall    2C : {m['recall_2c']:.4f}\n")
            f.write(f"    F1        2C : {m['f1_2c']:.4f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 主訓練函數（可被 SSO / GA / PSO fitness 函數呼叫）
# ─────────────────────────────────────────────────────────────────────────────

def train(local_args):
    """
    執行單次完整訓練（從資料載入到測試輸出）。

    Parameters
    ----------
    local_args : argparse.Namespace
        所有超參數設定（可被啟發式演算法動態覆蓋）。

    Returns
    -------
    (results, args)
        results : dict，包含 train_time, test_time, val_metrics, test_metrics
        args    : 更新後的 args（加上 final_ckpt 路徑）
    """
    global cm, global_step, tri_loss, loss_fn, args
    args = local_args

    init_seed(args.seed)
    global_step = 0

    # TensorBoard writer
    from torch.utils.tensorboard import SummaryWriter
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs/{args.data}_{args.feature}_{args.split}_{args.epochs}_{timestamp}")

    # 臨時 checkpoint（best model during training）
    checkpt_file = f"pretrained/{uuid.uuid4().hex}.pt"
    args.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. 資料集載入 ─────────────────────────────────────────────────
    num_point = 17  # COCO/YOLO skeleton
    args.input_channel = 2

    if args.data == "drunk":
        train_loader, val_loader, test_loader = load_drunk_dataset(args)
    else:
        raise ValueError(f"目前只支援 'drunk' 資料集，收到：{args.data}")

    # ── 2. 混淆矩陣初始化 ─────────────────────────────────────────────
    cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=args.cls).to(args.device)

    # ── 3. 模型建立 ───────────────────────────────────────────────────
    model = build_model(args, num_point=num_point, device=args.device)

    # ── 4. 優化器 ─────────────────────────────────────────────────────
    if args.opt == 0:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum, nesterov=True)
    elif args.opt == 1:
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    elif args.opt == 2:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)

    # ── 5. 排程器 ─────────────────────────────────────────────────────
    if args.scheduler == 0:
        if args.warm_up_epoch >= args.epochs:
            args.warm_up_epoch = max(0, args.epochs - 1)
            
        n_iter     = len(train_loader)
        num_steps  = int(args.epochs * n_iter)
        warmup_steps = int(args.warm_up_epoch * n_iter)
        scheduler = CosineLRScheduler(
            optimizer, t_initial=(num_steps - warmup_steps), lr_min=1e-5,
            warmup_lr_init=1e-7, warmup_t=warmup_steps,
            cycle_limit=1, t_in_epochs=False, warmup_prefix=False,
        )
    elif args.scheduler == 1:
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * len(train_loader), eta_min=0)
    else:
        scheduler = None

    # ── 6. 損失函數 ───────────────────────────────────────────────────
    class_weights = compute_class_weights(test_loader, args.cls, args.device)
    loss_fn  = LSCE(smoothing=0.1) if args.loss == "LSCE" \
               else torch.nn.CrossEntropyLoss(weight=class_weights)
    tri_loss = losses.TripletMarginLoss(margin=args.margin)

    # ── 7. 訓練迴圈 ───────────────────────────────────────────────────
    best_loss = float("inf")
    bad_counter = 0
    repo_val_metrics = None
    start_time = time.time()

    for epoch in range(args.epochs):
        train_metrics = train_step(model, optimizer, train_loader, writer, epoch, scheduler, args)
        val_metrics   = validate_step(model, val_loader, writer, epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {epoch+1:04d} | lr: {current_lr:.6f}")
        print(
            f" ├── [Train] Loss: {train_metrics['loss']:.4f} | "
            f"Acc 4C: {train_metrics['acc_4c']*100:.2f}% | "
            f"Acc 2C: {train_metrics['acc_2c']*100:.2f}% | "
            f"F1 2C: {train_metrics['f1_2c']:.4f}"
        )
        print(
            f" └── [Val]   Loss: {val_metrics['loss']:.4f} | "
            f"Acc 4C: {val_metrics['acc_4c']*100:.2f}% | "
            f"Acc 2C: {val_metrics['acc_2c']*100:.2f}% | "
            f"F1 2C: {val_metrics['f1_2c']:.4f}"
        )

        # Early stopping（以 val loss 為準）
        if val_metrics["loss"] <= best_loss:
            os.makedirs(os.path.dirname(checkpt_file), exist_ok=True)
            best_loss = val_metrics["loss"]
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
            repo_val_metrics = val_metrics
        else:
            bad_counter += 1

        if bad_counter >= args.patience:
            print(f"⚠️  Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - start_time

    # ── 8. 測試 ───────────────────────────────────────────────────────
    timestamp2 = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_test = time.time()
    test_metrics, args = test_step(
        model, test_loader, writer, epoch,
        save_path=f"results/confusion_matrix_{timestamp2}.png",
        checkpt_file=checkpt_file,
    )
    test_time = time.time() - start_test

    results = {
        "train_time"  : train_time,
        "test_time"   : test_time,
        "val_metrics" : repo_val_metrics,
        "test_metrics": test_metrics,
    }
    return results, args


# ─────────────────────────────────────────────────────────────────────────────
# 超時保護包裝器（用於 SSO / GA / PSO 適應度函數呼叫）
# ─────────────────────────────────────────────────────────────────────────────

def _train_target(args, result_queue):
    """子進程執行函數，將結果放入 Queue。"""
    try:
        results, updated_args = train(args)
        result_queue.put(("success", results, updated_args))
    except Exception as e:
        import traceback
        result_queue.put(("error", f"{e}\n{traceback.format_exc()}"))


def train_with_timeout(args, timeout: int = 1000,
                       max_gpu_usage: float = MAX_GPU_USAGE):
    """
    執行訓練，提供超時保護與 GPU 記憶體監控。

    若訓練在 timeout 秒內未完成，或 GPU 記憶體超過 max_gpu_usage，
    則強制終止（回傳 None）。

    Parameters
    ----------
    args : argparse.Namespace
    timeout : int
        最大訓練時間（秒）
    max_gpu_usage : float
        GPU 記憶體使用率上限（0~1）

    Returns
    -------
    (results, updated_args) 或 (None, None)（若超時或 GPU 超限）
    """
    result_queue = mp.Queue()
    process      = mp.Process(target=_train_target, args=(args, result_queue))
    process.start()

    stop_event     = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_gpu_usage,
        args=(process, max_gpu_usage, stop_event),
    )
    monitor_thread.start()

    start = time.time()
    try:
        while time.time() - start < timeout:
            if not process.is_alive():
                try:
                    status, *rest = result_queue.get_nowait()
                    if status == "error":
                        print(f"🚨 訓練過程發生錯誤:\n{rest[0]}")
                        return None, None
                    return rest[0], rest[1]  # results, updated_args
                except mp.queues.Empty:
                    print("🚨 進程異常結束，未回傳結果！")
                    return None, None

            if stop_event.is_set():
                print("🚨 GPU 記憶體超限，訓練已終止！")
                process.terminate()
                return None, None

            time.sleep(1)

        print(f"⏳ 訓練超時（{timeout}s），終止進程！")
        process.terminate()
        return None, None
    finally:
        process.join()
        monitor_thread.join()
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# 命令列直接執行
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    t_total = time.time()

    # 收集多次執行的指標
    metrics_history = {k: [] for k in [
        "val_acc_4c", "val_acc_3c", "val_acc_2c",
        "test_acc_4c", "test_acc_3c", "test_acc_2c",
        "test_precision", "test_recall", "test_f1",
        "train_time", "test_time",
    ]}
    log_file = "None"

    for i in range(args.run):
        results, args = train_with_timeout(args, timeout=args.timeout)

        if results is None:
            print(f"Run {i}: 超時（{args.timeout}s），跳過")
            for k in metrics_history:
                metrics_history[k].append(0)
        else:
            metrics_history["val_acc_4c"].append(results["val_metrics"]["acc_4c"])
            metrics_history["val_acc_3c"].append(results["val_metrics"]["acc_3c"])
            metrics_history["val_acc_2c"].append(results["val_metrics"]["acc_2c"])
            metrics_history["test_acc_4c"].append(results["test_metrics"]["acc_4c"])
            metrics_history["test_acc_3c"].append(results["test_metrics"]["acc_3c"])
            metrics_history["test_acc_2c"].append(results["test_metrics"]["acc_2c"])
            metrics_history["test_precision"].append(results["test_metrics"]["precision_2c"])
            metrics_history["test_recall"].append(results["test_metrics"]["recall_2c"])
            metrics_history["test_f1"].append(results["test_metrics"]["f1_2c"])
            metrics_history["train_time"].append(results["train_time"])
            metrics_history["test_time"].append(results["test_time"])

            # 儲存當次結果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = (
                f"results/record_{timestamp}_{args.data}_{args.feature}_split{args.split}_"
                f"4C-{metrics_history['test_acc_4c'][-1]:.4f}_"
                f"3C-{metrics_history['test_acc_3c'][-1]:.4f}_"
                f"2C-{metrics_history['test_acc_2c'][-1]:.4f}_"
                f"2CP-{metrics_history['test_precision'][-1]:.4f}_"
                f"2CR-{metrics_history['test_recall'][-1]:.4f}_"
                f"2CF-{metrics_history['test_f1'][-1]:.4f}.txt"
            )
            save_results(args, results, filename=log_file)

    # 輸出彙整結果（供 tools.py 的 prinfo 解析）
    print(f"Train cost: {time.time() - t_total:.4f}s")
    for key in ["val_acc_4c", "val_acc_3c", "val_acc_2c",
                "test_acc_4c", "test_acc_3c", "test_acc_2c",
                "test_f1", "test_precision", "test_recall",
                "train_time", "test_time"]:
        print(f"{key.replace('_', '_').title().replace('_', '_')}_list: {metrics_history[key]}")
    print(f"Log file: {log_file}")
