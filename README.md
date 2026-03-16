# Gait Analysis - Drunk Detection Project

基於骨架（Skeleton Based）的酒醉步態辨識研究，
使用圖卷積神經網路（STGCN++ 等）結合啟發式演算法（SSO / GA / PSO）進行最佳化參數搜索（NAS/HPO）。

> **🎓 本專案為碩士論文之相關研究**
> [碩士論文連結 (臺灣博碩士論文知識加值系統)](https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/ccd=H5Lt84/record?r1=2&h1=0)

---

## 💻 環境安裝與啟動

為確保所有依賴能夠正確執行，請按以下步驟建立虛擬環境：

### 1. 建立並啟動虛擬環境 (Python 3.10)
專案建議使用 Python 3.10。請在 `colab/` 目錄下建立名為 `colab_env_310` 的虛擬環境：

**Windows:**
```bash
cd d:\pythonProject\IC Lab\Gait_analysis\pyskl\colab
python -m venv colab_env_310
.\colab_env_310\Scripts\activate
```

**Linux/macOS:**
```bash
python3.10 -m venv colab_env_310
source colab_env_310/bin/activate
```

### 2. 安裝套件
在啟動虛擬環境後，使用 `pip` 依序安裝：
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📁 檔案架構說明

```
colab/
├── algorithms/          # 啟發式最佳化演算法
│   ├── base.py          # 抽象基類（共用 ckpt/log/resume/plot 邏輯）
│   ├── sso.py           # Simplified Swarm Optimization
│   ├── ga.py            # Genetic Algorithm
│   └── pso.py           # Particle Swarm Optimization
│
├── models/              # 骨架動作辨識模型
│   ├── stgcnpp_sso.py   # STGCN++（主力 NAS 模型）
│   ├── ctrgcn_sso.py    # CTRGCN
│   └── ...              # 其他模型 (DG-STGCN, AA-GCN, SkateFormer 等)
│
├── gcns/                # GCN 底層元件（TCN / GCN Block 等）
│
├── trainer/             # 訓練相關邏輯
│   ├── train.py         # 主訓練腳本（含 train_with_timeout 防止卡死）
│   └── predict.py       # 推論腳本（可選）
│
├── utils/               # 工具函數
│   ├── datasets_npz_aug_2c_flip.py  # 資料集讀取
│   ├── metrics.py       # 多類別評估指標
│   ├── loss.py          # LabelSmoothing CE
│   └── gpu_monitor.py   # GPU 記憶體監控
│
├── graph/               # 骨架圖結構定義 (支援多種 layout)
├── run_experiment.py    # ⭐️ NAS / HPO 實驗主要執行入口
└── experiment_config.yaml # 實驗參數配置檔
```

---

## 🚀 程式碼使用說明

我們提供了兩種主要的執行方式：直接訓練與參數搜索（NAS/HPO）。

### 1. 直接進行單次訓練 (Single Training)
可以直接透過 `trainer/train.py` 測試特定的參數組合：

```bash
python -m trainer.train \
    --data drunk --split 1 --model stgcn \
    --epochs 80 --batch 16 --lr 0.1 \
    --mt --lambda_val 0.5 --margin 0.6
```

### 2. 自動化超參數搜索實驗 (HPO / NAS) ⭐️
本專案特色是使用 `experiment_config.yaml` 搭配 `run_experiment.py` 來觸發 SSO/GA/PSO 進行結構與超參數的最佳化搜尋。

**執行方式：**
```bash
python run_experiment.py --config experiment_config_fast_test.yaml
```

**`experiment_config.yaml` 設定檔說明：**
- **experiment**: 指定使用的演算法 (SSO / GA / PSO)、迭代代數 (`Ngen`)、族群大小 (`Nsol`)。
- **fixed_args**: 訓練時固定不變的參數（例如指定的模型 `--model stgcn`）。
- **boundary**: 定義 NAS 與 HPO 的搜索範圍。
  - `[0.01, 0.1]`：連續整數/浮點數範圍。
  - `[[0, 1, 2]]`：特定的離散選項。

---

## ✨ 專案特色與防護機制

在進行網路架構搜索（NAS）時，隨機生成的超參數容易導致硬體資源崩潰。為此，專案實作了以下防護機制：

1. **防卡死機制（Timeout Protection）**: `train_with_timeout` 具自動超時中斷功能，若模型異常導致訓練過久會強制中斷。
2. **GPU 記憶體監控（OOM Prevention）**: 遇到結構過大超過上限時，系統會自動攔截防範並跳過，不影響整個優化的進行。
3. **安全捕捉訓練錯誤**: 避免底層神經網路的計算或型別對應錯誤，造成頂層 SSO 演算法崩潰。

---

## 🗂️ 輸出
- 搜尋結果：`sso_result/` / `pso_result/` / `GA_tri/`
- 訓練紀錄：`results/`
- 模型：`pretrained/`
- TensorBoard：`runs/`
