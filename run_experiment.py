"""
colab/run_experiment.py
=======================
讀取 experiment_config.yaml，使用 SSO, GA 或 PSO 進行超參數/NAS 搜尋。
將搜索演算法產生的解矩陣映射回 argparse 的參數，執行訓練並返回適應度。

執行方式：
    python run_experiment.py --config experiment_config.yaml
"""

import sys
import argparse
import yaml
import importlib
import copy

# 強制 Windows console 使用 UTF-8 輸出，避免印出 emoji 時產生 cp950 編碼錯誤
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


try:
    from algorithms import SSO, GA, PSO
    from trainer.train import train_with_timeout, parser as train_parser
except ImportError as e:
    print(f"匯入錯誤: {e}\n請確保你在 colab/ 目錄下執行此腳本。")
    sys.exit(1)

def load_config(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError:
        print("❌ 未安裝 PyYAML 套件！請執行：pip install pyyaml")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ 找不到設定檔：{config_path}")
        sys.exit(1)

def parse_boundaries(boundary_dict):
    """
    將 YAML 的列表定義轉換為系統接受的邊界格式。
    YAML 中 [[v1, v2]] 表示離散列表，[min, max] 則是 tuple。
    """
    parsed = {}
    ordered_keys = []
    
    for k, v in boundary_dict.items():
        ordered_keys.append(k)
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
            # 如 [[3, 5, 7]] -> 轉為 list [3, 5, 7]
            parsed[k] = v[0]
        elif isinstance(v, list) and len(v) == 2:
            # 如 [1, 4] 或 [0.15, 0.3] -> 轉為 tuple (1, 4) 或 (0.15, 0.3)
            parsed[k] = (v[0], v[1])
        else:
            raise ValueError(f"無法解析的邊界格式：{k}: {v}")
            
    return parsed, ordered_keys

def get_base_args(fixed_args):
    """從 trainer.train 取出基礎 args 藍本，填入 yaml 指定的 fixed_args"""
    # 建立 dummy sys.argv 避免 argparse 吃進命令列參數
    args = train_parser.parse_args([])
    for k, v in fixed_args.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args

def create_fitness_fn(base_args, ordered_keys):
    """建立包裹好 args 轉換的 fitness 函數"""
    def fitness(x):
        """
        將演算法生成的 x 陣列，對應寫入 args，然後執行訓練
        """
        # 深拷貝基礎 args 避免交互污染
        current_args = copy.deepcopy(base_args)
        
        # 建立記錄訊息
        record_msg = {}
        
        # 將 x 維度值指派給 args 屬性
        for i, key in enumerate(ordered_keys):
            val = x[i]
            
            # 根據 base_args 中的預設型別進行轉換
            default_val = getattr(base_args, key, None)
            if isinstance(default_val, bool):
                val = bool(val)
            elif isinstance(default_val, int):
                val = int(val)
            elif isinstance(default_val, float):
                val = float(val)
            else:
                if isinstance(val, (np.integer, int)):
                    val = int(val)
                elif isinstance(val, (np.floating, float)):
                    val = float(val)
                
            setattr(current_args, key, val)
            record_msg[key] = val
        
        print(f"\n[Fitness Eval] 嘗試參數組合:")
        for k in ordered_keys:
            print(f"  {k}: {getattr(current_args, k)}")
            
        # 執行訓練
        results, args_updated = train_with_timeout(current_args, timeout=current_args.timeout)
        
        if results is None:
            print("⚠️ 訓練超時或發生錯誤，適應度為 0")
            return 0.0, record_msg
            
        # 提取用來評估的 val_acc_2c (因應專案習慣，以 2類別 ACC 或 F1 為主)
        # 若 --test flag 有開，也可針對 test_metrics
        val_acc = results["val_metrics"]["acc_2c"]
        record_msg['val_acc'] = val_acc
        record_msg['test_acc'] = results["test_metrics"]["acc_2c"]
        record_msg['test_f1'] = results["test_metrics"]["f1_2c"]
        
        print(f"✅ 完成評估: Val Acc 2C = {val_acc:.4f}, Test Acc 2C = {record_msg['test_acc']:.4f}")
        
        # 回傳適應度值。SSO 預設 minimize，所以回傳負的 val_acc
        return -val_acc, record_msg
        
    return fitness

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment from YAML")
    parser.add_argument("--config", type=str, default="experiment_config.yaml", help="設定檔路徑")
    cmd_args = parser.parse_args()

    # 1. 載入設定
    import numpy as np  # Ensure numpy is available
    config = load_config(cmd_args.config)
    exp_cfg = config["experiment"]
    
    print("=" * 50)
    print(f"🚀 初始化最佳化實驗: {exp_cfg['algorithm']}")
    print("=" * 50)

    # 2. 定義搜索邊界與順序
    boundary, ordered_keys = parse_boundaries(config["boundary"])
    
    # 3. 準備基礎 args
    base_args = get_base_args(config.get("fixed_args", {}))
    
    # 4. 準備 Fitness 函數
    app_fitness = create_fitness_fn(base_args, ordered_keys)
    
    # 5. 建立演算法實體可以
    alg_name = exp_cfg.get("algorithm", "SSO").upper()
    kwargs = {
        "Ngen": exp_cfg.get("Ngen", 10),
        "Nsol": exp_cfg.get("Nsol", 10),
        "save_name": exp_cfg.get("save_name", "experiment"),
        "direction": exp_cfg.get("direction", "minimize"),
        "fitness": app_fitness,
        "boundary": boundary
    }
    
    if alg_name == "SSO":
        optimizer = SSO(**kwargs, **config.get("sso_params", {}))
    elif alg_name == "GA":
        optimizer = GA(**kwargs, **config.get("ga_params", {}))
    elif alg_name == "PSO":
        optimizer = PSO(**kwargs, **config.get("pso_params", {}))
    else:
        print(f"❌ 未知演算法：{alg_name}")
        sys.exit(1)
        
    # 6. 執行搜索
    print("\n⏳ 開始搜索流程...")
    optimizer.run()
    
    print(f"\n🎉 實驗完成！結果已儲存為：{optimizer.save_path}")
    print(f"🏆 最佳參數組合：")
    for k, v in zip(ordered_keys, optimizer.best_params):
        print(f"  {k}: {v}")
