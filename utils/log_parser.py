"""
utils/log_parser.py
===================
實驗 Log 解析工具。

從訓練腳本（trainer/train.py）輸出的 stdout 字串中，
提取各類評估指標的列表，供 Notebook 分析使用。

主要函數：
    parse_log(lines)          → 完整版解析（支援 4C/3C F1-score）
    parse_log_simple(lines)   → 簡化版解析（不含 4C/3C F1）
    fix_ckpt_to_result(...)   → 將 SSO/GA/PSO log.pkl 轉為最終結果 .pkl

典型用法（在 Notebook 中）：
    import subprocess
    result = subprocess.run(["python", "-m", "trainer.train", ...],
                            capture_output=True, text=True)
    metrics = parse_log(result.stdout.splitlines())
    print(metrics["test_acc_4c"])
"""

import re
import os
import uuid
import pickle
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 通用 Regex 工具
# ─────────────────────────────────────────────────────────────────────────────

_FLOAT_LIST_PATTERN = r"[\d.e+\-\s,]+"   # 用於 [v1, v2, ...] 列表的 regex
_FLOAT_PATTERN      = r"[\d.e+\-]+"      # 用於單一數值的 regex


def _extract_list(text: str, pattern: str) -> list:
    """從 text 中找到 pattern，回傳 float 列表。找不到則回傳 [0.0]。"""
    m = re.search(pattern, text)
    return list(map(float, m.group(1).split(","))) if m else [0.0]


def _extract_float(text: str, pattern: str) -> float:
    """從 text 中找到 pattern，回傳單一 float。找不到則回傳 0.0。"""
    m = re.search(pattern, text)
    return float(m.group(1)) if m else 0.0


def _extract_str(text: str, pattern: str) -> str:
    """從 text 中找到 pattern，回傳字串。找不到則回傳 'Unknown'。"""
    m = re.search(pattern, text)
    return m.group(1) if m else "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 完整版解析（支援 F1 4C/3C）
# ─────────────────────────────────────────────────────────────────────────────

def parse_log(lines: list) -> dict:
    """
    解析訓練腳本的 stdout，提取所有評估指標（含 4C/3C F1-score）。

    Parameters
    ----------
    lines : list of str
        訓練腳本的輸出行（可用 stdout.splitlines()）。

    Returns
    -------
    dict 包含以下 key：
        train_cost, train_time, test_time,
        val_acc_4c, val_acc_3c, val_acc_2c,
        test_acc_4c, test_acc_3c, test_acc_2c,
        test_f1, test_precision, test_recall,
        test_f1_4c, test_f1_3c,
        log_file
    """
    text = "\n".join(lines)

    return {
        "train_cost"   : _extract_float(text, rf"Train cost:\s*({_FLOAT_PATTERN})s"),
        "train_time"   : np.mean(_extract_list(text, rf"Train_time_list:\s*\[({_FLOAT_LIST_PATTERN})\]")),
        "test_time"    : np.mean(_extract_list(text, rf"Test_time_list:\s*\[({_FLOAT_LIST_PATTERN})\]")),
        "val_acc_4c"   : np.mean(_extract_list(text, rf"Val_acc_4c_list:\s*\[({_FLOAT_LIST_PATTERN})\]")),
        "val_acc_3c"   : np.mean(_extract_list(text, rf"Val_acc_3c_list:\s*\[({_FLOAT_LIST_PATTERN})\]")),
        "val_acc_2c"   : np.mean(_extract_list(text, rf"Val_acc_2c_list:\s*\[({_FLOAT_LIST_PATTERN})\]")),
        "test_acc_4c"  : np.mean(_extract_list(text, rf"Test_acc_4c_list:\s*\[({_FLOAT_LIST_PATTERN})\]")),
        "test_acc_3c"  : np.mean(_extract_list(text, rf"Test_acc_3c_list:\s*\[({_FLOAT_LIST_PATTERN})\]")),
        "test_acc_2c"  : np.mean(_extract_list(text, rf"Test_acc_2c_list:\s*\[({_FLOAT_LIST_PATTERN})\]")),
        "test_f1"      : np.mean(_extract_list(text, rf"Test_f1_list:\s*\[({_FLOAT_LIST_PATTERN})\]")),
        "test_precision": np.mean(_extract_list(text, rf"Test_precision_list:\s*\[({_FLOAT_LIST_PATTERN})\]")),
        "test_recall"  : np.mean(_extract_list(text, rf"Test_recall_list:\s*\[({_FLOAT_LIST_PATTERN})\]")),
        "test_f1_4c"   : np.mean(_extract_list(text, rf"Test_f1_4c_list:\s*\[({_FLOAT_LIST_PATTERN})\]")),
        "test_f1_3c"   : np.mean(_extract_list(text, rf"Test_f1_3c_list:\s*\[({_FLOAT_LIST_PATTERN})\]")),
        "log_file"     : _extract_str(text,  rf"Log file:\s*(results/[\w./_-]+\.txt)"),
    }


def parse_log_simple(lines: list) -> dict:
    """
    解析訓練腳本的 stdout（簡化版，不含 F1 4C/3C）。

    Parameters
    ----------
    lines : list of str

    Returns
    -------
    dict 包含：train_cost, train_time, test_time,
               val_acc_*, test_acc_*, test_f1, test_precision, test_recall, log_file
    """
    text = "\n".join(lines)
    result = parse_log(lines)
    # 移除 4C/3C F1 欄位
    result.pop("test_f1_4c", None)
    result.pop("test_f1_3c", None)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint 轉換工具（SSO / GA / PSO 中斷恢復後轉為最終結果）
# ─────────────────────────────────────────────────────────────────────────────

def fix_ckpt_to_result(
    log_path: str,
    data: str = "drunk",
    split: int = None,
    setting: str = None,
) -> bool:
    """
    將中斷產生的暫存 log.pkl（checkpoint）轉換為最終結果 .pkl 並移至 sso_result/。

    使用情境：
        - 訓練意外中斷，但 checkpoint 已包含部分結果
        - 手動終止實驗並想保留先前結果

    Parameters
    ----------
    log_path : str
        暫存 log.pkl 路徑。
    data : str
        資料集名稱。
    split : int
        資料集 split 編號。
    setting : str
        自訂描述字串。

    Returns
    -------
    bool : 成功轉換回傳 True，找不到檔案回傳 False
    """
    if not os.path.exists(log_path):
        print(f"❌ 找不到 checkpoint 檔案：{log_path}")
        return False

    with open(log_path, "rb") as f:
        log = pickle.load(f)

    # 找到最新有效的 gen 與 sol
    valid_gens = [g for g, v in log.items() if v != {}]
    if not valid_gens:
        print("❌ checkpoint 中找不到有效結果")
        return False

    max_gen = max(valid_gens)
    max_sol = max(log[max_gen].keys())

    # 從 checkpoint 中讀取最佳解資訊
    ggen         = log[max_gen][max_sol]["g"][2]
    gsol         = log[max_gen][max_sol]["g"][3]
    search_time  = log[ggen][gsol]["search_time"]

    # 存入 sso_result/
    uid       = uuid.uuid4().hex[:8]
    save_path = f"sso_result/{data}_{split}_{setting}_ggen{ggen}_gsol{gsol}_searchtime{search_time:.1f}_{uid}.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(log, f)

    # 刪除暫存 checkpoint
    os.remove(log_path)
    print(f"✅ Checkpoint 已轉換並儲存至：{save_path}")
    return True
