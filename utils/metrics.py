import numpy as np

def merge_cm_to_3_classes(cm):
    """
    合併類別 1,2 為新類別 12，將 4x4 混淆矩陣轉換為 3x3
    """
    # 將原來的 4x4 矩陣縮減為 3x3
    cm_3 = np.zeros((3, 3), dtype=int)

    # 直接使用 NumPy 陣列加總來合併行與列
    cm_3[0] = [cm[0, 0], cm[0, 1] + cm[0, 2], cm[0, 3]]
    cm_3[1] = [cm[1, 0] + cm[2, 0], cm[1, 1] + cm[1, 2] + cm[2, 1] + cm[2, 2], cm[1, 3] + cm[2, 3]]
    cm_3[2] = [cm[3, 0], cm[3, 1] + cm[3, 2], cm[3, 3]]

    return cm_3

def merge_cm_to_2_classes(cm):
    """
    合併四類別 (0,1,2,3) 為二類別 (0,123)
    """
    # 建立新的 2x2 混淆矩陣
    cm_2 = np.zeros((2, 2), dtype=int)  

    # 第一類 (0) 只保留原本的 0 類別數據
    cm_2[0, 0] = cm[0, 0]
    cm_2[0, 1] = np.sum(cm[0, 1:])  # 0 分類錯到 1,2,3 的數據

    # 第二類 (123) 合併 1,2,3
    cm_2[1, 0] = np.sum(cm[1:, 0])  # 1,2,3 錯分類為 0 的數據
    cm_2[1, 1] = np.sum(cm[1:, 1:])  # 1,2,3 的正確分類數據 + 錯分類到彼此的數據

    return cm_2

def compute_accuracy(cm, avg='micro'):
    if avg == 'micro':

        """
        從混淆矩陣計算 Accuracy
        (對角線相加/總和)
        """
        return np.trace(cm) / np.sum(cm)
    
    elif avg == 'macro':
        """
        從混淆矩陣計算 Macro Accuracy
        (每個類別獨立計算 Accuracy，最後取平均)
        """
        per_class_acc = np.zeros(cm.shape[0])  # 儲存每個類別的 Accuracy

        for i in range(cm.shape[0]):
            class_total = np.sum(cm[i, :])  # 該類別的所有樣本數 (TP + FN)
            if class_total > 0:
                per_class_acc[i] = cm[i, i] / class_total  # TP / (TP + FN)
            else:
                per_class_acc[i] = 0.0  # 避免 0 除錯誤

        return np.mean(per_class_acc)  # 取所有類別的平均
    
    else:
        print('Avg should be micro or macro.')


def compute_precision_recall_f1(cm, pos_label=1):
    """
    計算 Precision, Recall, F1-score，允許選擇 Positive Label（pos_label）。
    pred, labels
    [[TN, FN], [FP TP]] >>>>>>是我建立的混淆矩陣，所以Precision Recall要反著看
    
    :param cm: 2x2 混淆矩陣，形如 [[TN, FP], [FN, TP]]
    :param pos_label: 指定哪個類別為 Positive（0 或 1），預設為 1
    :return: 元組 (precision, recall, f1_score)
        - precision: 精確率 TP / (TP + FP)
        - recall: 召回率 TP / (TP + FN)
        - f1_score: F1 分數 2 * (precision * recall) / (precision + recall)
    :raises ValueError: 如果 pos_label 不是 0 或 1，或 cm 不是 2x2 矩陣
    """
    # 檢查混淆矩陣是否為 2x2
    if not (hasattr(cm, 'shape') and cm.shape == (2, 2)):
        raise ValueError("cm 必須是 2x2 的混淆矩陣")

    if pos_label == 1:  # 1 作為 Positive 類別
        TP = cm[1, 1]  # 正確預測 Positive
        FP = cm[0, 1]  # 負類錯誤分類為 Positive
        FN = cm[1, 0]  # Positive 錯誤分類為負類
    elif pos_label == 0:  # 0 作為 Positive 類別
        TP = cm[0, 0]  # 正確預測 Negative
        FP = cm[1, 0]  # Positive 錯誤分類為 Negative
        FN = cm[0, 1]  # Negative 錯誤分類為 Positive
    else:
        raise ValueError("pos_label 必須是 0 或 1")

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1_score