"""
trainer 模組
============
包含訓練與推論的核心邏輯：

    - trainer.train  → train() 函數（主訓練流程）
    - trainer.predict → 推論腳本

使用範例：
    from trainer.train import train, train_with_timeout
"""

from .train import train, train_with_timeout

__all__ = ["train", "train_with_timeout"]
