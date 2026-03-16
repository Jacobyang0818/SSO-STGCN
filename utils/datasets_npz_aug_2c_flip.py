# from pyskl.datasets import build_dataloader, build_dataset
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torch
from sklearn.model_selection import train_test_split
import os

def get_data_split_type(file_name):
    base_name = os.path.basename(file_name)  # 取得檔案名稱（不含路徑）
    if "train" in base_name:
        return "train"
    elif "val" in base_name:
        return "val"
    elif "test" in base_name:
        return "test"
    else:
        raise ValueError("檔案名稱無法判斷資料類型，請確認是否包含 'train', 'val' 或 'test' 關鍵字。")
    
class KeypointDataset(Dataset):
    def __init__(self, keypoints, labels, names, pose='coco', feature='j', fixed_frames=64, phase='train', flip=True):
        """
        初始化数据集，固定时间维度为 fixed_frames，并对 label=0 的样本进行数据增广
        :param keypoints: 原始关键点数据，形状为 (N, M, T, V, C)
        :param labels: 标签数据
        :param names: 样本名称列表
        :param pose: 骨架类型 ('nturgb+d', 'coco', 'openpose', 等)
        :param feature: 特征类型 ('j' 表示关节，'b' 表示骨骼，'m' 表示运动，或组合如 'bm')
        :param fixed_frames: 固定的时间帧数，默认为 64
        """
        self.pose = pose
        self.feature = feature
        self.fixed_frames = fixed_frames
        self.bone_pairs = self.get_bone_pairs()

        self.kps_flip_swap_pairs = {
            'coco': [ (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16) ],
            'blaze': [ (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16) ],
            'yolo': [ (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16) ],
        }

        # 用于存储最终处理后的数据
        processed_keypoints = []
        processed_labels = []
        processed_names = []
        
        for i in range(len(keypoints)):
            kp = keypoints[i]  # 形状 (M, T, V, C)
            label = labels[i]
            name = names[i]
            orig_T = kp.shape[1]  # 原始时间帧数

            # 如果 label == 0，执行数据增广
            if label == 0:
                augment_windows = [(0, 63), (13, 76), (26, 89)]  # 定义增广窗口   0~89   <0~63,  13~76, 26~89>
                for start, end in augment_windows:
                    if end + 1 <= orig_T:  # 确保窗口不会超出原始数据长度
                        new_kp = kp[:, start:end + 1]  # 直接裁剪片段，形状应为 (M, 64, V, C)
                        new_kp = self._adjust_frames(new_kp, orig_T=64)  # 确保调整后的时间长度仍然是 64
                        processed_keypoints.append(new_kp)
                        processed_labels.append(label)
                        processed_names.append(f"{name}_aug_{start}_{end}")

            else:
                # 原始样本（不论是否增广，原样保留）
                new_kp = self._adjust_frames(kp, orig_T)
                processed_keypoints.append(new_kp)
                processed_labels.append(label)
                processed_names.append(name)

        # 转换为 numpy 数组
        self.keypoints = np.stack(processed_keypoints, axis=0)
        self.labels = np.array(processed_labels)
        self.names = np.array(processed_names)
        
        # Flip aug mentation
        if flip is True:
            flipped_keypoints = []
            flipped_labels = []
            flipped_names = []

            for i in range(len(self.keypoints)):
                kp = self.keypoints[i]  # (M, T, V, C)
                flipped_kp = kp.copy()

                # 水平翻轉 x 座標
                flipped_kp[..., 0] = 1 - flipped_kp[..., 0]  # 假設 x 已正規化 (0~1)

                # 交換對應的關鍵點
                if self.pose in self.kps_flip_swap_pairs:
                    swap_pairs = self.kps_flip_swap_pairs[self.pose]
                    for idx1, idx2 in swap_pairs:
                        flipped_kp[..., [idx1, idx2], :] = flipped_kp[..., [idx2, idx1], :]  # 交換索引

            
                flipped_keypoints.append(flipped_kp)
                flipped_labels.append(self.labels[i])  # 標籤保持不變
                flipped_names.append(f"{self.names[i]}_flip")

            # 合併原始數據與翻轉數據
            self.keypoints = np.concatenate([self.keypoints, np.stack(flipped_keypoints, axis=0)], axis=0)
            self.labels = np.concatenate([self.labels, np.array(flipped_labels)], axis=0)
            self.names = np.concatenate([self.names, np.array(flipped_names)], axis=0)

        # 处理特征
        if 'b' in feature:
            self.keypoints = self.calculate_bone(self.keypoints)
        if 'm' in feature:
            self.keypoints = self.calculate_motion(self.keypoints)

        self.keypoints = self.keypoints[:, : , :, :, :2]

    def _adjust_frames(self, keypoints, orig_T):
        """调整关键点数据的时间维度到 fixed_frames"""
        M, T, V, C = keypoints.shape
        target_T = self.fixed_frames

        if orig_T >= target_T:
            # 如果原始帧数大于等于目标帧数，裁剪到前 target_T 帧
            return keypoints[:, :target_T, :, :]
        else:
            # 如果原始帧数小于目标帧数，用零填充
            pad_width = target_T - orig_T
            padding = [(0, 0), (0, pad_width), (0, 0), (0, 0)]  # 只在 T 维度填充
            return np.pad(keypoints, padding, mode='constant', constant_values=0)

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        return {
            'keypoint': torch.tensor(self.keypoints[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'name': self.names[idx]
        }

    def get_bone_pairs(self):
        """根据 pose 类型返回骨骼连接对（不变）"""
        if self.pose == 'nturgb+d':
            return ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
                    (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
                    (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11))
        elif self.pose == 'coco' or self.pose == 'blaze' or self.pose == 'yolo':
            return ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8),
                    (11, 0), (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))
        elif self.pose == 'openpose':
            return ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
                    (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15))
        else:
            raise ValueError(f"Unsupported pose type: {self.pose}")

    def calculate_bone(self, keypoints):
        """計算骨骼特徵（不變）"""
        N, M, T, V, C = keypoints.shape
        bone = np.zeros_like(keypoints, dtype=np.float32)

        for v1, v2 in self.bone_pairs:
            bone[..., v1, :] = keypoints[..., v1, :] - keypoints[..., v2, :]
            if C >= 3 and self.pose in ['openpose', 'coco', 'blaze', 'yolo']:
                score = (keypoints[..., v1, 2] + keypoints[..., v2, 2]) / 2
                bone[..., v1, 2] = score
            if C == 4 and self.pose == 'blaze':
                score = (keypoints[..., v1, 3] + keypoints[..., v2, 3]) / 2
                bone[..., v1, 3] = score
        return bone

    def calculate_motion(self, keypoints):
        """計算運動特徵（不變）"""
        N, M, T, V, C = keypoints.shape
        motion = np.zeros_like(keypoints, dtype=np.float32)
        diff = np.diff(keypoints, axis=2)
        motion[:, :, :T-1] = diff
        if C >= 3 and self.pose in ['openpose', 'coco', 'blaze', 'yolo']:
            score = (keypoints[:, :, 1:, :, 2] + keypoints[:, :, :T - 1, :, 2]) / 2
            motion[:, :, :T - 1, :, 2] = score
        if C == 4 and self.pose == 'blaze':
            score = (keypoints[:, :, 1:, :, 3] + keypoints[:, :, :T - 1, :, 3]) / 2
            motion[:, :, :T - 1, :, 3] = score
        return motion

# 更新 load_npz_dataset 函數以傳遞 fixed_frames
def load_npz_dataset(npz_file, workers_per_gpu=2, batch_size=16, pose='coco', feature='j', fixed_frames=64):
    data = np.load(npz_file, allow_pickle=True)
    keypoints = data['keypoints']  # NMTVC
    labels = data['labels']
    names = data['names']
    data_split = get_data_split_type(npz_file)
    torch.manual_seed(42)
    dataset = KeypointDataset(keypoints, labels, names, pose=pose, feature=feature, fixed_frames=fixed_frames, phase=data_split, flip=True)
    drop_last = (data_split == "train")
    shuffle = (data_split == "train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=workers_per_gpu, persistent_workers=True, pin_memory=True, drop_last=drop_last)   # 不能drop last 因為有類別不平衡
    return loader