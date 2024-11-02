from pyskl.datasets import build_dataloader, build_dataset
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from sklearn.model_selection import train_test_split

# def generate_subsets(names, train_ratio, valid_ratio, test_ratio):
#     unique_names = np.unique(names)
#     train_names, temp_names = train_test_split(unique_names, test_size=1-train_ratio, random_state=42)
#     valid_names, test_names = train_test_split(temp_names, test_size=(test_ratio/(valid_ratio+test_ratio)), random_state=42)
    

def load_npz_dataset_random_split(npz_file,  workers_per_gpu=2, batch_size=16):

    data = np.load(npz_file, allow_pickle=True)
    # 分三類 
    keypoints = data['keypoints']
    labels = data['labels']
    names = data['names']
    indices_suffix = np.array([int(name[-2:]) for name in names])

    # 條件篩選：label == 1 且尾碼在 1 到 5 之間
    low = (labels == 1) & ((indices_suffix >= 1) & (indices_suffix <= 5))

    # 條件篩選：label == 2 且尾碼在 6 到 10 之間
    medium = (labels == 2) & ((indices_suffix >= 6) & (indices_suffix <= 10))

    # 合併條件篩選出的索引
    final_indices = np.where(low | medium)[0]

    # 找到 label == 0 和 label == 3 的索引，這些索引保持不變
    indices_label_0_3 = np.where((labels == 0) | (labels == 3))[0]

    # 將所有要保留的索引合併（包含 label == 0、3）
    all_final_indices = np.concatenate((final_indices, indices_label_0_3))

    # 根據合併後的索引篩選 keypoints、labels 和 names
    filtered_keypoints = keypoints[all_final_indices]
    filtered_labels = labels[all_final_indices]
    filtered_names = names[all_final_indices]
    
    # 取代所有的
    filtered_labels = np.where((filtered_labels == 1) | (filtered_labels == 2), 1, filtered_labels)

    keypoints_train, keypoints_temp, labels_train, labels_temp, names_train, names_temp = train_test_split(
        filtered_keypoints, filtered_labels, filtered_names, test_size=0.4, stratify=filtered_labels, random_state=42
    )

    keypoints_valid, keypoints_test, labels_valid, labels_test, names_valid, names_test = train_test_split(
        keypoints_temp, labels_temp, names_temp, test_size=0.5, stratify=labels_temp, random_state=42
    )

    train_dataset = KeypointDataset(keypoints_train, labels_train, names_train)
    valid_dataset = KeypointDataset(keypoints_valid, labels_valid, names_valid)
    test_dataset = KeypointDataset(keypoints_test, labels_test, names_test)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers_per_gpu)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=workers_per_gpu)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers_per_gpu)

    return train_loader, valid_loader, test_loader

def load_dataset(csv_file, feats='j', batch_size=16, frames=200, pose_extractor='yolo', mode = 'full', workers_per_gpu=2, split=1):
    data = pd.read_csv(csv_file)
    cleaned_data = data.dropna()  # 去除空行
    dates = cleaned_data['date'].tolist()  # 將 'date' 列轉換為整數列表
    sagittal_lbs = cleaned_data['sagittal_lb'].tolist()
    sagittal_ubs = cleaned_data['sagittal_ub'].tolist()
    coronal_lbs = cleaned_data['coronal_lb'].tolist()
    coronal_ubs = cleaned_data['coronal_lb.1'].tolist()
    combined_train_dataset = None
    combined_val_dataset = None
    combined_test_dataset = None
    for idx, date in enumerate(dates):
        date = str(date)
        c_l = coronal_lbs[idx] # coronal_lb
        c_u = coronal_ubs[idx] # coronal_ub
        s_l = sagittal_lbs[idx] # sagittal_lb
        s_u = sagittal_ubs[idx] # sagittal_ub
        bound = [c_l, c_u, s_l, s_u]
        coronal_ann, sagittal_ann = None, None
        if mode in ['full', 'coronal']:
            coronal_ann = f'tools/data/drunk/{date}/splits_coronal/{date}_coronal_split_{split}_{pose_extractor}.pkl'
        if mode in ['full', 'sagittal']:
            sagittal_ann = f'tools/data/drunk/{date}/splits_sagittal/{date}_sagittal_split_{split}_{pose_extractor}.pkl'
        # 为每个日期构建数据集 (train, val, test)
        train_dataset, val_dataset, test_dataset = build_date_pose_datasets(coronal_ann, sagittal_ann, mode=mode, bound=bound, videos_per_gpu=batch_size, workers_per_gpu=workers_per_gpu, clip_len=frames, feats=feats, pose_extractor=pose_extractor)

        # 使用 dataset + dataset 的方式合并数据集
        if combined_train_dataset is None:
            combined_train_dataset = train_dataset
            combined_val_dataset = val_dataset
            combined_test_dataset = test_dataset
        else:
            combined_train_dataset += train_dataset
            combined_val_dataset += val_dataset
            combined_test_dataset += test_dataset
    
    dataloader_setting = dict(
        videos_per_gpu=batch_size,
        workers_per_gpu=workers_per_gpu,
        persistent_workers=True,
        seed=42,
        pin_memory=True)
    dataloader_setting = dict(dataloader_setting)
    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in [combined_train_dataset, combined_val_dataset, combined_test_dataset]
    ]
    training_loader = data_loaders[0]
    validation_loader = data_loaders[1]
    test_loader = data_loaders[2]
    return training_loader, validation_loader, test_loader


def build_date_pose_datasets(coronal_ann, sagittal_ann, mode='full', bound=[0.0, 1.0, 0.0, 1.0],videos_per_gpu=4, workers_per_gpu=2, clip_len = 200, feats ='j', pose_extractor='yolo'):
    dataset_type = 'PoseDataset'
    # 定義管道
    if pose_extractor in ['yolo', 'hrnet']:
        dataset = 'coco'
        CustomNormalize = 'CustomNormalize2D'
    else:
        dataset = 'blaze'
        CustomNormalize = 'Blaze3D'
        
    coronal_pipeline = [
        dict(type=CustomNormalize, img_shape=(2160, 3840), lb=bound[0], ub=bound[1], threshold=0.01, mode='coronal'), 
        dict(type='GenSkeFeat', dataset=dataset, feats=[feats]),
        dict(type='SampleFrames', clip_len=clip_len),
        dict(type='PoseDecode'),
        dict(type='FormatGCNInput', num_person=1),
        dict(type='Collect', keys=['keypoint', 'label','total_frames'], meta_keys=[]),
        dict(type='ToTensor', keys=['keypoint'])
    ]
    
    sagittal_pipeline = [
        dict(type=CustomNormalize, img_shape=(2160, 3840), lb=bound[2], ub=bound[3], threshold=0.01, mode='sagittal'), 
        dict(type='GenSkeFeat', dataset=dataset, feats=[feats]),
        dict(type='SampleFrames', clip_len=clip_len),
        dict(type='PoseDecode'),
        dict(type='FormatGCNInput', num_person=1),
        dict(type='Collect', keys=['keypoint', 'label','total_frames'], meta_keys=[]),
        dict(type='ToTensor', keys=['keypoint'])
    ]
    
    # Coronal數據集
    coronal_data = dict(
        videos_per_gpu=videos_per_gpu,
        workers_per_gpu=workers_per_gpu,
        train=dict(type=dataset_type, ann_file=coronal_ann, pipeline=coronal_pipeline, split='xsub_train'),
        val=dict(type=dataset_type, ann_file=coronal_ann, pipeline=coronal_pipeline, split='xsub_val'),
        test=dict(type=dataset_type, ann_file=coronal_ann, pipeline=coronal_pipeline, split='xsub_test')
    )
    
    # Sagittal數據集
    sagittal_data = dict(
        videos_per_gpu=videos_per_gpu,
        workers_per_gpu=workers_per_gpu,
        train=dict(type=dataset_type, ann_file=sagittal_ann, pipeline=sagittal_pipeline, split='xsub_train'),
        val=dict(type=dataset_type, ann_file=sagittal_ann, pipeline=sagittal_pipeline, split='xsub_val'),
        test=dict(type=dataset_type, ann_file=sagittal_ann, pipeline=sagittal_pipeline, split='xsub_test')
    )
    
    # 構建數據集
    coronal_datasets, sagittal_datasets, full_datasets = None, None, None
    if mode in ['full','coronal']:
        coronal_datasets = [
            build_dataset(coronal_data['train']), 
            build_dataset(coronal_data['val']), 
            build_dataset(coronal_data['test'])
        ]
        if mode != 'full':
            return coronal_datasets
    if mode in ['full','sagittal']:
        sagittal_datasets = [
            build_dataset(sagittal_data['train']), 
            build_dataset(sagittal_data['val']), 
            build_dataset(sagittal_data['test'])
        ]
        if mode != 'full':
            return sagittal_datasets
        
    if mode == 'full' :
        full_datasets = [coronal_datasets[i] + sagittal_datasets[i] for i in range(len(coronal_datasets))]
        return full_datasets
    

class KeypointDataset(Dataset):
    def __init__(self, keypoints, labels, names):
        self.keypoints = keypoints
        self.labels = labels
        self.names = names

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        return {
            'keypoints': torch.tensor(self.keypoints[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'name': self.names[idx]
        }