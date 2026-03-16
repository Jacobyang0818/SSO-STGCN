import copy as cp
import torch
import torch.nn as nn
# from mmcv.runner import load_checkpoint
from graph import Graph
# from pyskl.utils import Graph, cache_checkpoint
from gcns import mstcn, unit_gcn, unit_tcn, GCNHead
from torch.nn.init import xavier_uniform_
EPS = 1e-4

class MLPModel(nn.Module):
    def __init__(self, 
                 in_channels=2, 
                 num_classes=4, 
                 num_person=1, 
                 num_joints=17, 
                 time_steps=64, 
                 hidden_dim=256, 
                 dropout=0.5,
                 num_layers=2,
                 data_bn_type='VC'):
        super(MLPModel, self).__init__()

        self.in_channels = in_channels
        self.num_person = num_person
        self.num_joints = num_joints
        self.time_steps = time_steps
        self.data_bn_type = data_bn_type

        input_dim = in_channels * num_joints * time_steps

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_joints)
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        else:
            self.data_bn = nn.Identity()

        # 動態建立多層 MLP
        mlp_layers = []
        mlp_layers.append(nn.Linear(input_dim, hidden_dim))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*mlp_layers)

        self.head = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        print("MLP input shape:", x.shape)  # 應該是 [N, M, T, V, C]
        N, M, T, V, C = x.shape

        x = x.permute(0, 1, 3, 4, 2).contiguous()  # [N, M, V, C, T]
        
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))

        print("Flattened shape:", x.view(N * M, -1).shape)  # 應該是 [N*M, 2176]
        
        x = x.view(N * M, -1)  # Flatten to [N*M, in_features]

        feat = self.mlp(x)  # [N*M, hidden_dim]

        # embedding = x.view(N * M, -1)
        out = self.head(feat)

        return out, feat