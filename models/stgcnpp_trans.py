import copy as cp
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from pyskl.utils import Graph, cache_checkpoint
from pyskl.models.gcns.utils import mstcn, unit_gcn, unit_tcn
from pyskl.models.heads import GCNHead
from torch.nn.init import xavier_uniform_
EPS = 1e-4

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算位置编码
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: (seq_len, batch_size, embedding_size)
        x = x + self.pe[:x.size(0)]
        return x

def build_trans(out_channels=32, **tcn_kwargs):
    num_heads = tcn_kwargs.get('num_heads', 8)  # 设置默认值，例如8
    dropout = tcn_kwargs.get('dropout', 0.1)    # 设置默认值，例如0.
    # dim_feedforward = tcn_kwargs.get('dim_feedforward', 2048)
    encoder_layer = nn.TransformerEncoderLayer(d_model=out_channels,
                                               nhead=num_heads,
                                               dim_feedforward=out_channels,
                                               dropout=dropout)
    return nn.TransformerEncoder(encoder_layer, num_layers=1)


class STGCNBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 **kwargs):
        super().__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'
        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn', 'transformer']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn']


        self.tcn_type = tcn_type
        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        elif tcn_type == 'transformer':
            self.pos_encoder = PositionalEncoding(out_channels)
            self.tcn = build_trans(out_channels=out_channels, **tcn_kwargs)
            
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        elif tcn_type in ['transformer']:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
    
    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.gcn(x, A)

        if self.tcn_type == 'transformer':
            # 获取形状信息
            batch_size, channels, seq_len, num_nodes = x.size()
            
            # 调整形状以适配 Transformer
            x = x.permute(2, 0, 3, 1).contiguous()  # (seq_len, batch_size, num_nodes, channels)
            x = x.view(seq_len, batch_size * num_nodes, channels)  # (seq_len, batch_size*num_nodes, channels)
            
            # 添加位置编码
            x = self.pos_encoder(x)
            
            # 通过 Transformer
            x = self.tcn(x)  # (seq_len, batch_size*num_nodes, channels)
            
            # 调整回原始形状
            x = x.view(seq_len, batch_size, num_nodes, channels)
            x = x.permute(1, 3, 0, 2).contiguous()  # (batch_size, channels, seq_len, num_nodes)
        else:
            x = self.tcn(x)
            x = x + res
        # x = x + res
        return self.relu(x)
    
class STGCN(nn.Module):
    def __init__(self,
                graph_cfg,
                in_channels=3,
                base_channels=64,
                data_bn_type='VC',
                ch_ratio=2, # 每次放大兩倍通道數
                num_person=2,  # * Only used when data_bn_type == 'MVC'
                num_stages=10, # 有幾層
                inflate_stages=[5, 8],  # 通道數在這兩層增加
                down_stages=[5, 8], # 時間維度下採樣，壓縮時間維度
                pretrained=None,
                num_classes=4,
                head_dropout=0,
                **kwargs):
        
        super().__init__()
        # 讀圖
        self.graph = Graph(**graph_cfg)

        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)  # 透過 stride 控制下採樣
            in_channels = base_channels # input channel = 上一層的output channel
            if i in inflate_stages:  # 確定當前在需要放大通道數的層數
                inflate_times += 1   # 5 -> inlate_times = 1;    8->inflate_times = 2
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained
        
        self.head = GCNHead(
            num_classes=num_classes,
            in_channels=base_channels,
            dropout=head_dropout
        )
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        if (len(x.size()) == 6):
            x = x.squeeze(1)
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N, M) + x.shape[1:])
        
        embedding = x 
        x = self.head(x)

        return x, embedding
