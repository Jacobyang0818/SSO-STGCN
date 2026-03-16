import copy as cp
import torch
import torch.nn as nn
# from mmcv.runner import load_checkpoint
from graph import Graph
# from pyskl.utils import Graph, cache_checkpoint
from gcns import mstcn, unit_gcn, unit_tcn, GCNHead
from torch.nn.init import xavier_uniform_
EPS = 1e-4

class STGCNBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 tkernel=3,
                 act=0,
                 **kwargs):
        super().__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn']

        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)
        if tkernel == 3:
            ms_cfg = [(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1']
        elif tkernel == 5:
            ms_cfg = [(5, 1), (5, 2), (5, 3), (5, 4), ('max', 5), '1x1']
        elif tkernel == 7:
            ms_cfg = [(7, 1), (7, 2), (7, 3), (7, 4), ('max', 7), '1x1']
        else:
            print('tcn_kernel_size_error')
        

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
            
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels, stride=stride, ms_cfg=ms_cfg, **tcn_kwargs)
        
        if act == 0:
            self.act = nn.ReLU()
        elif act == 1:
            self.act = nn.ReLU6()
        elif act == 2:
            self.act = nn.Hardshrink()
        elif act == 3:
            self.act = nn.SiLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.act(x)
    

class SSO_STGCN(nn.Module):

    def __init__(self,
                graph_cfg,
                in_channels=3,
                data_bn_type='VC',
                num_person=2,  # * Only used when data_bn_type == 'MVC'
                pretrained=None,
                args = None,
                **kwargs):
        
        if args is None:
            print('Build model fail cause args Error')

        torch.manual_seed(42)

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

        num_stages = args.num_in + args.num_main + args.num_init
        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = args.base_channel

        # self.ch_ratio = ch_ratio
        # self.inflate_stages = inflate_stages
        # self.down_stages = down_stages

        modules = []
        
        # layer 1
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, self.base_channels, A.clone(), 1, act=args.act, tkernel=3, residual=False, **lw_kwargs[0])]
        
        '''init block'''
        # layer 2~num_input
        for i in range(1, args.num_init):
            stride = args.stride_init if i == 1 else 1  
            in_channels = self.base_channels # input channel = 上一層的output channel
            modules.append(STGCNBlock(self.base_channels, self.base_channels, A.clone(), stride, act=args.act, tkernel=args.tkernel_init, **lw_kwargs[i - 1]))
        
        '''input block'''
        # layer 2~num_input
        for i in range(0, args.num_in):
            stride = args.stride_in if i == 0 else 1  
            in_channels = self.base_channels # input channel = 上一層的output channel
            
            out_channels = args.oc_in
            self.base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, act=args.act, tkernel=args.tkernel_in, **lw_kwargs[i - 1]))
        
        '''main block''' 
        # layer num_input~num_input+num_main
        for i in range(0, args.num_main):
            stride = args.stride_main if i == 0 else 1
            in_channels = self.base_channels # input channel = 上一層的output channel

            out_channels = args.oc_main
            self.base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, act=args.act, tkernel=args.tkernel_main, **lw_kwargs[i - 1]))

        # if self.in_channels == self.base_channels:
        #     num_stages -= 1

        # self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.num_stages = len(self.gcn)
        self.pretrained = pretrained

        self.head = GCNHead(
            num_classes=args.cls,
            in_channels=self.base_channels,
            dropout=args.dropout_fc,
        )

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
        x, embedding = self.head(x)

        return x, embedding