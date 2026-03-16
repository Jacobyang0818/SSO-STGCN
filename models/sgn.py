import math
import torch
from torch import nn
from gcns import mstcn, unit_gcn, unit_tcn, GCNHead

class unit_sgn(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        # x: N, C, T, V; A: N, T, V, V
        x1 = x.permute(0, 2, 3, 1).contiguous()
        x1 = A.matmul(x1).permute(0, 3, 1, 2).contiguous()
        return self.relu(self.bn(self.conv(x1) + self.residual(x)))
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnitSGN(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        # x: (N, C, T, V), A: (N, T, V, V)
        x1 = x.permute(0, 2, 3, 1).contiguous()  # (N, T, V, C)
        x1 = A.matmul(x1).permute(0, 3, 1, 2).contiguous()  # (N, C, T, V)
        return self.relu(self.bn(self.conv(x1) + self.residual(x)))


class SGN(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_joints=25, T=64, bias=True, num_classes=4):
        super(SGN, self).__init__()

        self.T = T
        self.num_joints = num_joints
        self.base_channel = base_channels

        self.joint_bn = nn.BatchNorm1d(in_channels * num_joints)
        self.motion_bn = nn.BatchNorm1d(in_channels * num_joints)

        self.t_embed = self.embed_mlp(self.T, base_channels * 4, base_channels, bias=bias)
        self.s_embed = self.embed_mlp(self.num_joints, base_channels, base_channels, bias=bias)
        self.joint_embed = self.embed_mlp(in_channels, base_channels, base_channels, bias=bias)
        self.motion_embed = self.embed_mlp(in_channels, base_channels, base_channels, bias=bias)

        self.compute_A1 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=1, bias=bias)
        self.compute_A2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=1, bias=bias)

        self.tcn = nn.Sequential(
            nn.AdaptiveMaxPool2d((20, 1)),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=(3, 1), padding=(1, 0), bias=bias),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=1, bias=bias),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )

        self.gcn1 = UnitSGN(base_channels * 2, base_channels * 2, bias=bias)
        self.gcn2 = UnitSGN(base_channels * 2, base_channels * 4, bias=bias)
        self.gcn3 = UnitSGN(base_channels * 4, base_channels * 4, bias=bias)
        
        self.head = GCNHead(
            num_classes=num_classes,
            in_channels=512,
            dropout=0,
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.conv.weight, 0)
        nn.init.constant_(self.gcn2.conv.weight, 0)
        nn.init.constant_(self.gcn3.conv.weight, 0)

    def embed_mlp(self, in_channels, out_channels, mid_channels=64, bias=False):
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def compute_A(self, x):
        # X: (N, C, T, V)
        A1 = self.compute_A1(x).permute(0, 2, 3, 1).contiguous()  # (N, T, V, C)
        A2 = self.compute_A2(x).permute(0, 2, 1, 3).contiguous()  # (N, T, C, V)
        A = torch.matmul(A1, A2)  # (N, T, V, V)
        return F.softmax(A, dim=-1)

    def forward(self, joint):
        N, M, T, V, C = joint.shape

        joint = joint.reshape(N * M, T, V, C).permute(0, 3, 2, 1).contiguous()
        motion = F.pad(torch.diff(joint, dim=3), (0, 1), mode="constant", value=0)

        joint = self.joint_bn(joint.view(N * M, C * V, T))
        motion = self.motion_bn(motion.view(N * M, C * V, T))
        joint = joint.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()
        motion = motion.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()

        joint_embed = self.joint_embed(joint)
        motion_embed = self.motion_embed(motion)

        t_code = torch.eye(T, device=joint.device)[None, :, None].repeat(N * M, 1, V, 1)
        s_code = torch.eye(V, device=joint.device)[None, :, :, None].repeat(N * M, 1, 1, T)
        t_embed = self.t_embed(t_code).permute(0, 1, 3, 2).contiguous()
        s_embed = self.s_embed(s_code).permute(0, 1, 3, 2).contiguous()

        x = torch.cat([joint_embed + motion_embed, s_embed], 1)
        A = self.compute_A(x)

        for gcn in [self.gcn1, self.gcn2, self.gcn3]:
            x = gcn(x, A)

        x = x + t_embed
        x = self.tcn(x)

        # print(x.size())
        x = x.reshape((N, M) + x.shape[1:])

        x, embedding = self.head(x)

        return x, embedding

    
