import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.5):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)  # 最終分類層

    def forward(self, x):
        N, M, T, V, C = x.size()
        
        # 變換成 (N, T, C')，其中 C' = M * V * C
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, M, V, C)
        x = x.view(N, T, M * V * C)  # (N, T, C')

        lstm_out, _ = self.lstm(x)  # LSTM 計算
        embedding = lstm_out[:, -1, :]  # 取最後一個時間步的輸出作為 embedding
        out = self.fc(embedding)  # 全連接層進行分類
        return out, embedding  # 回傳分類結果 & embedding


# class PLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.5):
#         super(PLSTM, self).__init__()

#         self.torso_idx = [0, 1, 2, 3, 4]  # Spine, Neck, Head
#         self.left_arm_idx = [5, 7, 9]  # Left Shoulder, Elbow, Wrist, Hand
#         self.right_arm_idx = [6, 8, 10]  # Right Shoulder, Elbow, Wrist, Hand
#         self.left_leg_idx = [11, 13, 15]  # Left Hip, Knee, Ankle, Foot
#         self.right_leg_idx = [12, 14, 16]  # Right Hip, Knee, Ankle, Foot


#         # 每個部位獨立的 LSTM
#         self.lstm_torso = nn.LSTM(int((input_dim/17)*len(self.torso_idx)), hidden_dim, num_layers, batch_first=True, dropout=dropout)
#         self.lstm_left_arm = nn.LSTM(int((input_dim/17)*len(self.left_arm_idx)), hidden_dim, num_layers, batch_first=True, dropout=dropout)
#         self.lstm_right_arm = nn.LSTM(int((input_dim/17)*len(self.right_arm_idx)), hidden_dim, num_layers, batch_first=True, dropout=dropout)
#         self.lstm_left_leg = nn.LSTM(int((input_dim/17)*len(self.left_leg_idx)), hidden_dim, num_layers, batch_first=True, dropout=dropout)
#         self.lstm_right_leg = nn.LSTM(int((input_dim/17)*len(self.right_leg_idx)), hidden_dim, num_layers, batch_first=True, dropout=dropout)

#         # 共享的全連接層 (FC)
#         self.fc = nn.Linear(hidden_dim * 5, num_classes)  # 拼接 5 個部位的輸出

#     def forward(self, x):
#         """
#         x: (batch, num_person, time_steps, num_joints, coord) → (N, M, T, V, C)
#         """
#         N, M, T, V, C = x.shape  # 批次數, 人數, 時間步, 關節數, 坐標數

#         # **重新排列維度，合併 M 維度**
#         x = x.view(N * M, T, V, C)  # 變成 (N*M, T, V, C)

#         # **取出各個部位的 joints，並 reshape**
#         torso = x[:, :, self.torso_idx, :].reshape(N * M, T, -1)  # (N*M, T, 4*3)
#         left_arm = x[:, :, self.left_arm_idx, :].reshape(N * M, T, -1)  # (N*M, T, 4*3)
#         right_arm = x[:, :, self.right_arm_idx, :].reshape(N * M, T, -1)  # (N*M, T, 4*3)
#         left_leg = x[:, :, self.left_leg_idx, :].reshape(N * M, T, -1)  # (N*M, T, 4*3)
#         right_leg = x[:, :, self.right_leg_idx, :].reshape(N * M, T, -1)  # (N*M, T, 4*3)

#         # **將每個部位輸入到各自的 LSTM**
#         _, (ht_torso, _) = self.lstm_torso(torso)  
#         _, (ht_left_arm, _) = self.lstm_left_arm(left_arm)
#         _, (ht_right_arm, _) = self.lstm_right_arm(right_arm)
#         _, (ht_left_leg, _) = self.lstm_left_leg(left_leg)
#         _, (ht_right_leg, _) = self.lstm_right_leg(right_leg)

#         # **取最後一層 LSTM 的 hidden state**
#         torso_out = ht_torso[-1]  # (N*M, hidden_dim)
#         left_arm_out = ht_left_arm[-1]  # (N*M, hidden_dim)
#         right_arm_out = ht_right_arm[-1]  # (N*M, hidden_dim)
#         left_leg_out = ht_left_leg[-1]  # (N*M, hidden_dim)
#         right_leg_out = ht_right_leg[-1]  # (N*M, hidden_dim)

#         # **拼接所有部位的輸出**
#         final_embedding = torch.cat([torso_out, left_arm_out, right_arm_out, left_leg_out, right_leg_out], dim=-1)  # (N*M, hidden_dim*5)

#         # **分類**
#         out = self.fc(final_embedding)  # (N*M, num_classes)

#         # **最後 reshape 回 (N, M)**
#         out = out.view(N, M, -1).mean(dim=1)  # (N, num_classes)
#         final_embedding = final_embedding.view(N, M, -1).mean(dim=1)  # (N, hidden_dim*5)

#         return out, final_embedding  # 回傳分類結果 & embedding

import torch
import torch.nn as nn
import torch.nn.functional as F

class PartAwareLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(PartAwareLSTMCell, self).__init__()
        assert hidden_size % 5 == 0
        self.hidden_size = hidden_size // 5  # 17
        self.input_size = input_size
        self.num_keypoints = 17
        self.coord_dim = input_size // self.num_keypoints  # 動態計算：第一層 2，後續層 5

        self.divide_config = {
            'head': [0, 1, 2, 3, 4],
            'r_arm': [6, 8, 10],
            'l_arm': [5, 7, 9],
            'r_leg': [12, 14, 16],
            'l_leg': [11, 13, 15],
        }

        self.part_linears = nn.ModuleList([
            nn.Linear(len(joints) * self.coord_dim + hidden_size, 3 * self.hidden_size)
            for joints in self.divide_config.values()
        ])

        body_feat_dim = sum(len(joints) for joints in self.divide_config.values()) * self.coord_dim
        self.output_gate = nn.Linear(body_feat_dim + hidden_size, 5 * self.hidden_size)

    def forward(self, x, state):
        B = x.size(0)
        assert x.size(1) == self.input_size, f"Expected input size {self.input_size}, got {x.size(1)}"
        h_prev, c_prev = state
        c_parts = torch.chunk(c_prev, 5, dim=1)
        h = h_prev

        x = x.view(B, self.num_keypoints, -1)  # (B, 17, coord_dim)
        body_parts = []
        for joints in self.divide_config.values():
            part = torch.cat([x[:, j, :] for j in joints], dim=1)
            body_parts.append(part)

        out_gate_input = torch.cat(body_parts + [h], dim=1)
        o_all = torch.sigmoid(self.output_gate(out_gate_input))

        new_c_parts = []
        for idx, part_input in enumerate(body_parts):
            lstm_input = torch.cat([part_input, h], dim=1)
            gates = self.part_linears[idx](lstm_input)
            i, f, g = gates.chunk(3, dim=1)
            i, f, g = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g)
            c_new = f * c_parts[idx] + i * g
            new_c_parts.append(c_new)

        new_c = torch.cat(new_c_parts, dim=1)
        new_h = o_all * torch.tanh(new_c)
        return new_h, (new_h, new_c)

class PLSTM(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=85, num_layers=3, num_classes=40):
        super(PLSTM, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim

        self.cells = nn.ModuleList([
            PartAwareLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        self.fc1 = nn.Linear(hidden_dim, 75)
        self.fc2 = nn.Linear(75, num_classes)

    def forward(self, x):
        # x: (B, M, T, V, C)
        B, M, T, V, C = x.shape
        assert M == 1, f"Expected M=1, got {M}"
        assert V == 17 and C == 2, f"Expected (B, 1, T, 17, 2), got {x.shape}"
        
        # 去掉 M 維度並展平
        x = x.squeeze(1)  # (128, 1, 64, 17, 2) -> (128, 64, 17, 2)
        x = x.reshape(B, T, -1)  # (128, 64, 34)
        assert x.size(2) == self.input_dim, f"Expected input dim {self.input_dim}, got {x.size(2)}"

        h = torch.zeros(B, self.hidden_size, device=x.device)
        c = torch.zeros(B, self.hidden_size, device=x.device)

        for t in range(T):
            input_t = x[:, t, :]  # (B, 34)
            for i, cell in enumerate(self.cells):
                h, (h, c) = cell(input_t, (h, c))
                input_t = h

        x = F.relu(self.fc1(h))
        logits = self.fc2(x)
        return logits, h