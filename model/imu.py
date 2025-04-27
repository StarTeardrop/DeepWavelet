import torch
import torch.nn as nn


class BiasGRUPredictor(nn.Module):
    def __init__(self, imu_input_dim=10, rnn_hidden=128, img_feat_dim=512):
        super().__init__()
        self.rnn = nn.GRU(input_size=imu_input_dim, hidden_size=rnn_hidden, batch_first=True, bidirectional=True)

        self.img_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 96, 1, 1]
            nn.Flatten(),                  # [B, 96]
            nn.Linear(96, img_feat_dim),  
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.regressor = nn.Sequential(
            nn.Linear(rnn_hidden * 2 + img_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6)  # [ba(3), bg(3)]
        )

    def forward(self, imu_seq, img_feat):
        # imu_seq: [B, N, 10], img_feat: [B, 96, 128, 128]
        B = imu_seq.size(0)
        rnn_out, _ = self.rnn(imu_seq)           # [B, N, 2*rnn_hidden]
        rnn_feat = rnn_out[:, -1, :]             # [B, 2*rnn_hidden]

        img_feat = self.img_pool(img_feat)       # [B, img_feat_dim]
        fused_feat = torch.cat([rnn_feat, img_feat], dim=1)  # [B, 2*rnn_hidden + img_feat_dim]

        bias = self.regressor(fused_feat)        # [B, 6]
        ba, bg = bias[:, :3], bias[:, 3:]
        return ba, bg


class GRUNet(nn.Module):
    def __init__(self, input_dim=10, rnn_hidden=256, out_channels=96, grid_size=128):
        super().__init__()
        self.bias_predictor = BiasGRUPredictor(imu_input_dim=input_dim, rnn_hidden=128, img_feat_dim=256)
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=rnn_hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(rnn_hidden * 2, out_channels * (grid_size * grid_size))
        self.out_channels = out_channels
        self.grid_size = grid_size

    def preprocess_sequence(self, x, target_len=128):
        B, N, C = x.shape
        if N < target_len:
            pad_len = target_len - N
            padding = torch.zeros(B, pad_len, C, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        elif N > target_len:
            indices = torch.linspace(0, N - 1, target_len).long().to(x.device)  # [128]
            x = torch.stack([x[b][indices] for b in range(B)], dim=0)  # [B, 128, C]
        return x

    def forward(self, x, img_feat):
        B = x.shape[0]
        x = self.preprocess_sequence(x, target_len=128)  # [B, 128, 10]

        ba, bg = self.bias_predictor(x, img_feat)  # [B, 3], [B, 3]
        x = x.clone()
        x[:, :, 4:7] -= ba.unsqueeze(1)  
        x[:, :, 7:10] -= bg.unsqueeze(1)  

        feat_seq, _ = self.rnn(x)  # [B, 128, 2*hidden]
        feat_last = feat_seq[:, -1, :]  # [B, 512]


        feat_img = self.fc(feat_last)  # [B, 96*128*128]
        feat_img = feat_img.view(B, self.out_channels, self.grid_size, self.grid_size)

        # [B, 96, 128, 128]
        return feat_img, ba, bg


if __name__ == '__main__':
    data1 = torch.randn(1, 200, 10)
    img1_feat1 = torch.randn(1, 96, 128, 128)
    model = GRUNet()
    out = model(data1, img1_feat1)
