import torch
import torch.nn as nn
import torch.nn.functional as F


class SPNet(nn.Module):
    def __init__(self, global_feat=False):
        super(SPNet, self).__init__()
        self.global_feat = global_feat

        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        # [B, N, 3] -> [B, 3, N]
        x = x.transpose(2, 1)

        x = self.mlp1(x)  # [B, 64, N]
        x = self.mlp2(x)  # [B, 128, N]
        x = self.mlp3(x)  # [B, 128, N]

        # [B, 256, N] -> [B, 256]
        x_global = torch.max(x, 2)[0]

        if self.global_feat:
            return x_global  
        else:
            x_global_expanded = x_global.unsqueeze(2).repeat(1, 1, x.shape[2])  # [B, 128, N]
            x_cat = torch.cat([x, x_global_expanded], dim=1)  # [B, 128+128, N]
            x_out = x_cat.transpose(2, 1)
            return x_out  # [B, N, 256]

class PTI(nn.Module):
    def __init__(self, input_dim=256, out_channels=96, grid_size=128):
        super().__init__()
        self.grid_size = grid_size
        self.reduce = nn.Linear(input_dim, out_channels)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: [B, N, C_in]  e.g., [N, 4096, 256]
        returns: [B, out_channels, H, W]
        """
        B, N, _ = feats.shape
        device = feats.device
        H = W = self.grid_size

        idx = torch.arange(N, device=device)
        y = (idx // W).repeat(B, 1)  # [B, N]
        x = (idx % W).repeat(B, 1)   # [B, N]

        feats = self.reduce(feats)  # [B, N, out_channels]

        img = torch.zeros((B, feats.shape[2], H, W), dtype=feats.dtype,device=device)

        for b in range(B):
            img[b, :, y[b], x[b]] = feats[b].T 

        return img  # [B, C, H, W]



if __name__ == '__main__':
    data = torch.randn(1, 4096, 3)
    model = SPNet(global_feat=False)
    out1 = model(data)
    pti = PTI(input_dim=256, out_channels=96, grid_size=128)
    out2 = pti(out1)
    print(out1.shape)
    print(out2.shape)