import torch
import torch.nn as nn
import torch.nn.functional as F
from model.sf import SFNet
from model.ddaf import DDAF
from model.sp import SPNet, PTI
from model.utils import *
from model.imu import *


class SIFNet(nn.Module):
    def __init__(self, decode_channels=96, window_size=8, ):
        super().__init__()

        self.sf = SFNet(decode_channels=decode_channels, window_size=window_size)
        self.ddaf = DDAF(decode_channels, num_heads=8, LayerNorm_type='WithBias')
        self.spnet = SPNet(global_feat=False)
        self.pti = PTI(out_channels=decode_channels, grid_size=128)
        self.wf = WF(in_channels=decode_channels, decode_channels=decode_channels)
        self.gru = GRUNet()
        self.norm = nn.LayerNorm(decode_channels, eps=1e-6)  # final norm layer
        self.head = nn.Sequential(nn.Linear(decode_channels, 4),
                                    nn.Tanh()
                                    )

    def forward(self, x1, x2, p1, p2, imu):
        x1 = self.sf(x1)  # [N, 96, 128, 128]
        x2 = self.sf(x2)  # [N, 96, 128, 128]
        p1 = self.spnet(p1)  # [N, 4096, 256]
        p2 = self.spnet(p2)  # [N, 4096, 256]
        p1 = self.pti(p1)  # [N, 96, 128, 128]
        p2 = self.pti(p2)  # [N, 96, 128, 128]
        x1_p1_res = self.ddaf(x1, p1)  # [N, 96, 128, 128]
        x2_p2_res = self.ddaf(x2, p2)  # [N, 96, 128, 128]
        x_p_res = self.wf(x1_p1_res, x2_p2_res)  # [N, 96, 128, 128]
        imu_res, ba, bg = self.gru(imu, x_p_res)
        x_p_imu_res = self.ddaf(x_p_res, imu_res)  # [N, 96, 128, 128]
        out = self.norm(x_p_imu_res.mean([-2, -1]))  # [N, 96]
        out = self.head(out)  # [N, 4] -> Δx, Δy, Δyaw_sin, Δyaw_cos

        return out, ba, bg


if __name__ == '__main__':
    data1 = torch.randn(1, 3, 512, 512)
    data2 = torch.randn(1, 3, 512, 512)
    point1 = torch.randn(1, 4096, 3)
    point2 = torch.randn(1, 4096, 3)
    imu = torch.randn(1, 128, 10)
    model = SIFNet()
    output = model(data1, data2, point1, point2, imu)
    total_params = count_model_params(model)
    print(f'Total param1eters: {total_params}')
    flops = torchprofile.profile_macs(model, (data1, data2, point1, point2, imu))
    print(f"Flops: {flops}")
    gflops = flops / 1e9  
    print(f"GFlops: {gflops}")

    dummy_inputs = (
        data1,  # frame1_sonar_img
        data2,  # frame2_sonar_img
        point1,  # pointclouds 
        point2,
        imu  # imu_sequence
    )
    summary(model, input_data=dummy_inputs, col_names=["input_size", "output_size", "num_params"])
