import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from model.convnext import ConvNeXt
from model.fms import *
from model.ddaf import *
from model.utils import *


class SFNet(nn.Module):
    def __init__(self, decode_channels=96, window_size=8, ):
        super().__init__()
        # self.backbone = ConvNeXt(in_chans=3,
        #                          depths=[3, 3, 1, 3],
        #                          dims=[96, 192, 384, 768],
        #                          drop_path_rate=0.1,
        #                          layer_scale_init_value=1e-6)
        self.backbone = timm.create_model(
            model_name='convnext_tiny.in12k_ft_in1k_384', 
            features_only=True, 
            pretrained=True,  
            output_stride=32, 
            out_indices=(0, 1, 2, 3) 
        )
        self.conv_x2 = ConvBN(192, decode_channels, kernel_size=1)
        self.conv_x3 = ConvBN(384, decode_channels, kernel_size=1)
        self.conv_x4 = ConvBN(768, decode_channels, kernel_size=1)
        self.fuseFeature = FMS(in_ch=3 * decode_channels, out_ch=decode_channels, num_heads=8, window_size=window_size)
        self.ddaf_l = DDAF(decode_channels, num_heads=8, LayerNorm_type='WithBias')
        self.ddaf_h = DDAF(decode_channels, num_heads=8, LayerNorm_type='WithBias')
        self.WF = WF(in_channels=decode_channels, decode_channels=decode_channels)
        self.down = Conv(in_channels=3 * decode_channels, out_channels=decode_channels, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        res1 = x1  # [N, 96, 128, 128]
        res1_h, res1_w = x1.size()[-2:]
        # x1 [N, 96, 128, 128], x2 [N, 192, 64, 64], x3 [N, 384, 32, 32], x4 [N, 768, 16, 16]
        res2 = self.conv_x2(x2)  # [N, 96, 64, 64]
        res3 = self.conv_x3(x3)  # [N, 96, 32, 32]
        res4 = self.conv_x4(x4)  # [N, 96, 16, 16]
        #  [N, 96, 128, 128]
        res2 = F.interpolate(res2, size=(res1_h, res1_w), mode='bilinear', align_corners=True)
        res3 = F.interpolate(res3, size=(res1_h, res1_w), mode='bilinear', align_corners=True)
        res4 = F.interpolate(res4, size=(res1_h, res1_w), mode='bilinear', align_corners=True)
        # [N, 3*96, 128, 128] ([N, 288, 128, 128])
        middle_res = torch.cat([res2, res3, res4], dim=1)
        # [N, 96, 64, 64]
        res_feature_l, res_feature_h, res_feature_global, res_feature_local = self.fuseFeature(middle_res)

        res_local_h= self.ddaf_h(res_feature_h, res_feature_local)  # [N, 96, 64, 64]
        res_global_l = self.ddaf_l(res_feature_l, res_feature_global)  # [N, 96, 64, 64]
        res = self.WF(res_local_h, res_global_l)  # [N, 96, 64, 64]
        middle_res = self.down(middle_res)  # [N, 96, 128, 128]
        res = F.interpolate(res, size=(res1_h, res1_w), mode='bicubic', align_corners=False)  # [N, 96, 128, 128]
        res = middle_res + res  # [N, 96, 128, 128]
        res = self.WF(res, res1)  # [N, 96, 128, 128]

        return res


if __name__ == '__main__':
    data = torch.randn(1, 3, 512, 512)
    model = SFNet()
    output = model(data)
    print(output.shape)
