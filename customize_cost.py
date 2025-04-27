import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomizeCost(nn.Module):
    def __init__(self,
                 lambda_bias=1e-4,
                 smooth_alpha=1e-4,
                 smooth_beta=1e-4,
                 weight_pose=1.0,
                 weight_bias=1.0,
                 weight_smooth=1.0):
        super().__init__()
        self.bias_reg_lambda = lambda_bias
        self.alpha = smooth_alpha
        self.beta = smooth_beta
        self.weight_pose = weight_pose
        self.weight_bias = weight_bias
        self.weight_smooth = weight_smooth

    def forward(self, out, out_ba, out_bg, target_x, target_y, target_yaw_sin, target_yaw_cos):
        loss_ba = torch.mean(torch.norm(out_ba, dim=1)) 
        loss_bg = torch.mean(torch.norm(out_bg, dim=1))
        loss_bias_reg = self.bias_reg_lambda * (loss_ba + loss_bg)
        smooth_ba = torch.sum(torch.norm(out_ba[1:] - out_ba[:-1], dim=1))
        smooth_bg = torch.sum(torch.norm(out_bg[1:] - out_bg[:-1], dim=1))
        loss_smooth = self.alpha * smooth_ba + self.beta * smooth_bg
        loss_x = F.mse_loss(out[:, 0], target_x)
        loss_y = F.mse_loss(out[:, 1], target_y)
        loss_yaw_sin = F.mse_loss(out[:, 2], target_yaw_sin)
        loss_yaw_cos = F.mse_loss(out[:, 3], target_yaw_cos)

        loss_pose = loss_x + loss_y + loss_yaw_sin + loss_yaw_cos
        total_loss = self.weight_pose * loss_pose + self.weight_bias * loss_bias_reg + self.weight_smooth * loss_smooth

        return total_loss, loss_pose, loss_bias_reg, loss_smooth
