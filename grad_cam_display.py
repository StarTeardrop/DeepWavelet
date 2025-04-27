import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import cv2
from customize_dataset import *
from torch.utils.data import DataLoader
from model.sif import *
import warnings

warnings.filterwarnings("ignore")


def collate_fn(batch):
    batch = [item for item in batch if item is not None and len(item) > 0]
    if len(batch) == 0:
        return None  # 或 raise SkipBatchException
    return torch.utils.data.dataloader.default_collate(batch)


def visualize_feature_map_by_error(model, img1, img2, point1, point2, imu_sequence, target_xy, target_yaw_sin,
                                   target_yaw_cos, device, layer_name):
    fmap = None

    def hook_fn(module, input, output):
        nonlocal fmap
        fmap = output.detach()

    if layer_name == 'low_layer':
        layer = model.sf.fuseFeature.outconv_bn_relu_L[0]
    elif layer_name == 'high_layer':
        layer = model.sf.fuseFeature.outconv_bn_relu_H[0]
    elif layer_name == 'global_layer':
        layer = model.sf.fuseFeature.outconv_bn_relu_local[0]  # global与local设置反了
    elif layer_name == 'local_layer':
        layer = model.sf.fuseFeature.outconv_bn_relu_glb[0]
    elif layer_name == 'ddaf_layer':
        layer = model.ddaf
    else:
        raise ValueError("Unknown layer name")


    handle = layer.register_forward_hook(hook_fn)
    _ = model(img1, img2, point1, point2, imu_sequence)
    handle.remove()

    pred = model(img1, img2, point1, point2, imu_sequence)
    loss_x = F.mse_loss(pred[0][:, 0], target_xy[:, 0])
    loss_y = F.mse_loss(pred[0][:, 1], target_xy[:, 1])
    loss_yaw_sin = F.mse_loss(pred[0][:, 2], target_yaw_sin)
    loss_yaw_cos = F.mse_loss(pred[0][:, 3], target_yaw_cos)
    error = loss_x + loss_y + loss_yaw_sin + loss_yaw_cos

    if fmap.dim() == 4:
        fmap = fmap.squeeze(0)  # [C, H, W]

    weighted_map = torch.mean(fmap, dim=0) * error.item()  # [H, W]
    heatmap = weighted_map.detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-6)
    heatmap = cv2.resize(heatmap, (img1.shape[3], img1.shape[2]))  # [W, H]
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img1_np = img1.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    img1_np = (img1_np * 0.5 + 0.5) * 255
    img1_np = img1_np.astype(np.uint8)
    img1_np = cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(img1_np, 0.6, heatmap_color, 0.3, 0)

    return overlay, f"Feature Heatmap from {layer_name}"

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = 'StPereDataset'
    dataset_path = './Datasets/StPereDataset'
    dataset_split_path = './Split_Datasets/StPereDataset'
    dataset_map_name = None
    train_dataset = CustomizeDataset(dataset_name,
                                     dataset_path,
                                     dataset_split_path,
                                     dataset_map_name,
                                     'Train',
                                     transform=True)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True,
                              collate_fn=collate_fn)

    model = SIFNet().to(device)

    model_path = './Checkpoints/latest_checkpoint_StPereDataset.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    low_layer = model.sf.fuseFeature.outconv_bn_relu_L[0]
    high_layer = model.sf.fuseFeature.outconv_bn_relu_H[0]
    global_layer = model.sf.fuseFeature.outconv_bn_relu_glb[0]
    local_layer = model.sf.fuseFeature.outconv_bn_relu_local[0]


    layers_to_visualize = [
        'low_layer',
        'high_layer',
        'global_layer',
        'local_layer',
        'ddaf_layer'
    ]

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            if i == 100:
                img1, img2, point1, point2, imu_sequence, target_xy, target_yaw_sin, target_yaw_cos = data
                img1, img2, point1, point2, imu_sequence, target_xy, target_yaw_sin, target_yaw_cos = img1.to(device), \
                    img2.to(device), \
                    point1.to(device), \
                    point2.to(device), \
                    imu_sequence.to(device), \
                    target_xy.to(device), \
                    target_yaw_sin.to(device), \
                    target_yaw_cos.to(device)

                fig, axes = plt.subplots(1, 6, figsize=(30, 6)) 
                axes = axes.flatten()


                img1_np = img1.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                img1_np = (img1_np * 0.5 + 0.5) * 255
                img1_np = img1_np.astype(np.uint8)
                img1_np = cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR)
                axes[0].imshow(cv2.cvtColor(img1_np, cv2.COLOR_BGR2RGB))
                axes[0].set_title("Original Image")
                axes[0].axis('off')

                for idx, layer_name in enumerate(layers_to_visualize):
                    overlay_img, title = visualize_feature_map_by_error(model, img1, img2, point1, point2, imu_sequence,
                                                                        target_xy, target_yaw_sin, target_yaw_cos,
                                                                        device, layer_name)
                    axes[idx + 1].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
                    axes[idx + 1].set_title(title)
                    axes[idx + 1].axis('off')


                plt.tight_layout()
                plt.show()

                print()
