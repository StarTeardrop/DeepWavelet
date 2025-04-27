import sys
import os
import io
import json
from contextlib import redirect_stdout
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from model.sif import SIFNet
from model.utils import count_model_params, torchprofile
from customize_dataset import CustomizeDataset
from customize_cost import CustomizeCost
from Tools.logger_config import *
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


def collate_fn(batch):
    batch = [item for item in batch if item is not None and len(item) > 0]
    if len(batch) == 0:
        return None  # 或 raise SkipBatchException
    return torch.utils.data.dataloader.default_collate(batch)


def train_one_epoch(epoch, total_epochs, model, dataloader, device, criterion, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    total_loss_pose = 0.0
    total_loss_bias_reg = 0.0
    total_loss_bias_smooth = 0.0

    total_loss_list = []
    total_loss_pose_list = []
    total_loss_bias_reg_list = []
    total_loss_bias_smooth_list = []

    pbar = tqdm(
        dataloader, desc=f"Training {epoch}/{total_epochs}: ", miniters=1)
    for data in pbar:
        if data is None:
            continue
        sonar_data1, sonar_data2, point_data1, point_data2, imu_data, target_xy, target_yaw_sin, target_yaw_cos = data
        sonar_data1, sonar_data2 = sonar_data1.to(
            device), sonar_data2.to(device)
        point_data1, point_data2 = point_data1.to(
            device), point_data2.to(device)
        imu_data = imu_data.to(device)
        target_xy, target_yaw_sin, target_yaw_cos = target_xy.to(device), target_yaw_sin.to(device), target_yaw_cos.to(
            device)

        out, ba, bg = model(sonar_data1, sonar_data2,
                            point_data1, point_data2, imu_data)
        loss, loss_pose, loss_bias_reg, loss_smooth = criterion(out, ba, bg, target_xy[:, 0], target_xy[:, 1],
                                                                target_yaw_sin, target_yaw_cos)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_loss_pose += loss_pose.item()
        total_loss_bias_reg += loss_bias_reg.item()
        total_loss_bias_smooth += loss_smooth.item()

        total_loss_list.append(loss.item())
        total_loss_pose_list.append(loss_pose.item())
        total_loss_bias_reg_list.append(loss_bias_reg.item())
        total_loss_bias_smooth_list.append(loss_smooth.item())

        pbar.set_postfix(
            total_loss=loss.item(),
            loss_bias_reg=loss_bias_reg.item(),
            loss_bias_smooth=loss_smooth.item(),
            loss_pose=loss_pose.item(),
        )

    total_loss_min = min(total_loss_list)
    total_loss_max = max(total_loss_list)
    total_loss_pose_min = min(total_loss_pose_list)
    total_loss_pose_max = max(total_loss_pose_list)
    total_loss_bias_reg_min = min(total_loss_bias_reg_list)
    total_loss_bias_reg_max = max(total_loss_bias_reg_list)
    total_loss_bias_smooth_min = min(total_loss_bias_smooth_list)
    total_loss_bias_smooth_max = max(total_loss_bias_smooth_list)

    data_dict = {'total_loss_min': total_loss_min,
                 'total_loss_max': total_loss_max,
                 'total_loss_pose_min': total_loss_pose_min,
                 'total_loss_pose_max': total_loss_pose_max,
                 'total_loss_bias_reg_min': total_loss_bias_reg_min,
                 'total_loss_bias_reg_max': total_loss_bias_reg_max,
                 'total_loss_bias_smooth_min': total_loss_bias_smooth_min,
                 'total_loss_bias_smooth_max': total_loss_bias_smooth_max,
                 'total_loss_avg': total_loss / len(dataloader),
                 'total_loss_pose_avg': total_loss_pose / len(dataloader),
                 'total_loss_bias_reg_avg': total_loss_bias_reg / len(dataloader),
                 'total_loss_bias_smooth_avg': total_loss_bias_smooth / len(dataloader)}

    return total_loss / len(dataloader), total_loss_pose / len(dataloader), total_loss_bias_reg / len(
        dataloader), total_loss_bias_smooth / len(dataloader), data_dict


def evaluate(epoch, total_epochs, model, dataloader, device, criterion, evaluate_type: str):
    model.eval()
    total_loss = 0.0
    total_loss_pose = 0.0
    total_loss_bias_reg = 0.0
    total_loss_bias_smooth = 0.0

    total_loss_list = []
    total_loss_pose_list = []
    total_loss_bias_reg_list = []
    total_loss_bias_smooth_list = []

    pbar = tqdm(
        dataloader, desc=f"{evaluate_type} {epoch}/{total_epochs}: ", miniters=1)
    with torch.no_grad():
        for data in pbar:
            if data is None:
                continue
            sonar_data1, sonar_data2, point_data1, point_data2, imu_data, target_xy, target_yaw_sin, target_yaw_cos = data
            sonar_data1, sonar_data2 = sonar_data1.to(
                device), sonar_data2.to(device)
            point_data1, point_data2 = point_data1.to(
                device), point_data2.to(device)
            imu_data = imu_data.to(device)
            target_xy, target_yaw_sin, target_yaw_cos = target_xy.to(device), target_yaw_sin.to(
                device), target_yaw_cos.to(device)

            out, ba, bg = model(sonar_data1, sonar_data2,
                                point_data1, point_data2, imu_data)
            loss, loss_pose, loss_bias_reg, loss_smooth = criterion(out, ba, bg, target_xy[:, 0], target_xy[:, 1],
                                                                    target_yaw_sin, target_yaw_cos)
            total_loss += loss.item()
            total_loss_pose += loss_pose.item()
            total_loss_bias_reg += loss_bias_reg.item()
            total_loss_bias_smooth += loss_smooth.item()

            total_loss_list.append(loss.item())
            total_loss_pose_list.append(loss_pose.item())
            total_loss_bias_reg_list.append(loss_bias_reg.item())
            total_loss_bias_smooth_list.append(loss_smooth.item())
            pbar.set_postfix(
                total_loss=loss.item(),
                loss_bias_reg=loss_bias_reg.item(),
                loss_bias_smooth=loss_smooth.item(),
                loss_pose=loss_pose.item(),
            )

    total_loss_min = min(total_loss_list)
    total_loss_max = max(total_loss_list)
    total_loss_pose_min = min(total_loss_pose_list)
    total_loss_pose_max = max(total_loss_pose_list)
    total_loss_bias_reg_min = min(total_loss_bias_reg_list)
    total_loss_bias_reg_max = max(total_loss_bias_reg_list)
    total_loss_bias_smooth_min = min(total_loss_bias_smooth_list)
    total_loss_bias_smooth_max = max(total_loss_bias_smooth_list)

    data_dict = {'total_loss_min': total_loss_min,
                 'total_loss_max': total_loss_max,
                 'total_loss_pose_min': total_loss_pose_min,
                 'total_loss_pose_max': total_loss_pose_max,
                 'total_loss_bias_reg_min': total_loss_bias_reg_min,
                 'total_loss_bias_reg_max': total_loss_bias_reg_max,
                 'total_loss_bias_smooth_min': total_loss_bias_smooth_min,
                 'total_loss_bias_smooth_max': total_loss_bias_smooth_max,
                 'total_loss_avg': total_loss / len(dataloader),
                 'total_loss_pose_avg': total_loss_pose / len(dataloader),
                 'total_loss_bias_reg_avg': total_loss_bias_reg / len(dataloader),
                 'total_loss_bias_smooth_avg': total_loss_bias_smooth / len(dataloader)}

    return total_loss / len(dataloader), total_loss_pose / len(dataloader), total_loss_bias_reg / len(
        dataloader), total_loss_bias_smooth / len(dataloader), data_dict


def load_checkpoint(model, optimizer, checkpoint_path):
    """ Load model and optimizer state from a checkpoint file. """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    logger.info(
        f"Resumed training from epoch {epoch} with best val loss {best_val_loss:.4f}")
    return epoch, best_val_loss


def parse_args():
    parser = argparse.ArgumentParser(description="DeepWavelet training script")
    parser.add_argument('--log_path', type=str, default='./log', help='Logs save path')
    parser.add_argument('--data_save_path', type=str, default='./Data', help='Data save path')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=150, help='training epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--checkpoints_path', type=str, default='./Checkpoints', help='checkpoints save path')
    parser.add_argument('--load_pretrained_model', type=bool, default=False, help='load pretrained model or not')
    parser.add_argument('--resume_training', type=bool, default=True, help='resume training or not')
    parser.add_argument('--dataset_name', type=str, default='StPereDataset', help='dataset name')
    parser.add_argument('--dataset_path', type=str, default='./Datasets/StPereDataset', help='dataset path')
    parser.add_argument('--dataset_split_path', type=str, default='./Split_Datasets/StPereDataset',
                        help='dataset split path')
    parser.add_argument('--dataset_map_name', type=str, default='None',
                        help='if dataset name is StPereDataset, set map name None')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    log_dir = args.log_path
    data_save_path = args.data_save_path
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'Train_{args.dataset_name}_{current_time}.log')
    file_handler = logging.FileHandler(log_file, mode='a')  
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    device = torch.device(args.device)
    save_dir = args.checkpoints_path
    os.makedirs(save_dir, exist_ok=True)

    is_load_pretrained_model = args.load_pretrained_model
    is_resume_training = args.resume_training

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)  
        total_memory = torch.cuda.get_device_properties(
            0).total_memory / 1024 ** 3  
        logger.info(
            f" Using GPU: {device_name}, Total memory: {total_memory:.2f} GB")

    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    dataset_split_path = args.dataset_split_path
    dataset_map_name = args.dataset_map_name
    if dataset_map_name == 'None':
        dataset_map_name = None

    train_dataset = CustomizeDataset(dataset_name,
                                     dataset_path,
                                     dataset_split_path,
                                     dataset_map_name,
                                     'Train',
                                     transform=True)
    val_dataset = CustomizeDataset(dataset_name,
                                   dataset_path,
                                   dataset_split_path,
                                   dataset_map_name,
                                   'Val',
                                   transform=True)
    test_dataset = CustomizeDataset(dataset_name,
                                    dataset_path,
                                    dataset_split_path,
                                    dataset_map_name,
                                    'Test',
                                    transform=True)
    logger.info("  Loading dataset...")
    logger.info(
        f"  Train_dataset size: {len(train_dataset)}, Val_dataset size: {len(val_dataset)}, Test_dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True,
                             collate_fn=collate_fn)
    logger.info(
        f"  Train_loader size: {len(train_loader)}, Val_loader size: {len(val_loader)}, Test_loader size: {len(test_loader)}")

    model = SIFNet().to(device)
    model_params = count_model_params(model)
    logger.info(f'  Model parameters count: {model_params}')
    data1 = torch.randn(1, 3, 512, 512).to(device)
    data2 = torch.randn(1, 3, 512, 512).to(device)
    point1 = torch.randn(1, 4096, 3).to(device)
    point2 = torch.randn(1, 4096, 3).to(device)
    imu = torch.randn(1, 200, 10).to(device)
    dummy_inputs = (
        data1,
        data2,
        point1,
        point2,
        imu
    )

    str_buffer = io.StringIO()
    with redirect_stdout(str_buffer):
        summary(model, input_data=dummy_inputs, col_names=[
            "input_size", "output_size", "num_params"])
    summary_str = str_buffer.getvalue()
    logger.info("\n" + summary_str)

    criterion = CustomizeCost(lambda_bias=1.0,
                              smooth_alpha=1.0,
                              smooth_beta=1.0,
                              weight_pose=1.0,
                              weight_bias=1.0,
                              weight_smooth=1.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    warmup_ratio = args.warmup_ratio
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps) 
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    checkpoint_path = os.path.join(save_dir, f'latest_checkpoint_{dataset_name}.pth')
    model_save_path = os.path.join(save_dir, f"best_model_{dataset_name}.pth")
    if dataset_map_name is not None:
        checkpoint_path = os.path.join(save_dir, f'latest_checkpoint_{dataset_name}_{dataset_map_name}.pth')
        model_save_path = os.path.join(save_dir, f"best_model_{dataset_name}_{dataset_map_name}.pth")
    start_epoch = 0
    best_val_loss = float('inf')

    if is_resume_training:
        if os.path.exists(checkpoint_path):
            logger.info(
                f"Loading resume training weights from: {checkpoint_path}")
            start_epoch, best_val_loss = load_checkpoint(
                model, optimizer, checkpoint_path)

    elif is_load_pretrained_model:
        logger.info(f"Loading pretrained weights from: {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))

    logger.info(" Start training...")

    if dataset_map_name is not None:
        train_data_save_path = os.path.join(data_save_path,
                                            f'{dataset_name}_{dataset_map_name}_train_{current_time}.txt')
        val_data_save_path = os.path.join(data_save_path, f'{dataset_name}_{dataset_map_name}_val_{current_time}.txt')
        test_data_save_path = os.path.join(data_save_path, f'{dataset_name}_{dataset_map_name}_test_{current_time}.txt')
    else:
        train_data_save_path = os.path.join(data_save_path, f'{dataset_name}_train_{current_time}.txt')
        val_data_save_path = os.path.join(data_save_path, f'{dataset_name}_val_{current_time}.txt')
        test_data_save_path = os.path.join(data_save_path, f'{dataset_name}_test_{current_time}.txt')

    for epoch in range(start_epoch, num_epochs):
        logger.info(
            f" Epoch starting: {epoch + 1} -------------------------------------->")
        train_loss, train_loss_pose, train_loss_bias_reg, train_loss_bias_smooth, train_data_dict = train_one_epoch(
            epoch + 1, num_epochs, model, train_loader, device, criterion, optimizer, scheduler)
        logger.info(
            f"[Epoch {epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} | Train Loss Pose: {train_loss_pose:.4f} "
            f"| Train Loss Bias Reg: {train_loss_bias_reg:.12f} | Train Loss Bias Smooth: {train_loss_bias_smooth:.12f} "
            f"| LR: {optimizer.param_groups[0]['lr']:.6f}")

        val_loss, val_loss_pose, val_loss_bias_reg, val_loss_bias_smooth, val_data_dict = evaluate(epoch + 1,
                                                                                                   num_epochs,
                                                                                                   model,
                                                                                                   val_loader,
                                                                                                   device,
                                                                                                   criterion,
                                                                                                   evaluate_type='Val')
        logger.info(
            f"[Epoch {epoch + 1}/{num_epochs}] Val Loss: {val_loss:.4f} | Val Loss Pose: {val_loss_pose:.4f} "
            f"| Val Loss Bias Reg: {val_loss_bias_reg:.12f} | Val Loss Bias Smooth: {val_loss_bias_smooth:.12f}")
        test_loss, test_loss_pose, test_loss_bias_reg, test_loss_bias_smooth, test_data_dict = evaluate(epoch + 1,
                                                                                                        num_epochs,
                                                                                                        model,
                                                                                                        test_loader,
                                                                                                        device,
                                                                                                        criterion,
                                                                                                        evaluate_type='Test')

        logger.info(
            f"[Epoch {epoch + 1}/{num_epochs}] Test Loss: {test_loss:.4f} | Test Loss Pose: {test_loss_pose:.4f} "
            f"| Test Loss Bias Reg: {test_loss_bias_reg:.12f} | Test Loss Bias Smooth: {test_loss_bias_smooth:.12f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f" Best model updated at epoch {epoch + 1}")

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(
            f" Lasted model is updated at epoch: {epoch + 1} in {checkpoint_path}")

        logger.info(
            f" Epoch ending: {epoch + 1} <-----------------------------------------")
        logger.info('')

        with open(train_data_save_path, "a") as f:
            json.dump(train_data_dict, f)
            f.write("\n")

        with open(val_data_save_path, "a") as f:
            json.dump(val_data_dict, f)
            f.write("\n")

        with open(test_data_save_path, "a") as f:
            json.dump(test_data_dict, f)
            f.write("\n")

    logger.info(" Training finished...")
