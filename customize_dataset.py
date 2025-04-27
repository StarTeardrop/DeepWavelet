import os
import ast
import cv2
import time as time_sys

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from Tools.logger_config import logger
from Tools.utils import *
from Tools.imu_pre_integrator import *
from Tools.sonar_image import *
from Tools.customize_transform import *

dataset_list = ['StPereDataset', 'CustomizeDataset']
dataset_type = ['Train', 'Test', 'Val']

stpere_acc_mean = [-0.38232704, -0.30764668, -9.76701361]
stpere_acc_std = [0.07167835, 0.16780735, 0.03995271]
stpere_gyro_mean = [0.00158169, 0.00259857, 0.01214766]
stpere_gyro_std = [0.01101386, 0.01016888, 0.0420227]
stpere_x_min = -4.27
stpere_x_max = 3.58
stpere_y_min = -5.93
stpere_y_max = 4.26

customize_map1_acc_mean = [-0.21430491, 0.01669972, 9.7940403]
customize_map1_acc_std = [0.77429883, 0.17171059, 0.28620792]
customize_map1_gyro_mean = [-0.00079409, 0.00015851, 0.02943916]
customize_map1_gyro_std = [0.0067883, 0.0177805, 0.12930705]
customize_map1_x_min = -1.67
customize_map1_x_max = 1.67
customize_map1_y_min = -1.68
customize_map1_y_max = 1.65

customize_map2_acc_mean = [-0.26284583, 0.0148343, 9.79274917]
customize_map2_acc_std = [0.78356049, 0.0481006, 0.27698049]
customize_map2_gyro_mean = [-0.0007741, 0.00015003, 0.0290295]
customize_map2_gyro_std = [0.00569903, 0.01845033, 0.13794111]
customize_map2_x_min = -1.67
customize_map2_x_max = 1.66
customize_map2_y_min = -1.66
customize_map2_y_max = 1.64

customize_map3_acc_mean = [-0.23414587, 0.01911273, 9.79358213]
customize_map3_acc_std = [0.7542369, 0.04169458, 0.25859259]
customize_map3_gyro_mean = [-0.00102239, 0.00035167, 0.04184699]
customize_map3_gyro_std = [0.00552521, 0.01786503, 0.15862547]
customize_map3_x_min = -1.64
customize_map3_x_max = 1.67
customize_map3_y_min = -1.64
customize_map3_y_max = 1.64


class CustomizeDataset(Dataset):
    def __init__(self, dataset_name: str,
                 dataset_path: str,
                 split_dataset_path: str,
                 map_type: str,
                 dataset_type: str,
                 transform: bool):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.split_dataset_path = split_dataset_path
        self.map_type = map_type
        self.dataset_type = dataset_type
        self.transform = transform

        if self.map_type is None:
            self.acc_mean = stpere_acc_mean
            self.acc_std = stpere_acc_std
            self.gyro_mean = stpere_gyro_mean
            self.gyro_std = stpere_gyro_std
            self.x_min = stpere_x_min
            self.x_max = stpere_x_max
            self.y_min = stpere_y_min
            self.y_max = stpere_y_max
        elif self.map_type == 'map1':
            self.acc_mean = customize_map1_acc_mean
            self.acc_std = customize_map1_acc_std
            self.gyro_mean = customize_map1_gyro_mean
            self.gyro_std = customize_map1_gyro_std
            self.x_min = customize_map1_x_min
            self.x_max = customize_map1_x_max
            self.y_min = customize_map1_y_min
            self.y_max = customize_map1_y_max
        elif self.map_type == 'map2':
            self.acc_mean = customize_map2_acc_mean
            self.acc_std = customize_map2_acc_std
            self.gyro_mean = customize_map2_gyro_mean
            self.gyro_std = customize_map2_gyro_std
            self.x_min = customize_map2_x_min
            self.x_max = customize_map2_x_max
            self.y_min = customize_map2_y_min
            self.y_max = customize_map2_y_max
        elif self.map_type == 'map3':
            self.acc_mean = customize_map3_acc_mean
            self.acc_std = customize_map3_acc_std
            self.gyro_mean = customize_map3_gyro_mean
            self.gyro_std = customize_map3_gyro_std
            self.x_min = customize_map3_x_min
            self.x_max = customize_map3_x_max
            self.y_min = customize_map3_y_min
            self.y_max = customize_map3_y_max

        if self.dataset_name not in dataset_list:
            raise ValueError(f"Dataset name should be one of {dataset_list}")

        if self.dataset_type not in dataset_type:
            raise ValueError(f"Dataset type should be one of {dataset_type}")

        if not os.listdir(self.split_dataset_path):
            raise ValueError(f"Split dataset path is empty, please check the path: {self.split_dataset_path}")

    def __len__(self):
        data_len = 0
        if self.dataset_name == 'StPereDataset':
            for interval_name in os.listdir(self.split_dataset_path):
                interval_path = os.path.join(self.split_dataset_path, interval_name)
                data_type_path = os.path.join(interval_path, self.dataset_type)
                data_len += len(os.listdir(data_type_path))

        if self.dataset_name == 'CustomizeDataset':
            for interval_name in os.listdir(self.split_dataset_path):
                map_path = os.path.join(self.split_dataset_path, interval_name, self.map_type)
                data_type_path = os.path.join(map_path, self.dataset_type)
                data_len += len(os.listdir(data_type_path))
        return data_len

    def __getitem__(self, item):
        root_dir_list = []
        imu_pre_integration_list = []
        sonar_data_list = []
        ground_truth_list = []

        imu_pre_integration_list_older = []
        sonar_data_list_older = []

        imu_transform_list = []
        imu_transform_list_older = []
        sonar_transform_list = []
        sonar_transform_list_older = []
        ground_truth_transform_list = []

        imu_pre_integration_relative_xy = None
        imu_pre_integration_relative_yaw = None
        sonar_final_image = None
        ground_truth_relative_robot_xy = None
        ground_truth_relative_imu_yaw = None

        if self.dataset_name == 'StPereDataset':
            for interval_name in os.listdir(self.split_dataset_path):
                interval_path = os.path.join(self.split_dataset_path, interval_name)
                data_type_path = os.path.join(interval_path, self.dataset_type)
                for data_dir in os.listdir(data_type_path):
                    root_dir = os.path.join(data_type_path, data_dir)
                    root_dir_list.append(root_dir)

            data_root_dir = root_dir_list[item]
            imu_data_root_path = os.path.join(data_root_dir, 'imu.txt')
            sonar_data_root_path = os.path.join(data_root_dir, 'sonar.txt')
            ground_truth_root_path = os.path.join(data_root_dir, 'ground_truth.txt')
            sonar_data_root_path_older = os.path.join(data_root_dir, 'sonar_older.txt')

            with open(imu_data_root_path, 'r') as f:
                for line in f.readlines():
                    imu_pre_integration_list.append(ast.literal_eval(line))  
            with open(sonar_data_root_path, 'r') as f:
                for line in f.readlines():
                    sonar_data_list.append(ast.literal_eval(line)) 
            with open(sonar_data_root_path_older, 'r') as f:
                for line in f.readlines():
                    sonar_data_list_older.append(ast.literal_eval(line)) 
            with open(ground_truth_root_path, 'r') as f:
                for line in f.readlines():
                    ground_truth_list.append(ast.literal_eval(line))  

            if len(ground_truth_list) >= 2:
                for imu in imu_pre_integration_list:
                    roll, pitch, yaw, ax, ay, az, gx, gy, gz = float(imu['roll']), float(imu['pitch']), float(
                        imu['yaw']), \
                        float(imu['ax']), float(imu['ay']), float(imu['az']), \
                        float(imu['gx']), float(imu['gy']), float(imu['gz'])
                    time = float(imu['time'])
                    roll_new, pitch_new, yaw_new, ax_new, ay_new, az_new, gx_new, gy_new, gz_new = stpere_rotation_imu(
                        roll,
                        pitch,
                        yaw,
                        ax,
                        ay,
                        az,
                        gx,
                        gy,
                        gz)
                    imu_transform_list.append(
                        [time, roll_new, pitch_new, yaw_new, ax_new, ay_new, az_new, gx_new, gy_new, gz_new])

                for sonar in sonar_data_list:
                    time, yaw, bins = float(sonar['time']), float(sonar['transducerAngle']), sonar['bins']
                    yaw_new = stpere_rotation_sonar(yaw)
                    sonar_transform_list.append([time, yaw_new, bins])
                for sonar in sonar_data_list_older:
                    time, yaw, bins = float(sonar['time']), float(sonar['transducerAngle']), sonar['bins']
                    yaw_new = stpere_rotation_sonar(yaw)
                    sonar_transform_list_older.append([time, yaw_new, bins])

                imu_position, imu_velocity, imu_attitude = imu_pre_integrator(imu_transform_list)
                imu_pre_integration_relative_xy = imu_position[:2]
                imu_pre_integration_relative_yaw = imu_attitude[2]
                lat1, lon1, lat2, lon2 = float(ground_truth_list[0]['latitude']), float(
                    ground_truth_list[0]['longitude']), \
                    float(ground_truth_list[-1]['latitude']), float(ground_truth_list[-1]['longitude'])
                ground_truth_relative_gps_xy = stpere_gps_distance(lat1, lon1, lat2, lon2)
                ground_truth_relative_robot_xy = ground_truth_relative_gps_xy
                ground_truth_relative_imu_yaw = imu_transform_list[-1][3] - imu_transform_list[0][3]

                sonar_image_older = generate_stpere_sonar_image(sonar_transform_list_older, img_size=(512, 512))
                sonar_image = generate_stpere_sonar_image(sonar_transform_list, img_size=(512, 512))
                sonar_image_older_gray = cv2.cvtColor(sonar_image_older, cv2.COLOR_BGR2GRAY)
                sonar_image_gray = cv2.cvtColor(sonar_image, cv2.COLOR_BGR2GRAY)
                older_cfar_mask = stpere_soca_cfar(sonar_image_older_gray, guard_size=2, background_size=5,
                                                   threshold_scale=1.3)
                cfar_mask = stpere_soca_cfar(sonar_image_gray, guard_size=2, background_size=5, threshold_scale=1.3)
                older_denoised_bgr = cv2.bitwise_and(sonar_image_older, sonar_image_older, mask=older_cfar_mask)
                denoised_bgr = cv2.bitwise_and(sonar_image, sonar_image, mask=cfar_mask)
                older_point_cloud = generate_stpere_sonar_point_cloud(older_denoised_bgr, threshold=220)
                point_cloud = generate_stpere_sonar_point_cloud(denoised_bgr, threshold=220)

                if self.transform:
                    older_denoised_rgb = cv2.cvtColor(older_denoised_bgr, cv2.COLOR_BGR2RGB)
                    denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
                    sonar_image_older_tensor = sonar_rgb_img_transform(older_denoised_rgb)
                    sonar_image_tensor = sonar_rgb_img_transform(denoised_rgb)
                    older_point_cloud_tensor = point_cloud_transform(older_point_cloud)
                    point_cloud_tensor = point_cloud_transform(point_cloud)
                    imu_pre_integration_relative_xy_tensor = value_general_transform(imu_pre_integration_relative_xy)
                    imu_pre_integration_relative_yaw_tensor = value_general_transform(imu_pre_integration_relative_yaw)
                    ground_truth_relative_robot_xy = ground_true_xy_normal(ground_truth_relative_robot_xy, self.x_min,
                                                                           self.x_max, self.y_min, self.y_max)
                    ground_truth_relative_robot_xy_tensor = value_general_transform(ground_truth_relative_robot_xy)
                    ground_truth_relative_imu_yaw = np.radians(ground_truth_relative_imu_yaw)
                    ground_truth_relative_imu_yaw_sin = np.sin(ground_truth_relative_imu_yaw)
                    ground_truth_relative_imu_yaw_cos = np.cos(ground_truth_relative_imu_yaw)
                    ground_truth_relative_imu_yaw_sin_tensor = value_general_transform(
                        ground_truth_relative_imu_yaw_sin)
                    ground_truth_relative_imu_yaw_cos_tensor = value_general_transform(
                        ground_truth_relative_imu_yaw_cos)
                    imu_process = process_imu_data(imu_transform_list)
                    imu_process = imu_to_normal(imu_process, self.acc_mean, self.acc_std, self.gyro_mean,
                                                self.gyro_std)
                    imu_list_tensor = value_general_transform(imu_process)

                    return sonar_image_older_tensor, \
                        sonar_image_tensor, \
                        older_point_cloud_tensor, \
                        point_cloud_tensor, \
                        imu_list_tensor, \
                        ground_truth_relative_robot_xy_tensor, \
                        ground_truth_relative_imu_yaw_sin_tensor, \
                        ground_truth_relative_imu_yaw_cos_tensor
            else:
                return None

        if self.dataset_name == 'CustomizeDataset':
            for interval_name in os.listdir(self.split_dataset_path):
                map_path = os.path.join(self.split_dataset_path, interval_name, self.map_type)
                data_type_path = os.path.join(map_path, self.dataset_type)
                for data_dir in os.listdir(data_type_path):
                    root_dir = os.path.join(data_type_path, data_dir)
                    root_dir_list.append(root_dir)

            data_root_dir = root_dir_list[item]
            imu_data_root_path = os.path.join(data_root_dir, 'imu.txt')
            sonar_data_root_path = os.path.join(data_root_dir, 'sonar.txt')
            ground_truth_root_path = os.path.join(data_root_dir, 'ground_truth.txt')
            with open(imu_data_root_path, 'r') as f:
                for line in f.readlines():
                    imu_pre_integration_list.append(ast.literal_eval(line))  

            with open(sonar_data_root_path, 'r') as f:
                for line in f.readlines():
                    sonar_data_list.append(ast.literal_eval(line)) 

            with open(ground_truth_root_path, 'r') as f:
                for line in f.readlines():
                    ground_truth_list.append(ast.literal_eval(line)) 

            if len(ground_truth_list) >= 2:
                for imu in imu_pre_integration_list:
                    time, qx, qy, qz, qw, ax, ay, az, gx, gy, gz = float(imu['time']), float(imu['qx']), float(
                        imu['qy']), \
                        float(imu['qz']), float(imu['qw']), float(imu['ax']), float(imu['ay']), float(imu['az']), \
                        float(imu['gx']), float(imu['gy']), float(imu['gz'])
                    r = R.from_quat([qx, qy, qz, qw])
                    euler_angles = r.as_euler('xyz', degrees=True) 
                    roll, pitch, yaw = euler_angles
                    imu_transform_list.append([time, roll, pitch, yaw, ax, ay, az, gx, gy, gz])

                imu_position, imu_velocity, imu_attitude = imu_pre_integrator(imu_transform_list)
                imu_pre_integration_relative_xy = imu_position[:2]
                imu_pre_integration_relative_yaw = imu_attitude[2]

                for gt in ground_truth_list:
                    time, qx, qy, qz, qw, x, y, z = float(gt['time']), float(gt['qx']), float(gt['qy']), \
                        float(gt['qz']), float(gt['qw']), float(gt['x']), float(gt['y']), float(gt['z'])
                    r = R.from_quat([qx, qy, qz, qw])
                    euler_angles = r.as_euler('xyz', degrees=True)  

                    roll, pitch, yaw = euler_angles
                    ground_truth_transform_list.append([time, roll, pitch, yaw, x, y, z])
                ground_truth_relative_robot_xy = [a - b for a, b in zip(ground_truth_transform_list[-1][4:6],
                                                                        ground_truth_transform_list[0][4:6])]
                ground_truth_relative_imu_yaw = ground_truth_transform_list[-1][3] - ground_truth_transform_list[0][3]

                sonar_older_name = sonar_data_list[0]['sonar_image_path'].split('./')[1]
                sonar_name = sonar_data_list[-1]['sonar_image_path'].split('./')[1]
                sonar_older_path = os.path.join(self.dataset_path, self.map_type, sonar_older_name)
                sonar_path = os.path.join(self.dataset_path, self.map_type, sonar_name)
                sonar_image_older = cv2.imread(sonar_older_path)
                sonar_image = cv2.imread(sonar_path)
                sonar_image_older_gray = cv2.cvtColor(sonar_image_older, cv2.COLOR_BGR2GRAY)
                sonar_image_gray = cv2.cvtColor(sonar_image, cv2.COLOR_BGR2GRAY)
                older_cfar_mask = stpere_soca_cfar(sonar_image_older_gray, guard_size=2, background_size=5,
                                                   threshold_scale=1.3)
                cfar_mask = stpere_soca_cfar(sonar_image_gray, guard_size=2, background_size=5, threshold_scale=1.3)
                older_denoised_bgr = cv2.bitwise_and(sonar_image_older, sonar_image_older, mask=older_cfar_mask)
                denoised_bgr = cv2.bitwise_and(sonar_image, sonar_image, mask=cfar_mask)
                older_point_cloud = generate_customize_sonar_point_cloud(older_denoised_bgr, threshold=120)
                point_cloud = generate_customize_sonar_point_cloud(denoised_bgr, threshold=120)

                if self.transform:
                    older_denoised_bgr = make_square_top(older_denoised_bgr)
                    denoised_bgr = make_square_top(denoised_bgr)
                    older_denoised_rgb = cv2.cvtColor(older_denoised_bgr, cv2.COLOR_BGR2RGB)
                    denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)

                    sonar_image_older_tensor = sonar_rgb_img_transform(older_denoised_rgb)
                    sonar_image_tensor = sonar_rgb_img_transform(denoised_rgb)
                    older_point_cloud_tensor = point_cloud_transform(older_point_cloud)
                    point_cloud_tensor = point_cloud_transform(point_cloud)
                    imu_pre_integration_relative_xy_tensor = value_general_transform(imu_pre_integration_relative_xy)
                    imu_pre_integration_relative_yaw_tensor = value_general_transform(imu_pre_integration_relative_yaw)
                    ground_truth_relative_robot_xy = ground_true_xy_normal(ground_truth_relative_robot_xy,
                                                                           self.x_min,
                                                                           self.x_max, self.y_min,
                                                                           self.y_max)
                    ground_truth_relative_robot_xy_tensor = value_general_transform(ground_truth_relative_robot_xy)
                    ground_truth_relative_imu_yaw = np.radians(ground_truth_relative_imu_yaw)
                    ground_truth_relative_imu_yaw_sin = np.sin(ground_truth_relative_imu_yaw)
                    ground_truth_relative_imu_yaw_cos = np.cos(ground_truth_relative_imu_yaw)
                    ground_truth_relative_imu_yaw_sin_tensor = value_general_transform(
                        ground_truth_relative_imu_yaw_sin)
                    ground_truth_relative_imu_yaw_cos_tensor = value_general_transform(
                        ground_truth_relative_imu_yaw_cos)
                    imu_process = process_imu_data(imu_transform_list)
                    imu_process = imu_to_normal(imu_process, self.acc_mean, self.acc_std, self.gyro_mean,
                                                self.gyro_std)
                    imu_list_tensor = value_general_transform(imu_process)

                    return sonar_image_older_tensor, \
                        sonar_image_tensor, \
                        older_point_cloud_tensor, \
                        point_cloud_tensor, \
                        imu_list_tensor, \
                        ground_truth_relative_robot_xy_tensor, \
                        ground_truth_relative_imu_yaw_sin_tensor, \
                        ground_truth_relative_imu_yaw_cos_tensor

        return None


if __name__ == '__main__':


    test_dataset = CustomizeDataset('CustomizeDataset',
                                    './Datasets/CustomizeDataset',
                                    './Split_Datasets/CustomizeDataset',
                                    'map3',
                                    'Train',
                                    transform=True)

    print(len(test_dataset))
    all_x = []
    all_y = []
    for data in test_dataset:
        if data is not None:
            all_x.append(data[5].numpy()[0])
            all_y.append(data[5].numpy()[1])
            print([data[5].numpy()[0], data[5].numpy()[1]])
        print(len(all_x))
    x_max = max(all_x)
    x_min = min(all_x)
    y_max = max(all_y)
    y_min = min(all_y)
    print("Customize max (x, y):", x_max, y_max)
    print("Customize min (x, y):", x_min, y_min)
