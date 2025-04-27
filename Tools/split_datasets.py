import os
import sys
import numpy as np
import random
from bisect import bisect_left
from logger_config import logger

dataset_list = ['StPereDataset', 'CustomizeDataset']


class Split_Datasets:
    def __init__(self, dataset_name: str, dataset_path: str, is_frame2frame: bool, frame_interval: int,
                 dataset_save_path: str, map_type: str):
        self.dataset_name = dataset_name
        if self.dataset_name not in dataset_list:
            logger.error(f'{self.dataset_name} is not supported.')
            return

        self.dataset_path = dataset_path
        self.is_frame2frame = is_frame2frame
        self.frame_interval = frame_interval
        self.dataset_save_path = dataset_save_path
        self.map_type = map_type
        if self.map_type is not None:
            self.dataset_path = os.path.join(self.dataset_path, self.map_type)

        self.imu_data = []
        self.ground_true_data = []
        self.sonar_data = []

        self.save_data_list = []

        random.seed(0)

    def detect_data_type(self):
        if self.dataset_name == 'StPereDataset':
            self.read_stpere_dataset()
        if self.dataset_name == 'CustomizeDataset':
            self.read_customize_dataset()

    def read_customize_dataset(self):
        if os.path.exists(self.dataset_path):
            logger.info(f'Find {self.dataset_name} dataset path: {self.dataset_path}...')
            for file_name in os.listdir(self.dataset_path):
                if file_name.endswith('imu.txt'):
                    logger.info(f'Find {file_name}...')
                    with open(os.path.join(self.dataset_path, file_name)) as f:
                        for line in f:
                            if line.startswith('\n'):
                                continue
                            parts = line.split()
                            imu_data_dict = {'time': parts[0],
                                             'qx': parts[1],
                                             'qy': parts[2],
                                             'qz': parts[3],
                                             'qw': parts[4],
                                             'ax': parts[5],
                                             'ay': parts[6],
                                             'az': parts[7],
                                             'gx': parts[8],
                                             'gy': parts[9],
                                             'gz': parts[10]}
                            self.imu_data.append(imu_data_dict)
                elif file_name.endswith('ground_truth.txt'):
                    logger.info(f'Find {file_name}...')
                    with open(os.path.join(self.dataset_path, file_name)) as f:
                        for line in f:
                            if line.startswith('\n'):
                                continue
                            parts = line.split()
                            ground_true_data_dict = {'time': parts[0],
                                                     'qx': parts[1],
                                                     'qy': parts[2],
                                                     'qz': parts[3],
                                                     'qw': parts[4],
                                                     'x': parts[5],
                                                     'y': parts[6],
                                                     'z': parts[7]}
                            self.ground_true_data.append(ground_true_data_dict)
                elif file_name.endswith('sonar.txt'):
                    logger.info(f'Find {file_name}...')
                    with open(os.path.join(self.dataset_path, file_name)) as f:
                        for line in f:
                            if line.startswith('\n'):
                                continue
                            parts = line.split()
                            sonar_data_dict = {'time': parts[0],
                                               'sonar_image_path': parts[1]}
                            self.sonar_data.append(sonar_data_dict)
        all_acc = []  # ax, ay, az
        all_gyro = []
        for imu_data in self.imu_data:
            imu_list_ = [[float(imu_data['ax']), float(imu_data['ay']), float(imu_data['az']), float(imu_data['gx']),
                          float(imu_data['gy']), float(imu_data['gz'])]]
            imu_array = np.array(imu_list_)
            acc = imu_array[:, 0:3]  # ax, ay, az
            gyro = imu_array[:, 3:6]  # gx, gy, gz
            all_acc.append(acc)
            all_gyro.append(gyro)
        all_acc = np.concatenate(all_acc, axis=0)  # shape: (total_len, 3)
        all_gyro = np.concatenate(all_gyro, axis=0)  # shape: (total_len, 3)
        acc_mean = np.mean(all_acc, axis=0)
        acc_std = np.std(all_acc, axis=0)
        gyro_mean = np.mean(all_gyro, axis=0)
        gyro_std = np.std(all_gyro, axis=0)
        print("Customize acc mean (ax, ay, az):", acc_mean)
        print("Customize acc std", acc_std)
        print("Customize gyro mean (gx, gy, gz):", gyro_mean)
        print("Customize gyro std", gyro_std)

    def read_stpere_dataset(self):
        if os.path.exists(self.dataset_path):
            logger.info(f'Find {self.dataset_name} dataset path: {self.dataset_path}...')
            for file_name in os.listdir(self.dataset_path):
                if file_name.endswith('MTi.log'):
                    logger.info(f'Find {file_name}...')
                    with open(os.path.join(self.dataset_path, file_name)) as f:
                        for line in f:
                            if line.startswith('%') or line.startswith('\n'):
                                continue
                            parts = line.split()
                            imu_data_dict = {'time': parts[0],
                                             'roll': parts[1],
                                             'pitch': parts[2],
                                             'yaw': parts[3],
                                             'gx': parts[4],
                                             'gy': parts[5],
                                             'gz': parts[6],
                                             'ax': parts[7],
                                             'ay': parts[8],
                                             'az': parts[9]}
                            self.imu_data.append(imu_data_dict)
                elif file_name.endswith('IS.log'):
                    logger.info(f'Find {file_name}...')
                    with open(os.path.join(self.dataset_path, file_name)) as f:
                        for line in f:
                            if line.startswith('%') or line.startswith('\n'):
                                continue
                            parts = line.split()
                            sonar_data_dict = {'time': parts[0],
                                               'sensor_time': parts[1],
                                               'transducerAngle': parts[2],
                                               'bins': parts[3:]}
                            self.sonar_data.append(sonar_data_dict)

                elif file_name.endswith('DGPS.log'):
                    logger.info(f'Find {file_name}...')
                    with open(os.path.join(self.dataset_path, file_name), encoding='windows-1252') as f:
                        for line in f:
                            if line.startswith('%') or line.startswith('\n'):
                                continue
                            parts = line.split()
                            gps_data_dict = {'time': parts[0],
                                             'latitude': parts[1],
                                             'longitude': parts[2]}
                            self.ground_true_data.append(gps_data_dict)

        all_acc = []  # ax, ay, az
        all_gyro = []
        for imu_data in self.imu_data:
            imu_list_ = [[float(imu_data['ax']), float(imu_data['ay']), float(imu_data['az']), float(imu_data['gx']),
                          float(imu_data['gy']), float(imu_data['gz'])]]
            imu_array = np.array(imu_list_)
            acc = imu_array[:, 0:3]  # ax, ay, az
            gyro = imu_array[:, 3:6]  # gx, gy, gz
            all_acc.append(acc)
            all_gyro.append(gyro)
        all_acc = np.concatenate(all_acc, axis=0)  # shape: (total_len, 3)
        all_gyro = np.concatenate(all_gyro, axis=0)  # shape: (total_len, 3)
        acc_mean = np.mean(all_acc, axis=0)
        acc_std = np.std(all_acc, axis=0)
        gyro_mean = np.mean(all_gyro, axis=0)
        gyro_std = np.std(all_gyro, axis=0)
        print("Stpere acc mean (ax, ay, az):", acc_mean)
        print("Stpere acc std", acc_std)
        print("Stpere gyro mean (gx, gy, gz):", gyro_mean)
        print("Stpere gyro std:", gyro_std)

    def split_train_val_dataset(self, ratio=0.8):
        sonar_timestamps = [data['time'] for data in self.sonar_data]
        while sonar_timestamps and (
                not (self.imu_data[0]['time'] <= sonar_timestamps[0]) or
                not (self.ground_true_data[0]['time'] <= sonar_timestamps[0])
        ):
            logger.warning("First sonar frame is outside IMU or Ground_True data range. Removing it.")
            sonar_timestamps.pop(0)
            self.sonar_data.pop(0)

        while sonar_timestamps and (
                not (sonar_timestamps[-1] <= self.imu_data[-1]['time']) or
                not (sonar_timestamps[-1] <= self.ground_true_data[-1]['time'])
        ):
            logger.warning("Last sonar frame is outside IMU or GPS data range. Removing it.")
            sonar_timestamps.pop(-1)
            self.sonar_data.pop(-1)

        imu_frame_times = [data['time'] for data in self.imu_data]
        closest_imu_index = max(
            [i for i in range(len(imu_frame_times)) if float(imu_frame_times[i]) <= float(sonar_timestamps[0])])
        self.imu_data = [self.imu_data[closest_imu_index]] + self.imu_data[closest_imu_index + 1:]

        ground_true_frame_times = [data['time'] for data in self.ground_true_data]
        ground_true_gps_index = max(
            [i for i in range(len(ground_true_frame_times)) if
             float(ground_true_frame_times[i]) <= float(sonar_timestamps[0])])
        self.ground_true_data = ([self.ground_true_data[ground_true_gps_index]] +
                                 self.ground_true_data[ground_true_gps_index + 1:])

        selected_frames = []
        all_frames = sonar_timestamps
        for i in range(0, len(sonar_timestamps), self.frame_interval - 1):
            selected_frames.append(sonar_timestamps[i])
        if self.frame_interval == 2:
            selected_frames.pop(-1)

        save_data_path_ = os.path.join(self.dataset_save_path, self.dataset_name)
        os.makedirs(save_data_path_, exist_ok=True)
        save_data_path = os.path.join(save_data_path_, f'{self.frame_interval}_frame_interval')
        if self.map_type is not None:
            save_data_path = os.path.join(save_data_path, self.map_type)
        os.makedirs(save_data_path, exist_ok=True)
        train_path = os.path.join(save_data_path, 'Train')
        os.makedirs(train_path, exist_ok=True)
        test_path = os.path.join(save_data_path, 'Test')
        os.makedirs(test_path, exist_ok=True)
        val_path = os.path.join(save_data_path, 'Val')
        os.makedirs(val_path, exist_ok=True)

        for i, frame_time in enumerate(selected_frames):
            imu_start_frame = self.find_closest_data(self.imu_data, frame_time)
            imu_end_frame = self.find_closest_data(self.imu_data,
                                                   selected_frames[i + 1] if i + 1 < len(selected_frames) else
                                                   sonar_timestamps[-1])
            sonar_start_frame = self.find_closest_data(self.sonar_data, frame_time)
            sonar_end_frame = self.find_closest_data(self.sonar_data,
                                                     selected_frames[i + 1] if i + 1 < len(selected_frames) else
                                                     sonar_timestamps[-1])
            ground_true_frame = self.find_closest_data(self.ground_true_data, frame_time)
            ground_true_end_frame = self.find_closest_data(self.ground_true_data,
                                                           selected_frames[i + 1] if i + 1 < len(selected_frames) else
                                                           sonar_timestamps[-1])

            imu_data_in_range = [imu_frame for imu_frame in self.imu_data if
                                 frame_time <= imu_frame['time'] <= imu_end_frame['time']]

            sonar_data_in_range = [sonar_frame for sonar_frame in self.sonar_data if
                                   frame_time <= sonar_frame['time'] <= sonar_end_frame['time']]
            ground_true_data_in_range = [ground_true_frame for ground_true_frame in self.ground_true_data if
                                         frame_time <= ground_true_frame['time'] <= ground_true_end_frame['time']]

            save_data_dict = {'imu_range': imu_data_in_range,
                              'sonar_range': sonar_data_in_range,
                              'ground_true_range': ground_true_data_in_range}
            self.save_data_list.append(save_data_dict)

        if self.dataset_name == 'CustomizeDataset':
            # random.shuffle(self.save_data_list)
            total_count = len(self.save_data_list)
            train_split = int(total_count * ratio)  # 80% 
            test_split = int(total_count * ((1 - ratio) / 2 + ratio))  # 10% 
            train_data = self.save_data_list[:train_split]
            test_data = self.save_data_list[train_split:test_split]
            val_data = self.save_data_list[test_split:]

            self.save_list_data_to_txt(train_data, train_path)
            self.save_list_data_to_txt(test_data, test_path)
            self.save_list_data_to_txt(val_data, val_path)
            logger.info(f'🚀 {self.dataset_name}: imu, sonar, ground_true data saved to txt files successfully...')

        if self.dataset_name == 'StPereDataset':
            grouped_data = [self.save_data_list[i:i + 2] for i in range(len(self.save_data_list) - 1)]
            # random.shuffle(grouped_data)
            total_count = len(grouped_data)
            train_split = int(total_count * ratio)  # 80% 
            test_split = int(total_count * ((1 - ratio) / 2 + ratio))  # 10% 
            train_data = grouped_data[:train_split]
            test_data = grouped_data[train_split:test_split]
            val_data = grouped_data[test_split:]

            self.save_list_stpere_data_to_txt(train_data, train_path)
            self.save_list_stpere_data_to_txt(test_data, test_path)
            self.save_list_stpere_data_to_txt(val_data, val_path)
            logger.info(f'🚀 {self.dataset_name}: imu, sonar, ground_true data saved to txt files successfully...')

    @staticmethod
    def save_list_stpere_data_to_txt(data_list: list, save_path: str):
        for i, data in enumerate(data_list):
            save_file_path = os.path.join(save_path, f'{i}')
            os.makedirs(save_file_path, exist_ok=True)

            imu_file_path = os.path.join(save_file_path, 'imu.txt')
            sonar_file_path = os.path.join(save_file_path, 'sonar.txt')
            ground_true_file_path = os.path.join(save_file_path, 'ground_truth.txt')

            sonar_file_path_older = os.path.join(save_file_path, 'sonar_older.txt')

            with open(imu_file_path, 'w') as f:
                for imu_data in data[1]['imu_range']:
                    f.write(f'{imu_data}' + '\n')
            with open(sonar_file_path, 'w') as f:
                for sonar_data in data[1]['sonar_range']:
                    f.write(f'{sonar_data}' + '\n')
            with open(sonar_file_path_older, 'w') as f:
                for sonar_data in data[0]['sonar_range']:
                    f.write(f'{sonar_data}' + '\n')
            with open(ground_true_file_path, 'w') as f:
                for ground_true_data in data[1]['ground_true_range']:
                    f.write(f'{ground_true_data}' + '\n')

    @staticmethod
    def save_list_data_to_txt(data_list: list, save_path: str):
        for i, data in enumerate(data_list):
            save_file_path = os.path.join(save_path, f'{i}')
            os.makedirs(save_file_path, exist_ok=True)
            imu_file_path = os.path.join(save_file_path, 'imu.txt')
            sonar_file_path = os.path.join(save_file_path, 'sonar.txt')
            ground_true_file_path = os.path.join(save_file_path, 'ground_truth.txt')
            with open(imu_file_path, 'w') as f:
                for imu_data in data['imu_range']:
                    f.write(f'{imu_data}' + '\n')
            with open(sonar_file_path, 'w') as f:
                for sonar_data in data['sonar_range']:
                    f.write(f'{sonar_data}' + '\n')
            with open(ground_true_file_path, 'w') as f:
                for ground_true_data in data['ground_true_range']:
                    f.write(f'{ground_true_data}' + '\n')

    @staticmethod
    def find_closest_data(data_list, target_time):
        target_time = float(target_time)

        sorted_data = sorted(data_list, key=lambda x: float(x['time']))
        timestamps = [float(item['time']) for item in sorted_data]

        idx = bisect_left(timestamps, target_time)

        if idx == 0:
            return sorted_data[0]
        if idx == len(timestamps):
            return sorted_data[-1]

        before = sorted_data[idx - 1]
        after = sorted_data[idx]

        return before if abs(float(before['time']) - target_time) < abs(float(after['time']) - target_time) else after


if __name__ == '__main__':
    # StPere_dataset = Split_Datasets('StPereDataset',
    #                                 '../Datasets/StPereDataset',
    #                                 False, None, '../Split_Datasets', map_type=None)
    # StPere_dataset.detect_data_type()
    # StPere_dataset = Split_Datasets('CustomizeDataset',
    #                                     '../Datasets/CustomizeDataset',
    #                                     False, None, '../Split_Datasets', map_type='map1')
    # StPere_dataset.detect_data_type()

    # Split StPereDataset
    for i in range(180, 190):
        StPere_dataset = Split_Datasets('StPereDataset',
                                        '../Datasets/StPereDataset',
                                        False, i, '../Split_Datasets', map_type=None)
        StPere_dataset.detect_data_type()
        StPere_dataset.split_train_val_dataset(ratio=0.8)

    # for i in range(2, 7):
    #     StPere_dataset = Split_Datasets('CustomizeDataset',
    #                                     '../Datasets/CustomizeDataset',
    #                                     False, i, '../Split_Datasets', map_type='map3')
    #     StPere_dataset.detect_data_type()
    #     StPere_dataset.split_train_val_dataset(ratio=0.8)
