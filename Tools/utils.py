import numpy as np
from scipy.spatial.transform import Rotation as R
from pyproj import Proj, Transformer


def stpere_rotation_imu(roll, pitch, yaw, ax, ay, az, gx, gy, gz):
    acc = np.array([ax, ay, az])  # ax, ay, az
    gyro = np.array([gx, gy, gz])  # gx, gy, gz
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    R_imu = r.as_matrix()
    R_change = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    R_new = R_change @ R_imu
    r_new = R.from_matrix(R_new)
    roll_new, pitch_new, yaw_new = r_new.as_euler('xyz', degrees=True)
    acc_new = R_new @ acc
    gyro_new = R_new @ gyro
    return roll_new, pitch_new, yaw_new, acc_new[0], acc_new[1], acc_new[2], gyro_new[0], gyro_new[1], gyro_new[2]


def stpere_gps_to_robot(delta_position_gps, yaw_degree):
    yaw = np.radians(90 + yaw_degree)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    R_yaw_inv = np.array([
        [cos_yaw, sin_yaw],
        [-sin_yaw, cos_yaw]
    ])

    delta_position_robot = R_yaw_inv @ delta_position_gps
    return delta_position_robot


def stpere_rotation_sonar(yaw):
    r = R.from_euler('xyz', [0, 0, yaw], degrees=False)
    R_sonar = r.as_matrix()
    R_change = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    R_new = R_change @ R_sonar
    r_new = R.from_matrix(R_new)
    _, _, yaw_new = r_new.as_euler('xyz', degrees=False)

    return yaw_new


def nmea_to_decimal(coord):
    degrees = int(coord) // 100
    minutes = coord - degrees * 100
    return degrees + minutes / 60


def stpere_gps_distance(lat1, lon1, lat2, lon2):
    lat1 = nmea_to_decimal(lat1)
    lon1 = nmea_to_decimal(lon1)
    lat2 = nmea_to_decimal(lat2)
    lon2 = nmea_to_decimal(lon2)
    proj_enu = Transformer.from_crs(
        crs_from="epsg:4326",  
        crs_to=f"+proj=tmerc +lat_0={lat1} +lon_0={lon1} +k=1 +x_0=0 +y_0=0 +ellps=WGS84",
        always_xy=True
    )
    x1, y1 = proj_enu.transform(lon1, lat1)
    x2, y2 = proj_enu.transform(lon2, lat2)

    dx = x2 - x1  
    dy = y2 - y1  
    return dx, dy


def process_imu_data(imu_list):
    imu_list = np.array(imu_list)
    imu_length = len(imu_list)

    if imu_length < 128:
        padding_size = 128 - imu_length
        padding_data = np.zeros((padding_size, len(imu_list[0]))) 
        imu_list = np.vstack([imu_list, padding_data])  
    elif imu_length > 128:
        indices = np.linspace(0, imu_length - 1, 128, dtype=int) 
        imu_list = imu_list[indices] 

    return imu_list


def imu_to_normal(imu_list, imu_acc_mean, imu_acc_std, imu_gyro_mean, imu_gyro_std):
    imu_acc = imu_list[:, 4:7]  
    imu_gyro = imu_list[:, 7:10] 
    imu_acc = (imu_acc - imu_acc_mean) / imu_acc_std 
    imu_gyro = (imu_gyro - imu_gyro_mean) / imu_gyro_std  
    imu_list[:, 4:7] = imu_acc  
    imu_list[:, 7:10] = imu_gyro 
    return imu_list


def ground_true_xy_normal(ground_true_xy, x_min, x_max, y_min, y_max):
    x = ground_true_xy[0]
    y = ground_true_xy[1]
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (y - y_min) / (y_max - y_min) - 1

    return x_norm, y_norm


def save_data_to_txt(pred_x, pred_y, pred_yaw_sin, pred_yaw_cos,
                     target_x, target_y, target_yaw_sin, target_yaw_cos,
                     time_cost,
                     file_name):
    with open(file_name, 'a') as f:
        f.write(f"{pred_x:.6f}, {pred_y:.6f}, {pred_yaw_sin:.6f}, {pred_yaw_cos:.6f}, "
                f"{target_x:.6f}, {target_y:.6f}, {target_yaw_sin:.6f}, {target_yaw_cos:.6f}, {time_cost:.6f}\n")


if __name__ == '__main__':
    imu_list = np.random.rand(100, 10)  
    processed_imu_list = process_imu_data(imu_list)
    print(processed_imu_list.shape)
