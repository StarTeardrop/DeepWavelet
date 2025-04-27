import numpy as np
from scipy.spatial.transform import Rotation as R


def imu_pre_integrator(imu_list):
    position = np.zeros(3) 
    velocity = np.zeros(3)  
    attitude = R.from_euler('xyz', [0, 0, 0])

    for i in range(1, len(imu_list)): 
        imu_data_prev = imu_list[i - 1]
        imu_data = imu_list[i]

        time, roll, pitch, yaw, ax, ay, az, gx, gy, gz = imu_data
        time_prev, roll_prev, pitch_prev, yaw_prev, ax_prev, ay_prev, az_prev, gx_prev, gy_prev, gz_prev = imu_data_prev

        dt = time - time_prev

        angular_velocity = 0.5 * (np.array([gx, gy, gz]) + np.array([gx_prev, gy_prev, gz_prev]))

        delta_rotation = R.from_rotvec(angular_velocity * dt)  
        attitude = attitude * delta_rotation  

        rotation_matrix = attitude.as_matrix()
        acc_world = rotation_matrix.dot(np.array([ax, ay, az]))  

        acc_world[2] += 9.81 

        acc_world_prev = rotation_matrix.dot(np.array([ax_prev, ay_prev, az_prev])) 
        acc_world_prev[2] += 9.81  
        acc_avg = 0.5 * (acc_world + acc_world_prev)

        velocity += acc_avg * dt 

        position += velocity * dt

    # print("Position: ", position)

    return position, velocity, attitude.as_euler('xyz', degrees=True)

