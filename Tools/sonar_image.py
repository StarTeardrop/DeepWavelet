import numpy as np
import cv2
from scipy.ndimage import uniform_filter
import open3d as o3d


def make_square_top(image):
    height, width = image.shape[:2]

    max_size = max(height, width)

    new_image = np.zeros((max_size, width, 3), dtype=np.uint8)

    top_padding = max_size - height

    new_image[top_padding:, :] = image

    return new_image

def generate_stpere_sonar_image(sonar_transform_list, img_size=(512, 512), max_radius=None, threshold=50):
    center = (img_size[0] // 2, img_size[1] // 2)
    if max_radius is None:
        max_radius = min(img_size) // 2
    image = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)

    all_echo_intensity = np.concatenate([data[2] for data in sonar_transform_list]).astype(np.float32)
    min_intensity = np.min(all_echo_intensity)
    max_intensity = np.max(all_echo_intensity)
    intensity_range = max_intensity - min_intensity if max_intensity != min_intensity else 1

    max_bin_len = max(len(data[2]) for data in sonar_transform_list)
    radius_array = np.linspace(0, max_radius, max_bin_len)

    for i in range(len(sonar_transform_list)):
        time, angle, bins = sonar_transform_list[i]
        bins = np.array(bins, dtype=float)
        bin_len = len(bins)

        if i > 0:
            _, prev_angle, prev_bins = sonar_transform_list[i - 1]
            angle_diff = angle - prev_angle
            num_steps = int(np.abs(angle_diff) // (np.pi / 180)) 

            if num_steps > 0:
                for j in range(1, num_steps + 1):
                    inter_angle = prev_angle + j * (angle_diff / (num_steps + 1))
                    inter_bins = prev_bins  

                    cos_val = np.cos(inter_angle)
                    sin_val = np.sin(inter_angle)
                    for k in range(len(inter_bins)):
                        radius = radius_array[k]
                        intensity = float(inter_bins[k])
                        norm = (intensity - min_intensity) / intensity_range
                        val = int(norm * 255)
                        color = (val, val, val)

                        x = int(center[0] + radius * cos_val)
                        y = int(center[1] + radius * sin_val)
                        if 0 <= x < img_size[0] and 0 <= y < img_size[1]:
                            image[y, x] = color

        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        for k in range(bin_len):
            radius = radius_array[k]
            intensity = bins[k]
            if intensity > threshold:
                color = (0, 255, 255)  # Yellow for obstacle
            else:
                color = (255, 0, 0)  # Blue for free space

            x = int(center[0] + radius * cos_val)
            y = int(center[1] + radius * sin_val)
            if 0 <= x < img_size[0] and 0 <= y < img_size[1]:
                image[y, x] = color

    return image


def stpere_soca_cfar(image_gray, guard_size=2, background_size=4, threshold_scale=1.4):
    img = image_gray.astype(np.float32)
    h, w = img.shape
    total_size = guard_size + background_size

    mean_kernel_size = 2 * total_size + 1
    background_mean = uniform_filter(img, size=mean_kernel_size)

    guard_kernel_size = 2 * guard_size + 1
    guard_mean = uniform_filter(img, size=guard_kernel_size)

    background_only_mean = (background_mean * (mean_kernel_size ** 2) -
                            guard_mean * (guard_kernel_size ** 2)) / \
                           ((mean_kernel_size ** 2) - (guard_kernel_size ** 2))

    threshold = background_only_mean * threshold_scale
    detection = img > threshold

    binary_output = detection.astype(np.uint8) * 255
    return binary_output


def generate_stpere_sonar_point_cloud(image, threshold=100):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    height, width = gray.shape

    scale_x = 50.0 / height
    scale_y = 50.0 / width 

    cx = width / 2
    cy = height / 2

    indices = np.argwhere(gray > threshold)

    point_cloud = []
    for y_img, x_img in indices:
        x = -(y_img - cy) * scale_x  
        y = (x_img - cx) * scale_y  
        point_cloud.append([x, y, 0.0])

    return np.array(point_cloud)


def generate_customize_sonar_point_cloud(image, threshold=100):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    height, width = gray.shape

    scale_x = 10.0 / height  
    scale_y = 18.0 / width  

    cx = width / 2
    cy = height / 2


    indices = np.argwhere(gray > threshold)

    point_cloud = []
    for y_img, x_img in indices:
        x = -(y_img - cy) * scale_x  
        y = (x_img - cx) * scale_y  
        point_cloud.append([x, y, 0.0])

    return np.array(point_cloud)

def show_point_cloud_open3d(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.tile(np.array([[0.2, 0.7, 1.0]]), (points.shape[0], 1)) 
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])
