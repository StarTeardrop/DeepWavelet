import torch
import numpy as np


class Point_Cloud_Transform:
    def __init__(self, num_points=4096, voxel_size=0.01):
        self.num_points = num_points
        self.voxel_size = voxel_size

    def __call__(self, point_cloud):
        point_cloud = self.center_point_cloud(point_cloud)
        point_cloud = self.voxel_downsampling(point_cloud)
        point_cloud = self.scale_point_cloud(point_cloud)
        point_cloud = self.random_sampling(point_cloud)
        return torch.tensor(point_cloud, dtype=torch.float32)

    def center_point_cloud(self, pc):
        centroid = np.mean(pc, axis=0)
        return pc - centroid

    def scale_point_cloud(self, pc):
        max_distance = np.max(np.linalg.norm(pc, axis=1))
        return pc / max_distance

    def random_sampling(self, pc):
        if pc.shape[0] < self.num_points:
            indices = np.random.choice(pc.shape[0], self.num_points, replace=True)
        else:
            indices = np.random.choice(pc.shape[0], self.num_points, replace=False)
        return pc[indices]

    def voxel_downsampling(self, pc):
        voxel_indices = np.floor(pc / self.voxel_size).astype(int)

        voxel_coords, indices = np.unique(voxel_indices, axis=0, return_inverse=True)

        downsampled_pc = []
        for i in range(voxel_coords.shape[0]):
            grid_points = pc[indices == i]
            centroid = np.mean(grid_points, axis=0)
            downsampled_pc.append(centroid)

        return np.array(downsampled_pc)
