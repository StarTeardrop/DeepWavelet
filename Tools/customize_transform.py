import torch
from torchvision import transforms
from Tools.point_cloud import Point_Cloud_Transform

class Value_To_Tensor:
    def __call__(self, x):
        return torch.tensor(x, dtype=torch.float32)

sonar_rgb_img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

value_general_transform = Value_To_Tensor()

point_cloud_transform = Point_Cloud_Transform(num_points=4096, voxel_size=0.01)


