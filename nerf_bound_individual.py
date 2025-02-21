import os
from typing import Optional, Tuple, List, Union, Callable

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange
import cv2 

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = os.path.dirname(os.path.realpath(__file__))
    
class TestModel(nn.Module):
    def __init__(self,height,width,focal):
        super(TestModel, self).__init__()
        self.height=height
        self.width=width
        self.focal=focal

    def get_rays(
        self, c2w: torch.Tensor
        ):

        height=self.height
        width=self.width
        focal_length=self.focal.to(c2w)

        r"""
        Find origin and direction of rays through every pixel and camera origin.
        """
        # print(c2w.shape)
        # Apply pinhole camera model to gather directions at each pixel
        i, j = torch.meshgrid(
            torch.arange(width, dtype=torch.float32).to(c2w),
            torch.arange(height, dtype=torch.float32).to(c2w),
            indexing="ij",
        )
        
        i, j = i.transpose(-1, -2), j.transpose(-1, -2)
        directions = torch.stack(
            [
                (i - width * 0.5) / focal_length,
                -(j - height * 0.5) / focal_length,
                -torch.ones_like(i),
            ],
            dim=-1,
        )

        # Apply camera pose to directions
        rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)

        # Origin is same for all directions (the optical center)
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        #return rays_o, rays_d
        return rays_o,rays_d
    
    def forward(self,x):
        ray_o, rays_d=self.get_rays(x)
        return ray_o

if __name__ == "__main__":
    data = np.load(os.path.join(script_dir,"tiny_nerf_data.npz"))
    images = data["images"]
    poses = data["poses"]
    focal = data["focal"]

    testimgidx = 13
    testimg = images[testimgidx]
    testpose = poses[testimgidx]

    cv2img = cv2.cvtColor(testimg, cv2.COLOR_RGB2BGR)
    testimg = cv2.cvtColor(cv2img, cv2.COLOR_RGB2BGR)

    testimg = torch.Tensor(testimg).to(device)
    testpose = torch.Tensor(testpose).to(device)
    height, width = testimg.shape[:2]
    #height,width=torch.Tensor(height).to(device),torch.Tensor(width).to(device)
    focal = torch.Tensor(focal).to(device)

    #rays_o, rays_d = get_rays(height, width, focal, testpose)
    #print(rays_o.shape)
    #print(rays_d.shape)

    ray_model=TestModel(height,width,focal)
    model = BoundedModule(ray_model, testpose)
    ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
    pose_input = BoundedTensor(testpose, ptb)

    lb, ub = model.compute_bounds(x=(pose_input,), method="backward")

    print("Lower bounds: ", lb)
    print("Upper bounds: ", ub)


    