import os
from typing import Optional, Tuple, List, Union, Callable
from tqdm import tqdm

import math

import time
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange
import cv2 
import onnx
import itertools

torch.set_default_dtype(torch.float64)
torch.seed()

# import warnings

# warnings.filterwarnings("ignore")
# os.environ["PYTHONWARNINGS"] = "ignore"


#from torch.profiler import profile, record_function, ProfilerActivity

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = os.path.dirname(os.path.realpath(__file__))
    
class NeRF(nn.Module):
    r"""
    Neural radiance fields module.
    """

    def __init__(
        self,
        d_input: int = 3,
        n_layers: int = 8,
        d_filter: int = 256,
        skip: Tuple[int] = (4,),
        d_viewdirs: Optional[int] = None,
    ):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = nn.functional.relu
        self.d_viewdirs = d_viewdirs

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)]
            + [
                (
                    nn.Linear(d_filter + self.d_input, d_filter)
                    if i in skip
                    else nn.Linear(d_filter, d_filter)
                )
                for i in range(n_layers - 1)
            ]
        )

        # Bottleneck layers
        if self.d_viewdirs is not None:
            # If using viewdirs, split alpha and RGB
            self.alpha_out = nn.Linear(d_filter, 1)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
            self.output = nn.Linear(d_filter // 2, 3)
        else:
            # If no viewdirs, use simpler output
            self.output = nn.Linear(d_filter, 4)

    def forward(
        self, x: torch.Tensor, viewdirs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Forward pass with optional view direction.
        """

        # Cannot use viewdirs if instantiated with d_viewdirs = None
        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError("Cannot input x_direction if d_viewdirs was not given.")

        # Apply forward pass up to bottleneck
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # Apply bottleneck
        if self.d_viewdirs is not None:
            # Split alpha from network output
            alpha = self.alpha_out(x)

            # Pass through bottleneck to get RGB
            x = self.rgb_filters(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.act(self.branch(x))
            x = self.output(x)

            # Concatenate alphas to output
            x = torch.concat([x, alpha], dim=-1)
        else:
            # Simple output
            x = self.output(x)
        return x

class PositionalEncoderEnv(nn.Module):
    r"""
    Sine-cosine positional encoder for environmental parameters (hue and temperature).
    """

    def __init__(self, d_input: int, n_freqs: int, log_space: bool = False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        # self.d_output = self.d_input
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.0 ** torch.linspace(0.0, self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(
                2.0**0.0, 2.0 ** (self.n_freqs - 1), self.n_freqs
            )

        self.register_buffer("freq_bands", freq_bands)
        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        x_times_freqs = x[..., None] * self.freq_bands
        sin_values = torch.sin(x_times_freqs)
        cos_values = torch.cos(x_times_freqs)

        # An additional dimension to separate sin and cos
        fn_x = torch.stack([sin_values, cos_values], dim=-1)
        fn_x = fn_x.reshape(*x_times_freqs.shape[:-1], -1)

        # Concatenate in the order of sin(x*f), cos(x*f), ...
        fn_x = fn_x.transpose(-1, -2).reshape(*x.shape[:-1], -1)
        return torch.concat([x, fn_x], dim=-1)


class PositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(self, d_input: int, n_freqs: int, log_space: bool = False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.0 ** torch.linspace(0.0, self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(
                2.0**0.0, 2.0 ** (self.n_freqs - 1), self.n_freqs
            )
        self.register_buffer("freq_bands", freq_bands)
        #self.freq_bands=freq_bands

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    # def forward(self, x) -> torch.Tensor:
    #     r"""
    #     Apply positional encoding to input.
    #     """
    #     return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)
    def forward(self, x) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        x_times_freqs = x[..., None] * self.freq_bands
        sin_values = torch.sin(x_times_freqs)
        cos_values = torch.cos(x_times_freqs)

        # An additional dimension to separate sin and cos
        fn_x = torch.stack([sin_values, cos_values], dim=-1)
        fn_x = fn_x.reshape(*x_times_freqs.shape[:-1], -1)

        # Concatenate in the order of sin(x*f), cos(x*f), ...
        fn_x = fn_x.transpose(-1, -2).reshape(*x.shape[:-1], -1)
        return torch.concat([x, fn_x], dim=-1)


class TestModel(nn.Module):
    def __init__(self,input_type, total_height,total_width,start_height,start_width,end_height,end_width,focal_x,focal_y,\
                        xyzrpy,env,near,far,distance_to_infinity,n_samples,perturb,inverse_depth,\
                        kwargs_sample_stratified,n_samples_hierarchical,kwargs_sample_hierarchical,chunksize,\
                        encode,encode_env,encode_viewdirs,coarse_model,fine_model,\
                        raw_noise_std=0.0, print_flag=False
                        ):
        super(TestModel, self).__init__()
        self.input_type=input_type
        self.total_height=total_height
        self.total_width=total_width
        self.start_height=start_height
        self.end_width=end_width
        self.end_height=end_height
        self.start_width=start_width
        self.focal_x=focal_x
        self.focal_y=focal_y

        self.init_xyzrpy(xyzrpy)
        self.init_env(env)
        self.near,self.far=near,far
        self.distance_to_infinity=distance_to_infinity
        self.n_samples=n_samples
        self.perturb=perturb
        self.inverse_depth=inverse_depth
        self.kwargs_sample_stratified={} if kwargs_sample_stratified is None else kwargs_sample_stratified
        self.n_samples_hierarchical=n_samples_hierarchical
        self.kwargs_sample_hierarchical={}  if kwargs_sample_hierarchical is None else kwargs_sample_hierarchical
        self.chunksize=chunksize
        self.t_rand=torch.rand([n_samples])

        self.encode=encode
        self.encode_env=encode_env
        self.encode_viewdirs=encode_viewdirs
        self.model=coarse_model
        self.fine_model=fine_model 

        self.raw_noise_std=raw_noise_std
        if (start_height is None) or (end_height is None) or (start_width is None) or (end_width is None):
            self.noise_rand=None
        else:
            self.noise_rand=torch.randn((end_height-start_height)*(end_width-start_width),n_samples) * raw_noise_std


        self.print_flag=print_flag

    def update_height_and_width(self,start_height,end_height,start_width,end_width):
        self.start_height=start_height
        self.end_height=end_height
        self.start_width=start_width
        self.end_width=end_width
        if (start_height is None) or (end_height is None) or (start_width is None) or (end_width is None):
            self.noise_rand=None
        else:
            self.noise_rand=torch.randn((end_height-start_height)*(end_width-start_width),n_samples) * raw_noise_std

    def get_extrinsic_matrix(self,xyzrpy):
        x=xyzrpy[:,0:1]
        y=xyzrpy[:,1:2]
        z=xyzrpy[:,2:3]
        gamma = xyzrpy[:,3:4]
        beta = xyzrpy[:,4:5]
        alpha = xyzrpy[:,5:6]

        R00 = torch.cos(alpha)*torch.cos(beta)
        R01 = torch.cos(alpha)*torch.sin(beta)*torch.sin(gamma)-torch.sin(alpha)*torch.cos(gamma)
        R02 = torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma)+torch.sin(alpha)*torch.sin(gamma)
        R03 = x

        R10 = torch.sin(alpha)*torch.cos(beta)
        R11 = torch.sin(alpha)*torch.sin(beta)*torch.sin(gamma)+torch.cos(alpha)*torch.cos(gamma)
        R12 = torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma)-torch.cos(alpha)*torch.sin(gamma)
        R13 = y

        R20 = -torch.sin(beta)
        R21 = torch.cos(beta)*torch.sin(gamma)
        R22 = torch.cos(beta)*torch.cos(gamma)
        R23 = z

        # Concatenate the rotation matrix components and translation
        R_row0 = torch.cat([R00, R01, R02, x], dim=1).unsqueeze(1)  # First row (unsqueeze to add extra dimension)
        R_row1 = torch.cat([R10, R11, R12, y], dim=1).unsqueeze(1)  # Second row
        R_row2 = torch.cat([R20, R21, R22, z], dim=1).unsqueeze(1)  # Third row
        R_row3 = torch.cat([torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x), torch.ones_like(x)], dim=1).unsqueeze(1)  # Fourth row

        # Use torch.cat to concatenate along the second dimension (row-wise)
        extrinsic_matrices = torch.cat([R_row0, R_row1, R_row2, R_row3], dim=1)

        return extrinsic_matrices

    def init_xyzrpy(self,xyzrpy):
        self.x=float(xyzrpy[0])
        self.y=float(xyzrpy[1])
        self.z=float(xyzrpy[2])
        self.roll=float(xyzrpy[3])
        self.pitch=float(xyzrpy[4])
        self.yaw=float(xyzrpy[5])

        self.dist_to_object=float(np.linalg.norm(xyzrpy[:2],ord=2))
        self.initial_angle=float(np.arctan2(xyzrpy[1], xyzrpy[0]))
        self.offset_yaw=float(xyzrpy[5])
        self.current_angle=self.initial_angle

    def init_env(self,env):
        self.hue=float(env[0])
        self.satur=float(env[1])

    def update_angle(self,angle):
        self.current_angle=float(angle + self.initial_angle)

    def generate_camera_positions_around_object_torch(self,angle):
        z=self.z*torch.ones_like(angle).to(angle.device)
        current_angle=self.current_angle

        # Calculate the new x and y coordinates based on the updated angle
        x = self.dist_to_object * torch.cos(angle + self.initial_angle).to(angle.device)
        y = self.dist_to_object * torch.sin(angle + self.initial_angle).to(angle.device)

        yaw=current_angle- self.initial_angle + self.offset_yaw
        yaw=yaw*torch.ones_like(angle).to(angle.device)
        pitch = torch.zeros_like(angle).to(angle.device)  # Keeping pitch at 0
        roll = self.roll*torch.ones_like(angle).to(angle.device)  # Roll is constant

        positions_xyzrpy = torch.cat([x, y, z, roll, pitch, yaw], dim=-1)
        return positions_xyzrpy
    
    def generate_xyzrpy_torch(self,input):
        if self.input_type=="x":
            x=input
            y=self.y*torch.ones_like(input).to(input.device)
            z=self.z*torch.ones_like(input).to(input.device)

            roll=self.roll*torch.ones_like(input).to(input.device)
            pitch=self.pitch*torch.ones_like(input).to(input.device)
            yaw=self.yaw*torch.ones_like(input).to(input.device)

            hue=self.hue*torch.ones_like(input).to(input.device)
            satur=self.satur*torch.ones_like(input).to(input.device)
        
        elif self.input_type=="y":
            x=self.x*torch.ones_like(input).to(input.device)
            y=input
            z=self.z*torch.ones_like(input).to(input.device)

            roll=self.roll*torch.ones_like(input).to(input.device)
            pitch=self.pitch*torch.ones_like(input).to(input.device)
            yaw=self.yaw*torch.ones_like(input).to(input.device)

            hue=self.hue*torch.ones_like(input).to(input.device)
            satur=self.satur*torch.ones_like(input).to(input.device)
        elif self.input_type=="z":
            x=self.x*torch.ones_like(input).to(input.device)
            y=self.y*torch.ones_like(input).to(input.device)
            z=input

            roll=self.roll*torch.ones_like(input).to(input.device)
            pitch=self.pitch*torch.ones_like(input).to(input.device)
            yaw=self.yaw*torch.ones_like(input).to(input.device)

            hue=self.hue*torch.ones_like(input).to(input.device)
            satur=self.satur*torch.ones_like(input).to(input.device)
        elif self.input_type=="roll":
            x=self.x*torch.ones_like(input).to(input.device)
            y=self.y*torch.ones_like(input).to(input.device)
            z=self.z*torch.ones_like(input).to(input.device)

            roll=input
            pitch=self.pitch*torch.ones_like(input).to(input.device)
            yaw=self.yaw*torch.ones_like(input).to(input.device)

            hue=self.hue*torch.ones_like(input).to(input.device)
            satur=self.satur*torch.ones_like(input).to(input.device)
        elif self.input_type=="yaw":
            x=self.x*torch.ones_like(input).to(input.device)
            y=self.y*torch.ones_like(input).to(input.device)
            z=self.z*torch.ones_like(input).to(input.device)

            roll=self.roll*torch.ones_like(input).to(input.device)
            pitch=self.pitch*torch.ones_like(input).to(input.device)
            yaw=input

            hue=self.hue*torch.ones_like(input).to(input.device)
            satur=self.satur*torch.ones_like(input).to(input.device)

            
        elif self.input_type=="ry":
            x=self.x*torch.ones_like(input[..., 0:1]).to(input.device)
            y=self.y*torch.ones_like(input[..., 0:1]).to(input.device)
            z=self.z*torch.ones_like(input[..., 0:1]).to(input.device)

            roll=input[..., 0:1]
            pitch=self.pitch*torch.ones_like(input[..., 0:1]).to(input.device)
            yaw=input[..., 1:2]

            hue=self.hue*torch.ones_like(input[..., 0:1]).to(input.device)
            satur=self.satur*torch.ones_like(input[..., 0:1]).to(input.device)

        elif self.input_type=="xyz":
            x=input[..., 0:1]
            y=input[..., 1:2]
            z=input[..., 2:3]

            roll=self.roll*torch.ones_like(input[..., 0:1]).to(input.device)
            pitch=self.pitch*torch.ones_like(input[..., 0:1]).to(input.device)
            yaw=self.yaw*torch.ones_like(input[..., 0:1]).to(input.device)

            hue=self.hue*torch.ones_like(input[..., 0:1]).to(input.device)
            satur=self.satur*torch.ones_like(input[..., 0:1]).to(input.device)

        elif self.input_type=="xyzry":
            x=input[..., 0:1]
            y=input[..., 1:2]
            z=input[..., 2:3]

            roll=input[..., 3:4]
            pitch=self.pitch*torch.ones_like(input[..., 0:1]).to(input.device)
            yaw=input[..., 4:5]

            hue=self.hue*torch.ones_like(input[..., 0:1]).to(input.device)
            satur=self.satur*torch.ones_like(input[..., 0:1]).to(input.device)

        elif self.input_type=="env":
            x=self.x*torch.ones_like(input[..., 0:1]).to(input.device)
            y=self.y*torch.ones_like(input[..., 0:1]).to(input.device)
            z=self.z*torch.ones_like(input[..., 0:1]).to(input.device)

            roll=self.roll*torch.ones_like(input[..., 0:1]).to(input.device)
            pitch=self.pitch*torch.ones_like(input[..., 0:1]).to(input.device)
            yaw=self.yaw*torch.ones_like(input[..., 0:1]).to(input.device)

            hue=input[..., 0:1]
            satur=input[..., 1:2]

        positions_xyzrpy = torch.cat([x, y, z, roll, pitch, yaw], dim=-1)
        env=torch.cat([hue,satur], dim=-1)
        return positions_xyzrpy,env
        

    def get_rays(
        self, c2w: torch.Tensor, directions: torch.Tensor
        ):

        # total_height=self.total_height
        # total_width=self.total_width
        # start_height=self.start_height
        # start_width=self.start_width
        # end_height=self.end_height
        # end_width=self.end_width
        # focal_x_length=self.focal_x.to(c2w)
        # focal_y_length=self.focal_y.to(c2w)

        # print(c2w.shape)
        # # Apply pinhole camera model to gather directions at each pixel
        # i, j = torch.meshgrid(
        #     torch.arange(start=start_width, end=end_width, dtype=torch.float32).to(c2w),
        #     torch.arange(start=start_height,end=end_height, dtype=torch.float32).to(c2w),
        #     indexing="ij",
        # )
        
        # i, j = i.transpose(-1, -2), j.transpose(-1, -2)
        # directions = torch.stack(
        #     [
        #         (i - total_width * 0.5) / focal_x_length,
        #         -(j - total_height * 0.5) / focal_y_length,
        #         -torch.ones_like(i),
        #     ],
        #     dim=-1,
        # )

        # # Apply camera pose to directions
        # #rays_d = torch.sum(directions[..., None, :] * c2w[...,:3, :3], dim=-1)
        # directions=directions.reshape([(end_height-start_height)*(end_width-start_width),3])
        rays_d = torch.sum(directions * c2w[...,:3, :3], dim=-1)
        #print('rays_d inside function:',rays_d.shape)

        # Origin is same for all directions (the optical center)
        #rays_o = c2w[:3, -1].expand(rays_d.shape)
        rays_o=c2w[...,:3,-1]
        #print('rays_o inside function:',rays_o.shape)

        return rays_o,rays_d
    
    def get_directions(self):
        r"""
        Find origin and direction of rays through every pixel and camera origin.
        """
        # Apply pinhole camera model to gather directions at each pixel
        total_height=self.total_height
        total_width=self.total_width
        start_height=self.start_height
        start_width=self.start_width
        end_height=self.end_height
        end_width=self.end_width
        focal_x_length=self.focal_x.to(device)
        focal_y_length=self.focal_y.to(device)
        i, j = torch.meshgrid(
            torch.arange(start=start_width, end=end_width, dtype=torch.float32).to(device),
            torch.arange(start=start_height,end=end_height, dtype=torch.float32).to(device),
            indexing="ij",
        )
        
        i, j = i.transpose(-1, -2), j.transpose(-1, -2)
        directions = torch.stack(
            [
                (i - total_width * 0.5) / focal_x_length,
                -(j - total_height * 0.5) / focal_y_length,
                -torch.ones_like(i),
            ],
            dim=-1,
        )

        # Apply camera pose to directions
        #rays_d = torch.sum(directions[..., None, :] * c2w[...,:3, :3], dim=-1)
        directions=directions.reshape([(end_height-start_height)*(end_width-start_width),3])
        return directions[..., None, :]
    
    def sample_stratified(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Sample along ray from regularly-spaced bins.
        """

        near,far=self.near,self.far
        distance_to_infinity=self.distance_to_infinity
        n_samples=self.n_samples
        perturb=self.perturb
        inverse_depth=self.inverse_depth

        # Grab samples for space integration along ray
        t_vals = torch.linspace(0.0, 1.0, n_samples, device=rays_o.device)
        if not inverse_depth:
            # Sample linearly between `near` and `far`
            z_vals = near * (1.0 - t_vals) + far * (t_vals)
        else:
            # Sample linearly in inverse depth (disparity)
            z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

        # Draw uniform samples from bins along ray
        if perturb:
            mids = 0.5 * (z_vals[1:] + z_vals[:-1])
            upper = torch.concat([mids, z_vals[-1:]], dim=-1)
            lower = torch.concat([z_vals[:1], mids], dim=-1)
            #t_rand = torch.rand([n_samples], device=z_vals.device)
            t_rand=self.t_rand.to(z_vals.device)
            z_vals = lower + (upper - lower) * t_rand

        dists_vals = z_vals[..., 1:]-z_vals[..., :-1]
        dists_vals = torch.cat([dists_vals, distance_to_infinity * torch.ones_like(dists_vals[..., :1])], dim=-1)
        dists_vals= dists_vals.repeat(*rays_o.shape[:-1],1)

        #z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])
        z_vals = z_vals.repeat(*rays_o.shape[:-1],1)
        

        # Apply scale from `rays_d` and offset from `rays_o` to samples
        # pts: (width, height, n_samples, 3)
        #print('shapes:',rays_o[..., None, :].shape,rays_d[..., None, :].shape,z_vals[..., :, None].shape)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        return pts, z_vals,dists_vals
    
    def two_norm(self,inputs: torch.Tensor, dim: int,keepdim: bool =False) -> torch.Tensor:
        squared = torch.square(inputs)  # Square the elements
        summed = torch.sum(squared, dim=dim, keepdim=keepdim)  # Sum along the specified dimension
        norm_manual = torch.sqrt(summed)  # Take the square root
        return norm_manual

    def get_chunks(self,inputs: torch.Tensor) -> List[torch.Tensor]:
        r"""
        Divide an input into chunks.
        """

        chunksize=self.chunksize
        n_samples=self.n_samples
        
        return [inputs[:,i : i + chunksize] for i in range(0, n_samples, chunksize)]
    
    def prepare_chunks(
        self,
        points: torch.Tensor,
        encoding_function: Callable[[torch.Tensor], torch.Tensor],
        env: torch.Tensor, 
        encoding_function_env:  Callable[[torch.Tensor], torch.Tensor],
    ) -> List[torch.Tensor]:
        r"""
        Encode and chunkify points to prepare for NeRF model.
        """
        
        chunksize=self.chunksize

        #print("shapes:",env.shape,points[..., :2].shape)
        #env = env[:, None, ...]+torch.zeros_like(points[..., :2])
        env = env[...,None,:].repeat([1,self.n_samples,1])
        #print("env.shape:",env.shape)
        #env=env[:, None, ...].repeat([1,self.n_samples,1])
        env=encoding_function_env(env)
        points = encoding_function(points)
        # print('points.shape:',points.shape)
        # print('env.shape:',env.shape)
        points = torch.cat((points, env), dim=-1)
        # print('points.shape:',points.shape)
        points = self.get_chunks(points)
        return points

    def prepare_viewdirs_chunks(
        self,
        rays_d: torch.Tensor,
        encoding_function: Callable[[torch.Tensor], torch.Tensor]
    ) -> List[torch.Tensor]:
        r"""
        Encode and chunkify viewdirs to prepare for NeRF model.
        """
        # Prepare the viewdirs
        chunksize=self.chunksize
        #print('norm.shape:',torch.norm(rays_d, dim=-1, keepdim=True).shape)
        #print('norm2.shape:',torch.norm(rays_d, dim=-1).unsqueeze(-1).shape)
        
        #print(rays_d.shape)
        norm_manual=self.two_norm(rays_d,dim=-1, keepdim=True)
        #print(norm_manual.shape)
        tmp = 1/norm_manual
        viewdirs = rays_d*tmp
        #print('part1:',viewdirs.shape)
        viewdirs = viewdirs[:, None, ...].repeat([1,self.n_samples,1])
        #print('part2:',viewdirs.shape)
        viewdirs = encoding_function(viewdirs)
        #print('part3:',viewdirs.shape)
        viewdirs = self.get_chunks(viewdirs)
        #print('part4:',viewdirs[0].shape)
        return viewdirs

    def cumprod_exclusive(self,tensor: torch.Tensor) -> torch.Tensor:

        transmittance=[]
        for i in range(self.n_samples):
            tmp=torch.ones_like(tensor[..., 0]).to(tensor.device)
            for j in range(i):
                tmp=tmp*tensor[..., j]
            transmittance.append(tmp.unsqueeze(1))
            #print('tmp:',tmp.shape)
        transmittance=torch.cat(transmittance, dim=1)

        return transmittance
    
    def get_rgb_map(self,alpha:torch.Tensor, rgb:torch.Tensor)-> torch.Tensor:
        # tmp =torch.zeros_like(alpha[...,None, 0]).to(alpha.device)
        tmp = 0.0
        # Compute alpha * rgb outside the loop
        alpha_rgb = alpha[..., None] * rgb
        # Compute 1 - alpha outside the loop
        one_minus_alpha = 1 - alpha
        for i in reversed(range(self.n_samples)):
            tmp = alpha_rgb[:, i, :] + one_minus_alpha[:, i:i+1] * tmp
            # Use ReLU to clamp the value
            # tmp = 1 - torch.relu(1 - tmp)
            # tmp=torch.relu(tmp)
        return tmp

    
    def get_depth_map(self,alpha:torch.Tensor, z_vals:torch.Tensor)-> torch.Tensor:
        depth_map=torch.zeros_like(alpha[..., 0]).to(alpha.device)
        for i in reversed(range(self.n_samples)):
            depth_map=alpha[..., i]*z_vals[...,i]+(1-alpha[..., i])*depth_map
        
        return depth_map  
    
    def get_acc_map(self,alpha:torch.Tensor)-> torch.Tensor:
        acc_map=torch.zeros_like(alpha[..., 0]).to(alpha.device)
        for i in reversed(range(self.n_samples)):
            acc_map=alpha[..., i]+(1-alpha[..., i])*acc_map
        
        return acc_map
    
    def raw2outputs(
        self,
        raw: torch.Tensor,
        z_vals: torch.Tensor,
        dists_vals: torch.Tensor,
        rays_d: torch.Tensor,
        white_bkgd: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Convert the raw NeRF output into RGB and other maps.
        """

        dists_vals = dists_vals * self.two_norm(rays_d[..., None, :],dim=-1)

        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.0

        alpha = 1.0 - torch.exp(-nn.functional.relu((raw[..., 3] + noise )* dists_vals))

        rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]

        # transmittance=alpha*self.cumprod_exclusive(1.0-alpha)
        # print('transmittance.shape:',transmittance.shape)  
        # rgb_map = torch.sum(transmittance[..., None] * rgb, dim=-2)  # [n_rays, 3]
        # print('rgb_map.shape:',rgb_map.shape)

        # alpha_rgb = torch.cat([alpha[..., None], rgb], dim=-1)
        # return alpha_rgb

        #weights=self.get_weights(alpha)
        rgb_map=self.get_rgb_map(alpha,rgb)
        
        return rgb_map
        depth_map = self.get_depth_map(alpha,z_vals)
        acc_map = self.get_acc_map(alpha)

        disp_map = 1.0 / torch.max(
            1e-10 * torch.ones_like(depth_map), depth_map / acc_map
        )
        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])


        return rgb_map,depth_map,acc_map,alpha

    def nerf_forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        rays_env: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        r"""
        Compute forward pass through model(s).
        """
        near,far=self.near,self.far
        encoding_fn=self.encode
        coarse_model=self.model
        kwargs_sample_stratified=self.kwargs_sample_stratified
        n_samples_hierarchical=self.n_samples_hierarchical
        kwargs_sample_hierarchical=self.kwargs_sample_hierarchical
        fine_model=self.fine_model
        viewdirs_encoding_fn=self.encode_viewdirs
        env_encoding_fn=self.encode_env

        # Sample query points along each ray.
        query_points, z_vals, dists_vals = self.sample_stratified(rays_o,rays_d)
        #outputs = {"z_vals_stratified": z_vals}

        # if self.print_flag:
        #     print('query_points.shape:',query_points.shape)
        #     print('z_vals.shape:',z_vals.shape)
        # print('query_points.shape:',query_points.shape)
        
        # Prepare batches.
        batches = self.prepare_chunks(query_points, encoding_fn,rays_env, env_encoding_fn)
        #print('batches_legnth:',len(batches))
        #print('batch.shape:',batches[0].shape)
        if viewdirs_encoding_fn is not None:
            batches_viewdirs = self.prepare_viewdirs_chunks(
                rays_d, viewdirs_encoding_fn
            )
            #print('batches_viewdirs_legnth:',len(batches_viewdirs))
            #print('batch_viewdirs.shape:',batches_viewdirs[0].shape)
        else:
            batches_viewdirs = [None] * len(batches)
            print('batches_viewdirs is in composition of None.')
        

        # Coarse model pass.
        # Split the encoded points into "chunks", run the model on all chunks, and
        # concatenate the results (to avoid out-of-memory issues).

        predictions = []
        for batch, batch_viewdirs in zip(batches, batches_viewdirs):
            predictions.append(coarse_model(batch, viewdirs=batch_viewdirs))

        raw = torch.cat(predictions, dim=1)
        if self.print_flag:
            print('raw.shape:',raw.shape)

        # print('shapes:',batches[-1][:,-1,:].shape,raw[..., 3].shape)
        #return batches[-2][:,-1,:]
        # Perform differentiable volume rendering to re-synthesize the RGB image.
        rgb_map=self.raw2outputs(raw, z_vals, dists_vals, rays_d)
        return rgb_map
    
        rgb_map,depth_map,acc_map,alpha= self.raw2outputs(raw, z_vals, dists_vals, rays_d)
        
        print('rgb_map.shape:',rgb_map.shape)
        print('depth_map.shape:',depth_map.shape)
        print('acc_map.shape:',acc_map.shape)
        print('alpha.shape:',alpha.shape)

        # Store outputs.
        outputs["rgb_map"] = rgb_map
        outputs["depth_map"] = depth_map
        outputs["acc_map"] = acc_map
        outputs["alpha"] = alpha
        return outputs
    
    def forward(self,x,directions):

        x,rays_env=self.generate_xyzrpy_torch(x)
        x=self.get_extrinsic_matrix(x)
        #print(rays_env.shape)

        rays_o, rays_d=self.get_rays(x, directions)
        # return rays_o
        if self.print_flag:
            print('rays_o.shape:',rays_o.shape)
            print('rays_d.shape:',rays_d.shape)

        #return self.nerf_forward(rays_o,rays_d)

        rgb_map=self.nerf_forward(rays_o,rays_d,rays_env)
        return  rgb_map
    
        outputs=self.nerf_forward(rays_o,rays_d)
        res=outputs["rgb_map"]
        
        return res
    
class RGBModel(nn.Module):
    def __init__(self, n_samples):
        super(RGBModel, self).__init__()
        self.n_samples = n_samples


    def get_rgb_map(self,alpha:torch.Tensor, rgb:torch.Tensor)-> torch.Tensor:
        # tmp =torch.zeros_like(alpha[...,None, 0]).to(alpha.device)
        tmp = 0.0
        # Compute alpha * rgb outside the loop
        alpha_rgb = alpha[..., None] * rgb
        # Compute 1 - alpha outside the loop
        one_minus_alpha = 1 - alpha
        for i in reversed(range(self.n_samples)):
            tmp = alpha_rgb[:, i, :] + one_minus_alpha[:, i:i+1] * tmp
            # Use ReLU to clamp the value
            # tmp = 1 - torch.relu(1 - tmp)
            #tmp=torch.relu(tmp)

        return tmp


    def forward(self, alpha_rgb):
        alpha, rgb = alpha_rgb[..., 0], alpha_rgb[..., 1:]
        return self.get_rgb_map(alpha, rgb)
    
def extrinsic_matrix_to_xyzrpy(T):
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    R = T[:3, :3]

    def rotation_matrix_to_rpy(R):
        pitch = -np.arcsin(R[2, 0])
        if np.abs(np.cos(pitch)) > np.finfo(float).eps:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = 0
            yaw = np.arctan2(-R[0, 1], R[1, 1])
        return roll, pitch, yaw

    roll, pitch, yaw = rotation_matrix_to_rpy(R)
    return np.array([x, y, z, roll, pitch, yaw])

def compute_image_sampling(ray_model,directions,input,input_type,num_sampling,xyz_step, ry_step,hue_step,satur_step,\
                           start_height,end_height,start_width,end_width):
    
    inputpos= input.repeat((end_height-start_height)*(end_width-start_width),1)
    predicted_image=ray_model(inputpos, directions)

    image_ls,image_us=predicted_image,predicted_image

    for i in range(num_sampling):
        if input_type=="x" or input_type=="y" or input_type=="z" :
            random_tensor = xyz_step* (torch.rand(1, 1).to(input.device) * 2 - 1 )
        elif input_type=="roll" or input_type=="yaw":
            random_tensor = ry_step* (torch.rand(1, 1).to(input.device) * 2 - 1 )
        elif input_type=="ry":
            random_tensor = ry_step* (torch.rand(1, 2).to(input.device) * 2 - 1 )
        elif input_type=="xyz":
            random_tensor = xyz_step* (torch.rand(1, 3).to(input.device) * 2 - 1 )
        elif input_type=="xyzry":
            random_tensor_1 = xyz_step* (torch.rand(1, 3).to(input.device) * 2 - 1 )
            random_tensor_2 = ry_step* (torch.rand(1, 2).to(input.device) * 2 - 1 )
            random_tensor=torch.cat((random_tensor_1, random_tensor_2), dim=1)
        elif input_type=="env":
            random_tensor_1 = hue_step* (torch.rand(1, 1).to(input.device) * 2 - 1 )
            random_tensor_2 = satur_step* (torch.rand(1, 1).to(input.device) * 2 - 1 )
            random_tensor=torch.cat((random_tensor_1, random_tensor_2), dim=1)

        inputvec=input+random_tensor
        inputpos= inputvec.repeat((end_height-start_height)*(end_width-start_width),1)
        predicted_image=ray_model(inputpos, directions)

        image_ls=torch.min(image_ls,predicted_image)
        image_us=torch.max(image_us,predicted_image)

    image_ls=image_ls.reshape([end_height-start_height, end_width-start_width, 3]).detach().cpu().numpy()
    image_us=image_us.reshape([end_height-start_height, end_width-start_width, 3]).detach().cpu().numpy()

    if num_sampling!=0:
        del inputvec,random_tensor
    del inputpos,predicted_image
    
    return image_ls,image_us

def compute_image_bound(ray_model,input,input_type,num_sampling,xyz_step,ry_step,hue_step,satur_step,\
                        start_vis_height,end_vis_height,tile_height,start_vis_width,end_vis_width,tile_width,\
                        print_flag,visual_flag,device):
    image_lb=np.zeros((total_height,total_width,3))
    image_ub=np.zeros((total_height,total_width,3))
    image_ls=np.zeros((total_height,total_width,3))
    image_us=np.zeros((total_height,total_width,3))
    
    for start_height in range(start_vis_height,end_vis_height,tile_height):
        for start_width in range(start_vis_width,end_vis_width,tile_width):


            end_height=min(start_height+tile_height,end_vis_height)
            end_width=min(start_width+tile_width,end_vis_width)

            if input_type=="x" or input_type=="y" or input_type=="z" or input_type=="roll" or input_type=="yaw":
                dummy_inputpos = BoundedTensor(torch.rand(((end_height-start_height)*(end_width-start_width), 1), device=device))
            elif input_type=="ry":
                dummy_inputpos = BoundedTensor(torch.rand(((end_height-start_height)*(end_width-start_width), 2), device=device))
            elif input_type=="xyz":
                dummy_inputpos = BoundedTensor(torch.rand(((end_height-start_height)*(end_width-start_width), 3), device=device))
            elif input_type=="xyzry":
                dummy_inputpos = BoundedTensor(torch.rand(((end_height-start_height)*(end_width-start_width), 5), device=device))
            elif input_type=="env":
                dummy_inputpos = BoundedTensor(torch.rand(((end_height-start_height)*(end_width-start_width), 2), device=device))

            if print_flag:
                print('\n cur_height,cur_width:',start_height,start_width)
            #print('end_height,end_width:',end_height,end_width)
            #start_height,start_width=56,56
            #end_height,end_width=60,60
            
            ray_model.update_height_and_width(start_height,end_height,start_width,end_width)
            directions = ray_model.get_directions()
            # h_w = torch.tensor([start_height,end_height,start_width,end_width], device=device).unsqueeze(0)
            
            
            
            
            # exp=ray_model(inputpose, directions)

            if input_type=="x" or input_type=="y" or input_type=="z" or input_type=="xyz":
                ptb = PerturbationLpNorm(norm=np.inf, eps=xyz_step)
            elif input_type=="roll" or input_type=="yaw" or input_type=="ry":
                ptb = PerturbationLpNorm(norm=np.inf, eps=ry_step)
            elif input_type=="xyzry":
                eps_tensor=torch.tensor([[xyz_step,xyz_step,xyz_step,ry_step,ry_step]]).to(device)
                xl=(input-eps_tensor).repeat((end_height-start_height)*(end_width-start_width),1)
                xu=(input+eps_tensor).repeat((end_height-start_height)*(end_width-start_width),1)
                ptb = PerturbationLpNorm(x_L=xl,x_U=xu)
            elif input_type=="env":
                eps_tensor=torch.tensor([[hue_step,satur_step]]).to(device)
                xl=(input-eps_tensor).repeat((end_height-start_height)*(end_width-start_width),1)
                xu=(input+eps_tensor).repeat((end_height-start_height)*(end_width-start_width),1)
                ptb = PerturbationLpNorm(x_L=xl,x_U=xu)
            
            ls,us=compute_image_sampling(ray_model,directions,input,input_type,num_sampling,xyz_step, ry_step,hue_step,satur_step,\
                           start_height,end_height,start_width,end_width)
            

            inputpose= input.repeat((end_height-start_height)*(end_width-start_width),1)
            inputpose_ptb = BoundedTensor(inputpose, ptb)
            model = BoundedModule(ray_model, (dummy_inputpos, directions))
            # model.visualize('model_viewer')

            # print("computing ibp and crown")
            if print_flag:
                print("Start IBP")
            lb_ibp, ub_ibp = model.compute_bounds(x=(inputpose_ptb, directions), method="ibp")
            if print_flag:
                print("IBP finished")
            reference_interm_bounds = {}
            for node in model.nodes():
                if (node.perturbed
                    and isinstance(node.lower, torch.Tensor)
                    and isinstance(node.upper, torch.Tensor)):
                    reference_interm_bounds[node.name] = (node.lower, node.upper)
            if print_flag:
                print("Start forward")
            # backward_start_time = time.time()
            lb, ub = model.compute_bounds(
                x=(inputpose_ptb, directions),
                method="forward+backward",
                reference_bounds=reference_interm_bounds)
                
            # print("lb.shape:",lb.shape)
            if print_flag:
                print("Lower bounds: ", lb)
                print("Upper bounds: ", ub)

            lb=torch.clamp(lb,min=0,max=1)
            ub=torch.clamp(ub,min=0,max=1)

            # Establish the whole image by composing every tile
            if visual_flag:

                image_lb[start_height:end_height,start_width:end_width,:]=lb.reshape([end_height-start_height, end_width-start_width, 3]).detach().cpu().numpy()
                image_ub[start_height:end_height,start_width:end_width,:]=ub.reshape([end_height-start_height, end_width-start_width, 3]).detach().cpu().numpy()
                image_ls[start_height:end_height,start_width:end_width,:]=ls
                image_us[start_height:end_height,start_width:end_width,:]=us

            del ptb, inputpose_ptb,lb_ibp, ub_ibp,reference_interm_bounds, lb, ub,ls,us, model
            torch.cuda.empty_cache()

    return image_lb,image_ub,image_ls,image_us

# def save_data(save_path,images_lb,images_ub,images_noptb):
#     np.savez(save_path, images_lb=images_lb,images_ub=images_ub,images_noptb=images_noptb)

#     print(f"Data saved to {save_path}")

def save_data(save_path,images_lb,images_ub):
    np.savez(save_path, images_lb=images_lb,images_ub=images_ub)

    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    start_time=time.time()

    dataname='lego'
    n_samples = 32
    n_layers = 2
    d_filter = 128
    
    n_iters=100000
    chunksize = 2**5
    xyz_step,ry_step=0.0002,0.00010
    hue_step,satur_step=0.0002,0.0002
    xyz_eps,ry_eps=xyz_step*3,ry_step*3
    hue_eps,satur_eps=hue_step*50,satur_step*1
    input_type="env"
    num_sampling=3

    testimgidx = 13
    visual_flag=True#False#
    bound_method='forward'
    bound_whole_flag=True#False#
    xdown_factor,ydown_factor=8,8
    tile_height,tile_width=50, 50
    save_npz=False
    save_img=False

    images_lb=[]
    images_ub=[]
    images_ls=[]
    images_us=[]
    
    if input_type=="xyz" or input_type=="x" or input_type=="y" or input_type=="z":
        print("dataname,type,eps:",dataname,input_type,xyz_eps)
    elif input_type=="ry" or input_type=="roll" or  input_type=="yaw":
        print("dataname,type,eps:",dataname,input_type,ry_eps)
    elif input_type=="xyzry":
        print("dataname,type,eps:",dataname,input_type,xyz_eps,ry_eps)
    elif input_type=="env":
        print("dataname,type,eps:",dataname,input_type,hue_eps,satur_eps)

    datapath='data/'+dataname+'_env_data.npz'

    data = np.load(os.path.join(script_dir,datapath))
    images = data["images"]
    poses = data["poses"]
    focal = data["focal"]
    envs = data["env"]

    hue_min, hue_max = -30, 30
    sat_min, sat_max = -0.5, 0.5

    envs[:,0] = (envs[:,0]-hue_min)/(hue_max-hue_min)
    envs[:,1] = (envs[:,1]-sat_min)/(sat_max-sat_min)
    
    testimg = images[testimgidx]
    testpose = poses[testimgidx]
    #print(testpose.shape)

    cv2img = cv2.cvtColor(testimg, cv2.COLOR_RGB2BGR)
    testimg = cv2.cvtColor(cv2img, cv2.COLOR_RGB2BGR)
    testimg = torch.Tensor(testimg).to(device)
    testenv = envs[testimgidx]
    total_height, total_width = testimg.shape[:2]

    
    start_vis_height,end_vis_height=0,total_height//ydown_factor #0,0+tile_height*1#20,20+tile_height*1 #
    start_vis_width,end_vis_width=0,total_width//xdown_factor #20,20+tile_width*3 # 0,0+tile_width*3 #

    total_height, total_width=total_height//ydown_factor, total_width//xdown_factor
    start_vis_height_org,end_vis_height_org=start_vis_height*ydown_factor,end_vis_height*ydown_factor
    start_vis_width_org,end_vis_width_org=start_vis_width*xdown_factor,end_vis_width*xdown_factor


    xyzrpy_np = extrinsic_matrix_to_xyzrpy(testpose)
    xyzrpy=torch.Tensor(xyzrpy_np).to(device)
    extrinsic_matrix = torch.Tensor(testpose).to(device)
    env_np=testenv
    testenv=torch.Tensor(testenv).to(device)
    
    focal_x = torch.Tensor([focal/xdown_factor]).to(device)
    focal_y = torch.Tensor([focal/ydown_factor]).to(device)

    near, far = 2.0, 6.0
    distance_to_infinity=1e2
    perturb = False#True 
    inverse_depth = False
    kwargs_sample_stratified = {
        "n_samples": n_samples,
        "perturb": perturb,
        "inverse_depth": inverse_depth,
    }
    n_samples_hierarchical = 0
    kwargs_sample_hierarchical = {"perturb": perturb}
    

    d_input = 3
    env_input =2
    n_freqs = 10
    log_space = True
    n_freqs_views = 4

    
    skip = []

    raw_noise_std=0.0
    print_flag=False#True
    
    
    feature="env_"+str(dataname)+"_"+str(n_freqs)+"_"+str(n_freqs_views)+"_"+str(d_filter)+"_"+str(n_layers)+"_"+str(n_iters)

    encode = PositionalEncoder(d_input, n_freqs, log_space=log_space).to(device)
    #encode = lambda x: encoder(x)

    encode_env = PositionalEncoderEnv(env_input , n_freqs, log_space=log_space).to(device)
    
    encode_viewdirs = PositionalEncoder(d_input, n_freqs_views, log_space=log_space).to(device)
    #encode_viewdirs = lambda x: encoder_viewdirs(x)
    d_viewdirs = encode_viewdirs.d_output

    coarse_model = NeRF(
        encode.d_output+encode_env.d_output,
        n_layers=n_layers,
        d_filter=d_filter,
        skip=skip,
        d_viewdirs=d_viewdirs,
    )

    coarse_model.load_state_dict(torch.load(os.path.join(script_dir, 'pts/nerf-fine_'+feature+'.pt')))
    coarse_model.to(device)

    fine_model = NeRF(
        encode.d_output+encode_env.d_output,
        n_layers=n_layers,
        d_filter=d_filter,
        skip=skip,
        d_viewdirs=d_viewdirs,
    )
    fine_model.load_state_dict(torch.load(os.path.join(script_dir,'pts/nerf-fine_'+feature+'.pt')))
    fine_model.to(device)


    ray_model=TestModel(input_type,total_height,total_width,None,None,None,None,focal_x,focal_y,\
                        xyzrpy_np,env_np,near,far,distance_to_infinity,n_samples,perturb,inverse_depth,\
                        kwargs_sample_stratified,n_samples_hierarchical,kwargs_sample_hierarchical,chunksize,\
                        encode,encode_env,encode_viewdirs,coarse_model,fine_model,\
                        raw_noise_std,print_flag
                        ).to(device)
    
    # torch.onnx.export(ray_model,(dummy_inputpos,h_w),'onnx_net.onnx')
    
    
    x_start, x_end, x_step = xyzrpy_np[0], xyzrpy_np[0]+xyz_eps, xyz_step
    y_start, y_end, y_step = xyzrpy_np[1], xyzrpy_np[1]+xyz_eps, xyz_step
    z_start, z_end, z_step = xyzrpy_np[2], xyzrpy_np[2]+xyz_eps, xyz_step

    roll_start, roll_end, roll_step = xyzrpy_np[3], xyzrpy_np[3]+ry_eps, ry_step
    pitch_start, pitch_end, pitch_step = xyzrpy_np[4], xyzrpy_np[4]+ry_eps, ry_step
    yaw_start, yaw_end, yaw_step = xyzrpy_np[5], xyzrpy_np[5]+ry_eps, ry_step

    hue_start, hue_end, hue_step = env_np[0], env_np[0]+hue_eps, hue_step
    satur_start,  satur_end, satur_step = env_np[1], env_np[1]+satur_eps, satur_step


    x_vals = np.arange(x_start, x_end, x_step*2)
    y_vals = np.arange(y_start, y_end, y_step*2)
    z_vals = np.arange(z_start, z_end, z_step*2)

    roll_vals = np.arange(roll_start, roll_end, roll_step*2)
    pitch_vals = np.arange(pitch_start, pitch_end, pitch_step*2)
    yaw_vals = np.arange(yaw_start, yaw_end, yaw_step*2)

    hue_vals = np.arange(hue_start, hue_end, hue_step*2)
    satur_vals = np.arange(satur_start, satur_end, satur_step*2)

    if input_type in ["x", "y", "z","roll", "yaw"]:
        vals = {"x": x_vals, "y": y_vals, "z": z_vals, "roll":roll_vals, "yaw":yaw_vals}[input_type]  # Select the corresponding value list
        for cur_val in tqdm(vals):
            cur_val = float(cur_val)
            input_tensor = torch.tensor([cur_val]).to(device)

            image_lb, image_ub,image_ls,image_us = compute_image_bound(ray_model, input_tensor, input_type, num_sampling, xyz_step, ry_step,hue_step,satur_step,
                                                                start_vis_height, end_vis_height, tile_height,
                                                                start_vis_width, end_vis_width, tile_width, print_flag,
                                                                visual_flag, device)

            images_lb.append(image_lb)
            images_ub.append(image_ub)
            images_ls.append(image_ls)
            images_us.append(image_us)
    
    elif input_type=="ry":
        for cur_roll, cur_yaw in tqdm(itertools.product(roll_vals, yaw_vals)):
            cur_roll,cur_yaw=float(cur_roll),float(cur_yaw)
            input=torch.tensor([cur_roll,cur_yaw]).to(device)

            image_lb,image_ub,image_ls,image_us=compute_image_bound(ray_model,input,input_type,num_sampling, xyz_step,ry_step,hue_step,satur_step,\
                                start_vis_height,end_vis_height,tile_height,start_vis_width,end_vis_width,tile_width,\
                                print_flag,visual_flag,device)

            images_lb.append(image_lb)
            images_ub.append(image_ub)
            images_ls.append(image_ls)
            images_us.append(image_us)

    elif input_type=="xyz": 
        for cur_x, cur_y, cur_z in tqdm(itertools.product(x_vals, y_vals, z_vals)):
            cur_x, cur_y, cur_z=float(cur_x), float(cur_y), float(cur_z)
            input=torch.tensor([cur_x, cur_y, cur_z]).to(device)

            image_lb,image_ub,image_ls,image_us=compute_image_bound(ray_model,input,input_type,num_sampling, xyz_step,ry_step,hue_step,satur_step,\
                                start_vis_height,end_vis_height,tile_height,start_vis_width,end_vis_width,tile_width,\
                                print_flag,visual_flag,device)

            images_lb.append(image_lb)
            images_ub.append(image_ub)
            images_ls.append(image_ls)
            images_us.append(image_us)

    elif input_type=="xyzry":
        for cur_x, cur_y, cur_z,cur_roll, cur_yaw in tqdm(itertools.product(x_vals, y_vals, z_vals,roll_vals, yaw_vals)):
            cur_x, cur_y, cur_z=float(cur_x), float(cur_y), float(cur_z)
            cur_roll,cur_yaw=float(cur_roll),float(cur_yaw)
            input=torch.tensor([cur_x, cur_y, cur_z, cur_roll,cur_yaw]).to(device)
            
            image_lb,image_ub,image_ls,image_us=compute_image_bound(ray_model,input,input_type,num_sampling, xyz_step,ry_step,hue_step,satur_step,\
                                start_vis_height,end_vis_height,tile_height,start_vis_width,end_vis_width,tile_width,\
                                print_flag,visual_flag,device)

            images_lb.append(image_lb)
            images_ub.append(image_ub)
            images_ls.append(image_ls)
            images_us.append(image_us)

    elif input_type=="env":
        for cur_hue, cur_satur in tqdm(itertools.product(hue_vals, satur_vals)):
            cur_hue, cur_satur=float(cur_hue), float(cur_satur)
            input=torch.tensor([cur_hue, cur_satur]).to(device)
            
            image_lb,image_ub,image_ls,image_us=compute_image_bound(ray_model,input,input_type,num_sampling, xyz_step,ry_step,hue_step,satur_step,\
                                start_vis_height,end_vis_height,tile_height,start_vis_width,end_vis_width,tile_width,\
                                print_flag,visual_flag,device)

            images_lb.append(image_lb)
            images_ub.append(image_ub)
            images_ls.append(image_ls)
            images_us.append(image_us)

    images_lb=np.array(images_lb)
    images_ub=np.array(images_ub)
    images_ls=np.array(images_ls)
    images_us=np.array(images_us)

    output_image_lb=images_lb.min(axis=0)
    output_image_ub=images_ub.max(axis=0)
    output_image_ls=images_ls.min(axis=0)
    output_image_us=images_us.max(axis=0)

    image_bound_diff=output_image_ub-output_image_lb
    image_bound_diff_norm=np.linalg.norm(image_bound_diff,axis=2)
    image_sample_diff=output_image_us-output_image_ls
    image_sample_diff_norm=np.linalg.norm(image_sample_diff,axis=2)

    bound_MPG=np.mean(image_bound_diff_norm)
    bound_XPG=np.max(image_bound_diff_norm)
    sample_MPG=np.mean(image_sample_diff_norm)
    sample_XPG=np.max(image_sample_diff_norm)
    print('bound_MPG,bound_XPG:',bound_MPG,bound_XPG)
    print('sample_MPG,sample_XPG:',sample_MPG,sample_XPG)

    end_time=time.time()
    print('Running Time:',f"{(end_time-start_time)/60:.2f}",' min')


    if save_npz:
        if input_type=="xyz" or input_type=="x" or input_type=="y" or input_type=="z":
            output_feature="env_"+str(dataname)+"_error_"+str(xyz_eps).split(".")[1]+input_type
        elif input_type=="ry" or input_type=="roll" or  input_type=="yaw":
            output_feature="env_"+str(dataname)+"_error_"+str(ry_eps).split(".")[1]+input_type
        elif input_type=="xyzry":
            output_feature="env_"+str(dataname)+"_error_"+str(xyz_eps).split(".")[1]+"_"+str(xyz_eps).split(".")[1]+"_"+input_type
        elif input_type=="env":
            output_feature="env_"+str(dataname)+"_error_"+str(hue_eps).split(".")[1]+"_"+str(satur_eps).split(".")[1]+"_"+input_type
        
        save_output_path='image_bound/'+output_feature+'.npz'
        save_data(save_output_path,output_image_lb,output_image_ub)
    
    if visual_flag:
        fig, ax = plt.subplots(
            1,  4, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1, 1, 1]}
        )
        ax[0].imshow(
            output_image_lb
        )
        ax[0].set_title(f"image_lb")
        ax[1].imshow(
            output_image_ub
        )
        ax[1].set_title(f"image_ub")
        ax[2].imshow(
            output_image_ls
        )
        ax[2].set_title(f"image_ls")
        ax[3].imshow(
            output_image_us
        )
        #ax[3].imshow(testimg[start_height:end_height,start_height:end_width,:].detach().cpu().numpy())
        ax[3].set_title(f"image_us")


        if save_img:
            # imagename="bound_img_"+str(dataname)+"_error_"+str(eps).split(".")[1]+"_method_"+str(bound_method)+\
            #             "_anglestart_"+str(angle_start).split(".")[1]+"_angleend_"+str(angle_end).split(".")[1]+\
            #             "_features_"+str(n_freqs)+"_"+str(n_freqs_views)+"_"+str(d_filter)+"_"+str(n_layers)+"_"+str(n_iters)+\
            #             "_samples_"+str(n_samples)+"_inputdim_"+str(input_dim)+\
            #             "_xdown_"+str(xdown_factor)+"_ydown_"+str(ydown_factor)+"_whole.png"
            if input_type=="xyz" or input_type=="x" or input_type=="y" or input_type=="z":
                imagename="env_"+str(dataname)+"_error_"+str(xyz_eps).split(".")[1]+input_type+".png"
            elif input_type=="ry" or input_type=="roll" or  input_type=="yaw":
                imagename="env_"+str(dataname)+"_error_"+str(ry_eps).split(".")[1]+input_type+".png"
            elif input_type=="xyzry":
                imagename="env_"+str(dataname)+"_error_"+str(xyz_eps).split(".")[1]+"_"+str(xyz_eps).split(".")[1]+"_"+input_type+".png"
            elif input_type=="env":
                imagename="env_"+str(dataname)+"_error_"+str(hue_eps).split(".")[1]+"_"+str(satur_eps).split(".")[1]+"_"+input_type+".png"

            plt.savefig("output_img/"+imagename, bbox_inches='tight')

        plt.show()

    


    