import os
from typing import Optional, Tuple, List, Union, Callable

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange
import cv2 

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

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)

def get_rays(
    height: int, width: int, focal_x_length: float, focal_y_length: float,c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
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
            (i - width * 0.5) / focal_x_length,
            -(j - height * 0.5) / focal_y_length,
            -torch.ones_like(i),
        ],
        dim=-1,
    )

    # Apply camera pose to directions
    rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)
    #print(rays_d.shape)

    # Origin is same for all directions (the optical center)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    #print(rays_o.shape)
    return rays_o, rays_d

def sample_stratified(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    perturb: Optional[bool] = True,
    inverse_depth: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Sample along ray from regularly-spaced bins.
    """

    # Grab samples for space integration along ray
    t_vals = torch.linspace(0.0, 1.0, n_samples, device=rays_o.device)
    if not inverse_depth:
        # Sample linearly between `near` and `far`
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity)
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

    #print('z_vals.shape:',z_vals.shape)
    # Draw uniform samples from bins along ray
    if perturb:
        mids = 0.5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.concat([mids, z_vals[-1:]], dim=-1)
        lower = torch.concat([z_vals[:1], mids], dim=-1)
        t_rand = torch.rand([n_samples], device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])
    #print('z_vals2.shape:',z_vals.shape)

    # Apply scale from `rays_d` and offset from `rays_o` to samples
    # pts: (width, height, n_samples, 3)

    #print('shapes:',rays_o[..., None, :].shape,rays_d[..., None, :].shape,z_vals[..., :, None].shape)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals

def get_chunks(inputs: torch.Tensor, chunksize: int = 2**15) -> List[torch.Tensor]:
    r"""
    Divide an input into chunks.
    """
    return [inputs[i : i + chunksize] for i in range(0, inputs.shape[0], chunksize)]



def prepare_chunks(
    points: torch.Tensor,
    encoding_function: Callable[[torch.Tensor], torch.Tensor],
    chunksize: int = 2**15,
) -> List[torch.Tensor]:
    r"""
    Encode and chunkify points to prepare for NeRF model.
    """
    points = points.reshape((-1, 3))
    points = encoding_function(points)
    points = get_chunks(points, chunksize=chunksize)
    return points

def prepare_viewdirs_chunks(
    points: torch.Tensor,
    rays_d: torch.Tensor,
    encoding_function: Callable[[torch.Tensor], torch.Tensor],
    chunksize: int = 2**15,
) -> List[torch.Tensor]:
    r"""
    Encode and chunkify viewdirs to prepare for NeRF model.
    """
    # Prepare the viewdirs
    #print('norm.shape:',torch.norm(rays_d, dim=-1, keepdim=True).shape)
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    #print('part1:',viewdirs.shape)
    viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
    #print('part2:',viewdirs.shape)
    viewdirs = encoding_function(viewdirs)
    #print('part3:',viewdirs.shape)
    viewdirs = get_chunks(viewdirs, chunksize=chunksize)
    #print('part4:',viewdirs[0].shape)
    return viewdirs

def sample_pdf(
    bins: torch.Tensor, weights: torch.Tensor, n_samples: int, perturb: bool = False
) -> torch.Tensor:
    r"""
    Apply inverse transform sampling to a weighted set of points.
    """

    # Normalize weights to get PDF.
    pdf = (weights + 1e-5) / torch.sum(
        weights + 1e-5, -1, keepdims=True
    )  # [n_rays, weights.shape[-1]]

    # Convert PDF to CDF.
    cdf = torch.cumsum(pdf, dim=-1)  # [n_rays, weights.shape[-1]]
    cdf = torch.concat(
        [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
    )  # [n_rays, weights.shape[-1] + 1]

    # Take sample positions to grab from CDF. Linear when perturb == 0.
    if not perturb:
        u = torch.linspace(0.0, 1.0, n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])  # [n_rays, n_samples]
    else:
        u = torch.rand(
            list(cdf.shape[:-1]) + [n_samples], device=cdf.device
        )  # [n_rays, n_samples]

    # Find indices along CDF where values in u would be placed.
    u = u.contiguous()  # Returns contiguous tensor with same values.
    inds = torch.searchsorted(cdf, u, right=True)  # [n_rays, n_samples]

    # Clamp indices that are out of bounds.
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)  # [n_rays, n_samples, 2]

    # Sample from cdf and the corresponding bin centers.
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g)
    bins_g = torch.gather(
        bins.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g
    )

    # Convert samples to ray length.
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples  # [n_rays, n_samples]

def sample_hierarchical(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    z_vals: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    perturb: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Apply hierarchical sampling to the rays.
    """

    # Draw samples from PDF using z_vals as bins and weights as probabilities.
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    # print('shapes2:',z_vals_mid.shape,weights.shape)
    new_z_samples = sample_pdf(
        z_vals_mid, weights[..., 1:-1], n_samples, perturb=perturb
    )
    new_z_samples = new_z_samples.detach()

    # Resample points from ray based on PDF.
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    pts = (
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
    )  # [N_rays, N_samples + n_samples, 3]
    return pts, z_vals_combined, new_z_samples

def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""
    (Courtesy of https://github.com/krrish94/nerf-pytorch)

    Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
        is to be computed.
    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
        tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """

    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    #print('input.shape:',tensor.shape)
    cumprod = torch.cumprod(tensor, -1)
    #print('part1:',cumprod.shape)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, -1)
    #print('part2:',cumprod.shape)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod

def raw2outputs(
    raw: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    raw_noise_std: float = 0.0,
    white_bkgd: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Convert the raw NeRF output into RGB and other maps.
    """

    # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point. [n_rays, n_samples]
    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * dists)
    #print('alpha.shape:',alpha.shape)

    # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
    # The higher the alpha, the lower subsequent weights are driven.
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    # Compute weighted RGB map.
    rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

    # Estimated depth map is predicted distance.
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # Disparity map is inverse depth.
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
    )

    # Sum of weights along each ray. In [0, 1] up to numerical error.
    acc_map = torch.sum(weights, dim=-1)

    # To composite onto a white background, use the accumulated alpha map.
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    #print('weights.shape:',weights.shape)
    #print('shapes:',rgb_map.shape,depth_map.shape,acc_map.shape,weights.shape)
    return rgb_map, depth_map, acc_map, weights

def nerf_forward(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    encoding_fn: Callable[[torch.Tensor], torch.Tensor],
    coarse_model: nn.Module,
    kwargs_sample_stratified: dict = None,
    n_samples_hierarchical: int = 0,
    kwargs_sample_hierarchical: dict = None,
    fine_model=None,
    viewdirs_encoding_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    chunksize: int = 2**15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    r"""
    Compute forward pass through model(s).
    """

    # Set no kwargs if none are given.
    if kwargs_sample_stratified is None:
        kwargs_sample_stratified = {}
    if kwargs_sample_hierarchical is None:
        kwargs_sample_hierarchical = {}

    # Sample query points along each ray.
    query_points, z_vals = sample_stratified(
        rays_o, rays_d, near, far, **kwargs_sample_stratified
    )
    #print('query_points.shape:',query_points.shape)
    #print('z_vals.shape:',z_vals.shape)

    # Prepare batches.
    batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
    #print('batches')
    if viewdirs_encoding_fn is not None:
        batches_viewdirs = prepare_viewdirs_chunks(
            query_points, rays_d, viewdirs_encoding_fn, chunksize=chunksize
        )
    else:
        batches_viewdirs = [None] * len(batches)

    # Coarse model pass.
    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        predictions.append(coarse_model(batch, viewdirs=batch_viewdirs))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d)
    # rgb_map, depth_map, acc_map, weights = render_volume_density(raw, rays_o, z_vals)
    outputs = {"z_vals_stratified": z_vals}

    # Fine model pass.
    if n_samples_hierarchical > 0:
        # Save previous outputs to return.
        rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

        # Apply hierarchical sampling for fine query points.
        query_points, z_vals_combined, z_hierarch = sample_hierarchical(
            rays_o,
            rays_d,
            z_vals,
            weights,
            n_samples_hierarchical,
            **kwargs_sample_hierarchical,
        )

        # Prepare inputs as before.
        batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
        if viewdirs_encoding_fn is not None:
            batches_viewdirs = prepare_viewdirs_chunks(
                query_points, rays_d, viewdirs_encoding_fn, chunksize=chunksize
            )
        else:
            batches_viewdirs = [None] * len(batches)

        # Forward pass new samples through fine model.
        fine_model = fine_model if fine_model is not None else coarse_model
        predictions = []
        for batch, batch_viewdirs in zip(batches, batches_viewdirs):
            predictions.append(fine_model(batch, viewdirs=batch_viewdirs))
        raw = torch.cat(predictions, dim=0)
        raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals_combined, rays_d)

        # Store outputs.
        outputs["z_vals_hierarchical"] = z_hierarch
        outputs["rgb_map_0"] = rgb_map_0
        outputs["depth_map_0"] = depth_map_0
        outputs["acc_map_0"] = acc_map_0

    # Store outputs.
    outputs["rgb_map"] = rgb_map
    outputs["depth_map"] = depth_map
    outputs["acc_map"] = acc_map
    outputs["weights"] = weights
    return outputs

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

def rotation_matrix_from_rpy(roll, pitch, yaw):
    """
    Generate a rotation matrix from roll, pitch, and yaw angles.
    """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def xyzrpy_to_extrinsic_matrix(xyzrpy):
    [x, y, z, roll, pitch, yaw]=xyzrpy[:]
    """
    Convert position (x, y, z) and orientation (roll, pitch, yaw) to a 4x4 extrinsic matrix.
    """
    R = rotation_matrix_from_rpy(roll, pitch, yaw)
    T = np.eye(4)
    T[:3, :3] = R
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T

def generate_camera_positions_around_object(
    xyzrpy, num_pos
):
    z = xyzrpy[2]
    
    # Distance from the given camera position to the intersection point
    initial_pos = np.array(xyzrpy[:3])
    dist_to_object = np.linalg.norm(initial_pos[:2])
    angles = np.linspace(0, 2 * np.pi, num_pos, endpoint=False)
    
    positions_xyzrpy = [xyzrpy]  # Start with the given input position
    initial_angle = np.arctan2(initial_pos[1] , initial_pos[0] )
    initial_yaw=np.arctan2(-initial_pos[1], -initial_pos[0])
    
    for i, angle in enumerate(angles[1:], 1):  # Skip the first position (it's already added)
        # Adjust each angle relative to the initial angle
        current_angle = angle + initial_angle
        
        x =  dist_to_object * np.cos(current_angle)
        y =  dist_to_object * np.sin(current_angle)
        pos = np.array([x, y, z])
        
        direction =  - pos
        direction = direction / np.linalg.norm(direction)  
        
        # Compute yaw (rotation around z-axis) from the direction vector
        yaw = np.arctan2(direction[1], direction[0])-initial_yaw+xyzrpy[5]
        
        # Compute pitch (rotation around y-axis)
        pitch = 0  # Negative to point downward toward object
        
        # Set roll to 0 (no roll for simple pointing toward the object)
        roll = xyzrpy[3]
        
        # Append the position and orientation (xyzrpy)
        positions_xyzrpy.append([x, y, z, roll, pitch, yaw])
    
    return positions_xyzrpy

   
if __name__ == "__main__":

    dataname='tinydozer'
    n_samples = 32
    n_layers = 2
    d_filter = 128
    n_iters=20000
    xdown_factor,ydown_factor=1,1
    chunksize = 2**12
    testimgidx = 13
    num_pos = 20

    print("dataname,n_iter:",dataname,n_iters)

    datapath='data/'+dataname+'_data.npz'

    data = np.load(os.path.join(script_dir,datapath))
    images = data["images"]
    poses = data["poses"]
    focal = data["focal"]

    testimg = images[testimgidx]
    testpose = poses[testimgidx]

    #print("testpose:",testpose)

    xyzrpy=extrinsic_matrix_to_xyzrpy(testpose)
    print('old_xyzrpy:',xyzrpy)
    #xyzrpy[3]=1.2*xyzrpy[3]
    #xyzrpy[3]=0.5*xyzrpy[3]
    # xyzrpy=np.array([
    #     3.0239111610258766, -2.56887030318416,  0.71194667, 1.3932528, 0, 0.868
    # ])
    # xyzrpy=np.array([
    #     3.0239111610258766, -2.56887030318416, 0.71194667, 1.3932528, 0, 0.8665815886230233
    # ])
    # print('new_xyzrpy:',xyzrpy)

    positions_xyzrpy = generate_camera_positions_around_object(
    xyzrpy, num_pos
    )
    #print("returned_testpose:",returned_testpose)

    # hue = np.random.uniform(-30,30)
    # sat = np.random.uniform(-0.5,0.5)
    #hue = 30
    #sat = -0.5

    #print(f">>> Test: idx {testimgidx}; hue {hue}; sat {sat}")

    cv2img = cv2.cvtColor(testimg, cv2.COLOR_RGB2BGR)
    #cv2img = adjust_hue(cv2img, hue)
    #cv2img = adjust_saturation(cv2img, sat)
    testimg = cv2.cvtColor(cv2img, cv2.COLOR_RGB2BGR)

    testimg = torch.Tensor(testimg).to(device)
    focal = torch.Tensor(focal).to(device)
    #testenv = torch.Tensor([hue, sat]).to(device)

    #hue_min, hue_max = -30, 30
    #sat_min, sat_max = -0.5, 0.5

    #testenv[0] = (testenv[0]-hue_min)/(hue_max-hue_min)
    #testenv[1] = (testenv[1]-sat_min)/(sat_max-sat_min)

    print("height,width:",testimg.shape[0],testimg.shape[1])


    for i, pos in enumerate(positions_xyzrpy):
        print(f"Camera {i+1}: {pos}")
        testpose=xyzrpy_to_extrinsic_matrix(pos)
        testpose = torch.Tensor(testpose).to(device)
        
        height, width = testimg.shape[0]//ydown_factor,testimg.shape[1]//xdown_factor
        focal_x,focal_y=focal//xdown_factor,focal//ydown_factor

        rays_o, rays_d = get_rays(height, width, focal_x, focal_y,testpose)
        rays_o = rays_o.reshape([-1, 3])
        rays_d = rays_d.reshape([-1, 3])
        #print(rays_o.shape)
        #print(rays_d.shape)

        #rays_env = testenv.repeat(rays_o.shape[0],1)

        near, far = 2.0, 6.0
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
        n_freqs = 10
        log_space = True
        n_freqs_views = 4

        encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)
        encode = lambda x: encoder(x)

        encoder_viewdirs = PositionalEncoder(
            d_input, n_freqs_views, log_space=log_space
        )
        encode_viewdirs = lambda x: encoder_viewdirs(x)
        d_viewdirs = encoder_viewdirs.d_output

        
        skip = []
        feature=str(dataname)+"_"+str(n_freqs)+"_"+str(n_freqs_views)+"_"+str(d_filter)+"_"+str(n_layers)+"_"+str(n_iters)

        model = NeRF(
            encoder.d_output,#+encoder_env.d_output, # Add two additional dimensions for environmental purutrbation
            n_layers=n_layers,
            d_filter=d_filter,
            skip=skip,
            d_viewdirs=d_viewdirs,
        )
        model.load_state_dict(torch.load(os.path.join(script_dir, 'pts/nerf-fine_'+feature+'.pt')))
        model.to(device)

        fine_model = NeRF(
            encoder.d_output,#+encoder_env.d_output, # Add two additional dimensions for environmental purutrbation
            n_layers=n_layers,
            d_filter=d_filter,
            skip=skip,
            d_viewdirs=d_viewdirs,
        )
        fine_model.load_state_dict(torch.load(os.path.join(script_dir, 'pts/nerf-fine_'+feature+'.pt')))
        fine_model.to(device)


        
        outputs = nerf_forward(
            rays_o,
            rays_d,
            near,
            far,
            encode,
            fine_model,
            kwargs_sample_stratified=kwargs_sample_stratified,
            n_samples_hierarchical=n_samples_hierarchical,
            kwargs_sample_hierarchical=kwargs_sample_hierarchical,
            fine_model=fine_model,
            viewdirs_encoding_fn=encode_viewdirs,
            chunksize=chunksize,
        )


        rgb_predicted = outputs["rgb_map"]
        loss = torch.nn.functional.mse_loss(rgb_predicted, testimg[0::ydown_factor,0::xdown_factor,:].reshape(-1, 3))
        print("Loss:", loss.item())

        tmp=rgb_predicted.reshape([height, width, 3])
        #print(tmp[20,30,:])
        #print(testimg[20,30,:])

        # Plot example outputs
        fig, ax = plt.subplots(
            1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [1, 1]}
        )
        ax[0].imshow(
            rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy()
        )
        ax[0].set_title(f"Iteration: {n_iters}")
        ax[1].imshow(testimg[0::ydown_factor,0::xdown_factor,:].detach().cpu().numpy())
        ax[1].set_title(f"Target")

        plt.show()
        input(f"Press Enter to see the next predicted image... ({i+1}/{num_pos})")