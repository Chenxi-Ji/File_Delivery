import os
from typing import Optional, Tuple, List, Union, Callable
from tqdm import tqdm

import time
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange
import cv2 

# from augment_dataset import adjust_hue, adjust_saturation
# from nerf_simple_env import get_rays, nerf_forward, PositionalEncoder, NeRF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Computing the input 
# if not os.path.exists('tiny_nerf_data.npz'):
#   !wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
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

def adjust_saturation(image, saturation_scale=1.5):    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Adjust the saturation
    hsv[:, :, 1] *= (saturation_scale+1)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    
    # Convert back to BGR
    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return adjusted_image

def adjust_hue(image, hue_shift=10):    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Adjust the hue
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 360  # OpenCV hue range is [0,179]
    
    # Convert back to BGR
    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return adjusted_image

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

    # Draw uniform samples from bins along ray
    if perturb:
        mids = 0.5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.concat([mids, z_vals[-1:]], dim=-1)
        lower = torch.concat([z_vals[:1], mids], dim=-1)
        t_rand = torch.rand([n_samples], device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])

    # Apply scale from `rays_d` and offset from `rays_o` to samples
    # pts: (width, height, n_samples, 3)
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
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
    viewdirs = encoding_function(viewdirs)
    viewdirs = get_chunks(viewdirs, chunksize=chunksize)
    return viewdirs

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

    return rgb_map, depth_map, acc_map, weights

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
    cumprod = torch.cumprod(tensor, -1)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, -1)
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

    return rgb_map, depth_map, acc_map, weights

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

    # Prepare batches.
    batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
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

    # Origin is same for all directions (the optical center)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def render_image(
    model: nn.Module,
    fine_model: Optional[nn.Module],
    encode: Callable[[torch.Tensor], torch.Tensor],
    encode_viewdirs: Optional[Callable[[torch.Tensor], torch.Tensor]],
    test_pose: torch.Tensor,
    hue: float,
    saturation: float,
    data_path: str,
    testimgidx: int = 13,
    near: float = 2.0,
    far: float = 6.0,
    n_samples: int = 64,
    perturb: bool = True,
    inverse_depth: bool = False,
    n_samples_hierarchical: int = 64,
    perturb_hierarchical: bool = True,
    chunksize: int = 2**14,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render an image based on the provided hue and saturation.
    Returns the rendered image and the ground truth image.
    """
    # Load data
    data = np.load(os.path.join(data_path, "tiny_nerf_data.npz"))
    images = data["images"]
    poses = data["poses"]
    focal = float(data["focal"])

    # Get ground truth image
    testimg = images[testimgidx]
    testpose_np = poses[testimgidx]

    # Adjust hue and saturation
    cv2img = cv2.cvtColor(testimg, cv2.COLOR_RGB2BGR)
    cv2img = adjust_hue(cv2img, hue)
    cv2img = adjust_saturation(cv2img, saturation)
    adjusted_img = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)

    # Convert to torch tensors
    testimg_tensor = torch.Tensor(adjusted_img).to(device)
    testpose_tensor = torch.Tensor(testpose_np).to(device)
    focal_tensor = torch.Tensor([focal]).to(device)

    height, width = testimg.shape[:2]
    rays_o, rays_d = get_rays(height, width, focal, testpose_tensor)
    rays_o = rays_o.reshape([-1, 3])
    rays_d = rays_d.reshape([-1, 3])

    # Forward pass through NeRF
    outputs = nerf_forward(
        rays_o,
        rays_d,
        near,
        far,
        encode,
        model,
        kwargs_sample_stratified={
            "n_samples": n_samples,
            "perturb": perturb,
            "inverse_depth": inverse_depth,
        },
        n_samples_hierarchical=n_samples_hierarchical,
        kwargs_sample_hierarchical={"perturb": perturb_hierarchical},
        fine_model=fine_model,
        viewdirs_encoding_fn=encode_viewdirs,
        chunksize=chunksize,
    )

    rgb_predicted = outputs["rgb_map"]
    rgb_image = rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy()
    rgb_image = np.clip(rgb_image, 0, 1)

    # Ground truth image
    ground_truth = testimg_tensor.reshape([height, width, 3]).detach().cpu().numpy()
    ground_truth = np.clip(ground_truth, 0, 1)

    return rgb_image, ground_truth

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

def xyzrpy_to_extrinsic_matrix(xyzrpy):
    x=xyzrpy[0:1]
    y=xyzrpy[1:2]
    z=xyzrpy[2:3]
    gamma = xyzrpy[3:4]
    beta = xyzrpy[4:5]
    alpha = xyzrpy[5:6]

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
    R_row0 = torch.cat([R00, R01, R02, x], dim=-1)  # First row
    R_row1 = torch.cat([R10, R11, R12, y], dim=-1)  # Second row
    R_row2 = torch.cat([R20, R21, R22, z], dim=-1)  # Third row
    R_row3 = torch.cat([torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x), torch.ones_like(x)], dim=-1)  # Fourth row

    # Use torch.cat to concatenate along the second dimension (row-wise)
    extrinsic_matrices = torch.cat([R_row0.unsqueeze(-2), R_row1.unsqueeze(-2), R_row2.unsqueeze(-2), R_row3.unsqueeze(-2)], dim=-2)


    return extrinsic_matrices

if __name__ == "__main__":
    dataname='tinydozer'
    n_samples = 32
    n_layers = 2
    d_filter = 128
    n_iters=100000
    xdown_factor,ydown_factor=1,1
    chunksize = 2**17
    eps=0.001
    testimgidx = 13
    ptb_type='xyzrpy'

    datapath='data/'+dataname+'_data.npz'

    data = np.load(os.path.join(script_dir,datapath))
    images = data["images"]
    poses = data["poses"]
    focal = data["focal"]

    testimg = images[testimgidx]
    testpose = poses[testimgidx]

    # hue = np.random.uniform(-30,30)
    # sat = np.random.uniform(-0.5,0.5)
    # hue = 30
    # sat = -0.5

    #print(f">>> Test: idx {testimgidx}; hue {hue}; sat {sat}")

    cv2img = cv2.cvtColor(testimg, cv2.COLOR_RGB2BGR)
    # cv2img = adjust_hue(cv2img, hue)
    # cv2img = adjust_saturation(cv2img, sat)
    testimg = cv2.cvtColor(cv2img, cv2.COLOR_RGB2BGR)

    testimg = torch.Tensor(testimg).to(device)
    focal = torch.Tensor(focal).to(device)

    height, width = testimg.shape[0]//ydown_factor,testimg.shape[1]//xdown_factor
    focal_x,focal_y=focal//xdown_factor,focal//ydown_factor

    input_type= "xyzrpy"#"extrinsic_matrix"
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

    xyzrpy = extrinsic_matrix_to_xyzrpy(testpose)
    print('xyzrpy:',xyzrpy)
    xyzrpy=torch.Tensor(xyzrpy).to(device)
    extrinsic_matrix = torch.Tensor(testpose).to(device)

    testpose = torch.Tensor(testpose).to(device)


    rays_o, rays_d = get_rays(height, width, focal_x,focal_y, testpose)
    rays_o = rays_o.reshape([-1, 3])
    rays_d = rays_d.reshape([-1, 3])

    outputs = nerf_forward(
        rays_o,
        rays_d,
        near,
        far,
        encode,
        model,
        kwargs_sample_stratified=kwargs_sample_stratified,
        n_samples_hierarchical=n_samples_hierarchical,
        kwargs_sample_hierarchical=kwargs_sample_hierarchical,
        fine_model=fine_model,
        viewdirs_encoding_fn=encode_viewdirs,
        chunksize=chunksize,
    )


    rgb_predicted_no_ptb = outputs["rgb_map"]
    #loss = torch.nn.functional.mse_loss(rgb_predicted_no_ptb, testimg.reshape(-1, 3))
    #print("Loss:", loss.item())


    # Compute lower and upper bound based on sampling
    torch.seed()

    image_no_ptb=rgb_predicted_no_ptb.reshape([height, width, 3]).detach().cpu().numpy()
    image_exp_lb=image_no_ptb
    image_exp_ub=image_no_ptb

    start_time=time.time()
    for i in tqdm(range(10000)):
        if input_type=="xyzrpy":
            if ptb_type=='xyzrpy':
                random_tensor=(2*torch.rand(6)-1).to(device)
            elif ptb_type=='xyz':
                random_tensor=torch.cat((2*torch.rand(3)-1, torch.zeros(3)), dim=0).to(device)
            elif ptb_type=='rpy':
                random_tensor=torch.cat((torch.zeros(3),2*torch.rand(3)-1), dim=0).to(device)
            
            ptb=eps*random_tensor
            # ptb=eps*(2*torch.rand_like(xyzrpy)-1)
            # print('shape:',ptb.shape)
            inputpose=xyzrpy_to_extrinsic_matrix(xyzrpy+ptb)
        elif input_type=="extrinsic_matrix":
            ptb=eps*(2*torch.rand_like(extrinsic_matrix)-1)
            inputpose=extrinsic_matrix+ptb

        rays_o, rays_d = get_rays(height, width, focal_x,focal_y, inputpose)
        rays_o = rays_o.reshape([-1, 3])
        rays_d = rays_d.reshape([-1, 3])

        outputs = nerf_forward(
            rays_o,
            rays_d,
            near,
            far,
            encode,
            model,
            kwargs_sample_stratified=kwargs_sample_stratified,
            n_samples_hierarchical=n_samples_hierarchical,
            kwargs_sample_hierarchical=kwargs_sample_hierarchical,
            fine_model=fine_model,
            viewdirs_encoding_fn=encode_viewdirs,
            chunksize=chunksize,
        )

        rgb_predicted_ptb = outputs["rgb_map"]
        image_ptb=rgb_predicted_ptb.reshape([height, width, 3]).detach().cpu().numpy()

        image_exp_lb=np.minimum(image_exp_lb,image_ptb)
        image_exp_ub=np.maximum(image_exp_ub,image_ptb)

    end_time=time.time()
    print('Running Time for For Loop:',f"{(end_time-start_time):.2f}",' sec')

    # Compute difference between ub and lb
    image_diff=image_exp_ub-image_exp_lb
    image_diff_combined=np.linalg.norm(image_diff,axis=2)
    image_MPG=np.mean(image_diff_combined)
    image_XPG=np.max(image_diff_combined)
    print('image_MPG,image_XPG:',image_MPG,image_XPG)


    # Plot example outputs
    fig, ax = plt.subplots(
        1, 4, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1, 1, 1]}
    )
    ax[0].imshow(
        image_exp_lb
    )
    ax[0].set_title(f"Sampled lb")
    ax[1].imshow(
        image_exp_ub
    )
    ax[1].set_title(f"Sampled ub")
    ax[2].imshow(
        image_no_ptb
    )
    ax[2].set_title(f"Image no ptb")
    ax[3].imshow(testimg[0::ydown_factor,0::xdown_factor,:].detach().cpu().numpy())
    ax[3].set_title(f"Ground Truth")

    if input_type=="xyzrpy":
        input_dim=6
    elif input_type=="extrinsic_matrix":
        input_dim=16

    imagename="exp_img_"+str(dataname)+"_error_"+str(eps).split(".")[1]+\
                "_features_"+str(n_freqs)+"_"+str(n_freqs_views)+"_"+str(d_filter)+"_"+str(n_layers)+"_"+str(n_iters)+\
                "_samples_"+str(n_samples)+"_inputdim_"+str(input_dim)+\
                "_xdown_"+str(xdown_factor)+"_ydown_"+str(ydown_factor)+"_whole.png"
    plt.savefig("output_img/"+imagename, bbox_inches='tight')


    plt.show()