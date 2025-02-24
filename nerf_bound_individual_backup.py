import os
from typing import Optional, Tuple, List, Union, Callable

import time
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


class TestModel(nn.Module):
    def __init__(self,total_height,total_width,start_height,start_width,end_height,end_width,focal_x,focal_y,\
                        near,far,distance_to_infinity,n_samples,perturb,inverse_depth,\
                        kwargs_sample_stratified,n_samples_hierarchical,kwargs_sample_hierarchical,chunksize,\
                        encode,encode_viewdirs,coarse_model,fine_model,\
                        raw_noise_std=0.0, print_flag=False
                        ):
        super(TestModel, self).__init__()
        self.total_height=total_height
        self.total_width=total_width
        self.start_height=start_height
        self.end_width=end_width
        self.end_height=end_height
        self.start_width=start_width
        self.focal_x=focal_x
        self.focal_y=focal_y

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
        self.encode_viewdirs=encode_viewdirs
        self.model=coarse_model
        self.fine_model=fine_model 

        self.raw_noise_std=raw_noise_std
        self.noise_rand=torch.randn((end_height-start_height)*(end_width-start_width),n_samples) * raw_noise_std


        self.print_flag=print_flag

    def get_rays(
        self, c2w: torch.Tensor
        ):

        total_height=self.total_height
        total_width=self.total_width
        start_height=self.start_height
        start_width=self.start_width
        end_height=self.end_height
        end_width=self.end_width
        focal_x_length=self.focal_x.to(c2w)
        focal_y_length=self.focal_y.to(c2w)

        r"""
        Find origin and direction of rays through every pixel and camera origin.
        """
        # print(c2w.shape)
        # Apply pinhole camera model to gather directions at each pixel
        i, j = torch.meshgrid(
            torch.arange(start=start_width, end=end_width, dtype=torch.float32).to(c2w),
            torch.arange(start=start_height,end=end_height, dtype=torch.float32).to(c2w),
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
        rays_d = torch.sum(directions[..., None, :] * c2w[...,:3, :3], dim=-1)
        #print('rays_d inside function:',rays_d.shape)

        # Origin is same for all directions (the optical center)
        #rays_o = c2w[:3, -1].expand(rays_d.shape)
        rays_o=c2w[...,:3,-1]
        #print('rays_o inside function:',rays_o.shape)

        return rays_o,rays_d
    
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
    ) -> List[torch.Tensor]:
        r"""
        Encode and chunkify points to prepare for NeRF model.
        """
        
        chunksize=self.chunksize
        points = encoding_function(points)
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
        rgb_map=torch.zeros_like(alpha[...,None, 0]).to(alpha.device)
        for i in reversed(range(self.n_samples)):
            #print('shapes:',alpha[..., None,i].shape,rgb[...,i, :].shape)
            rgb_map=alpha[..., None, i]*rgb[...,i, :]+(1-alpha[...,None, i])*rgb_map
        
        return rgb_map    
    
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

        raw_noise_std=self.raw_noise_std
        noise_rand=self.noise_rand.to(rays_d.device)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists_vals = dists_vals * self.two_norm(rays_d[..., None, :],dim=-1)
        #print(dists_vals.shape)

        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.0
        if raw_noise_std > 0.0:
            noise = noise_rand

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point. [n_rays, n_samples]
        tmp10=raw[..., 3] + noise
        tmp11=-nn.functional.relu(raw[..., 3] + noise)
        tmp12=dists_vals
        tmp2=-nn.functional.relu(raw[..., 3] + noise) * dists_vals
        tmp3= torch.exp(-nn.functional.relu(raw[..., 3] + noise) * dists_vals)
        alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * dists_vals)
        #alpha = -nn.functional.relu(raw[..., 3] + noise) * dists_vals
        #print(alpha.view(-1).min())
        #print(alpha.view(-1).max())
        #print('alpha.shape:',alpha.shape)
        
        rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
        # print(rgb.view(-1).min())
        # print(rgb.view(-1).max())
        # print('rgb.shape:',rgb.shape)

        # transmittance=alpha*self.cumprod_exclusive(1.0-alpha)
        # print('transmittance.shape:',transmittance.shape)  
        # rgb_map = torch.sum(transmittance[..., None] * rgb, dim=-2)  # [n_rays, 3]
        # print('rgb_map.shape:',rgb_map.shape)


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
        rays_d: torch.Tensor
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

        # Sample query points along each ray.
        query_points, z_vals, dists_vals = self.sample_stratified(rays_o,rays_d)
        outputs = {"z_vals_stratified": z_vals}
        # if self.print_flag:
        #     print('query_points.shape:',query_points.shape)
        #     print('z_vals.shape:',z_vals.shape)
        # print('query_points.shape:',query_points.shape)
        
        # Prepare batches.
        batches = self.prepare_chunks(query_points, encoding_fn)
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
    
    def forward(self,x):

        rays_o, rays_d=self.get_rays(x)
        if self.print_flag:
            print('rays_o.shape:',rays_o.shape)
            print('rays_d.shape:',rays_d.shape)

        #return self.nerf_forward(rays_o,rays_d)

        rgb_map=self.nerf_forward(rays_o,rays_d)
        return  rgb_map
    
        outputs=self.nerf_forward(rays_o,rays_d)
        res=outputs["rgb_map"]
        
        return res

if __name__ == "__main__":
    data = np.load(os.path.join(script_dir,"tiny_nerf_data.npz"))
    images = data["images"]
    poses = data["poses"]
    focal = data["focal"]

    testimgidx = 13
    testimg = images[testimgidx]
    testpose = poses[testimgidx]
    #print(testpose.shape)

    cv2img = cv2.cvtColor(testimg, cv2.COLOR_RGB2BGR)
    testimg = cv2.cvtColor(cv2img, cv2.COLOR_RGB2BGR)

    testimg = torch.Tensor(testimg).to(device)
    testpose = torch.Tensor(testpose).to(device)
    total_height, total_width = testimg.shape[:2]
    tile_height,tile_width=2, 2
    #height,width=torch.Tensor(height).to(device),torch.Tensor(width).to(device)
    focal_x = torch.Tensor(focal).to(device)
    focal_y = torch.Tensor(focal).to(device)

    near, far = 2.0, 6.0
    distance_to_infinity=1e0
    n_samples = 32
    perturb = False#True 
    inverse_depth = False
    kwargs_sample_stratified = {
        "n_samples": n_samples,
        "perturb": perturb,
        "inverse_depth": inverse_depth,
    }
    n_samples_hierarchical = 0
    kwargs_sample_hierarchical = {"perturb": perturb}
    chunksize = 2**1

    d_input = 3
    n_freqs = 10
    log_space = True
    n_freqs_views = 4

    n_layers = 2
    d_filter = 128
    skip = []

    raw_noise_std=0.0
    print_flag=False#True
    visual_flag=False#True
    start_vis_height,end_vis_height=49,50
    start_vis_width,end_vis_width=0,1
    eps=0.003

    image_lb=np.zeros((total_height,total_width,3))
    image_ub=np.zeros((total_height,total_width,3))
    image_exp=np.zeros((total_height,total_width,3))

    encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)
    encode = lambda x: encoder(x)
    
    encoder_viewdirs = PositionalEncoder(d_input, n_freqs_views, log_space=log_space)
    encode_viewdirs = lambda x: encoder_viewdirs(x)
    d_viewdirs = encoder_viewdirs.d_output

    coarse_model = NeRF(
        encoder.d_output,
        n_layers=n_layers,
        d_filter=d_filter,
        skip=skip,
        d_viewdirs=d_viewdirs,
    )

    coarse_model.load_state_dict(torch.load(os.path.join(script_dir, './nerf-fine_10000.pt')))
    coarse_model.to(device)

    fine_model = NeRF(
        encoder.d_output,
        n_layers=n_layers,
        d_filter=d_filter,
        skip=skip,
        d_viewdirs=d_viewdirs,
    )
    fine_model.load_state_dict(torch.load(os.path.join(script_dir, './nerf-fine_10000.pt')))
    fine_model.to(device)

    start_time=time.time()

    # for start_height in range(0,total_height,tile_height):
    #     for start_width in range(0,total_width,tile_width):

    for start_height in range(start_vis_height,end_vis_height,tile_height):
        for start_width in range(start_vis_width,end_vis_width,tile_width):

            epoch_start_time=time.time()

            end_height=min(start_height+tile_height,total_height)
            end_width=min(start_width+tile_width,total_width)

            print('cur_height,cur_width:',start_height,start_width)
            #start_height,start_width=56,56
            #end_height,end_width=60,60
            
            ray_model=TestModel(total_height,total_width,start_height,start_width,end_height,end_width,focal_x,focal_y,\
                                near,far,distance_to_infinity,n_samples,perturb,inverse_depth,\
                                kwargs_sample_stratified,n_samples_hierarchical,kwargs_sample_hierarchical,chunksize,\
                                encode,encode_viewdirs,coarse_model,fine_model,\
                                raw_noise_std,print_flag
                                )
            inputpose=testpose.repeat((end_height-start_height)*(end_width-start_width),1,1)
            exp=ray_model.forward(inputpose)


            model = BoundedModule(ray_model, inputpose)
            #model.visualize('b')
            ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
            #ptb = PerturbationLpNorm(norm=np.inf, eps=0.001)
            inputpose_ptb = BoundedTensor(inputpose, ptb)

            lb, ub = model.compute_bounds(x=(inputpose_ptb,), method="backward")

            #print("Lower bounds: ", lb)
            #print("Upper bounds: ", ub)
            # print('bound:',lb.view(-1).min(),ub.view(-1).max())

            print('bound:',torch.min(lb,dim=0),torch.max(ub,dim=0))


            # Establish the whole image by composing every tile
            if visual_flag:
                lb_reshape=lb.reshape([end_height-start_height, end_width-start_width, 3])
                ub_reshape=ub.reshape([end_height-start_height, end_width-start_width, 3])
                exp_reshape=exp.reshape([end_height-start_height, end_width-start_width, 3])

                image_lb[start_height:end_height,start_width:end_width,:]=lb_reshape.detach().cpu().numpy()
                image_ub[start_height:end_height,start_width:end_width,:]=ub_reshape.detach().cpu().numpy()
                image_exp[start_height:end_height,start_width:end_width,:]=exp_reshape.detach().cpu().numpy()

            epoch_end_time=time.time()
            print('Epoch Running Time:', f"{epoch_end_time-epoch_start_time:.2f}", 'sec')

    end_time=time.time()
    print('Running Time:',f"{(end_time-start_time)/60:.2f}",' min')
    
    if visual_flag:

        fig, ax = plt.subplots(
            1,  4, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1, 1, 1]}
        )
        ax[0].imshow(
            image_lb[start_vis_height:end_vis_height,start_vis_width:end_vis_width]
        )
        ax[0].set_title(f"image_lb")
        ax[1].imshow(
            image_ub[start_vis_height:end_vis_height,start_vis_width:end_vis_width]
        )
        ax[1].set_title(f"image_ub")
        ax[2].imshow(
            image_exp[start_vis_height:end_vis_height,start_vis_width:end_vis_width]
        )
        ax[2].set_title(f"image_no_perturb")
        ax[3].imshow(
            testimg[start_vis_height:end_vis_height,start_vis_width:end_vis_width].detach().cpu().numpy()
        )
        #ax[3].imshow(testimg[start_height:end_height,start_height:end_width,:].detach().cpu().numpy())
        ax[3].set_title(f"Ground Truth")

        plt.show()

    


    