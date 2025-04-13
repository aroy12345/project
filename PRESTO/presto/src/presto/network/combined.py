#!/usr/bin/env python3

from typing import Union, Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, asdict, field
from contextlib import nullcontext
import time
import logging
import os

# Create a separate named logger for this module
def setup_model_logger():
    # Get the directory where combined.py is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(module_dir, 'log_prestoGIGA_model.txt')
    
    # Create a named logger
    logger = logging.getLogger('prestoGIGA')
    
    # Only add handlers if they don't already exist
    if not logger.handlers:
        # Create a file handler
        file_handler = logging.FileHandler(log_file, mode='a')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        # Don't propagate to the root logger
        logger.propagate = False
        
        logger.info("--- PrestoGIGA Model Logging Initialized ---\n")
    
    return logger

# Create the logger instance
model_logger = setup_model_logger()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import einops
from timm.models.vision_transformer import Attention, Mlp
from torch import distributions as dist

try:
    from diffusers.models.unet_1d import UNet1DOutput
except ImportError:
    from diffusers.models.unets.unet_1d import UNet1DOutput

from presto.network.layers import SinusoidalPositionalEncoding
from presto.network.dit import (
    DiT, DiTBlock, PatchEmbed, FinalLayer, 
    TimestepEmbedder, CondEmbedder,
    get_1d_sincos_pos_embed
)

from presto.ConvONets.encoder.unet import UNet
from presto.ConvONets.encoder.unet3d import UNet3D
from presto.ConvONets.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate
import torch.nn.functional as F
from presto.ConvONets.layers import ResnetBlockFC
from presto.ConvONets.common import normalize_coordinate, normalize_3d_coordinate, map2local
from torch_scatter import scatter_mean

class LocalVoxelEncoder(nn.Module):
    ''' 3D-convolutional encoder network for voxel input.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent code c
        hidden_dim (int): hidden dimension of the network
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        kernel_size (int): kernel size for the first layer of CNN
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    
    '''

    def __init__(self, dim=3, c_dim=128, unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, 
                 plane_resolution=512, grid_resolution=None, plane_type='xz', kernel_size=3, padding=0.1):
        super().__init__()
        self.actvn = F.relu
        if kernel_size == 1:
            self.conv_in = nn.Conv3d(1, c_dim, 1)
        else:
            self.conv_in = nn.Conv3d(1, c_dim, kernel_size, padding=1)

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        self.c_dim = c_dim

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution

        self.plane_type = plane_type
        self.padding = padding


    def generate_plane_features(self, p, c, plane='xz'):
        # p: [B, N_voxels, 3] e.g. [8, 32768, 3]
        # c: [B, N_voxels, C_dim] e.g. [8, 32768, 32] (Input c from forward)

        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # Shape: [B, N_voxels, 2]
        # index from corrected coordinate2index: [B, N_voxels] e.g. [8, 32768]
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        # Permute c to [B, C_dim, N_voxels] for scatter_mean's src argument
        c = c.permute(0, 2, 1) # Shape: [B, C_dim, N_voxels] e.g. [8, 32, 32768]

        # Output shape needs to be [B, C_dim, reso^2]
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)

        # Expand index to match the dimensions of src (c) for scattering
        # Target index shape for scatter_mean: [B, C_dim, N_voxels]
        # Current index shape: [B, N_voxels]
        # Add C_dim dimension and expand
        index = index.unsqueeze(1).expand_as(c) # Shape: [8, 32, 32768]

        # Scatter c along the last dimension (dim=-1)
        # src = c [B, C_dim, N_voxels]
        # index = index [B, C_dim, N_voxels] <--- Corrected shape
        # out = fea_plane [B, C_dim, reso^2]
        fea_plane = scatter_mean(src=c, index=index, dim=-1, out=fea_plane, dim_size=self.reso_plane**2)

        # Reshape fea_plane to expected [B, C_dim, reso, reso] format for UNet
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        # p: [B, N_voxels, 3] e.g. [8, 32768, 3]
        # c: [B, N_voxels, C_dim] e.g. [8, 32768, 32] (Input c from forward)

        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # Shape: [B, N_voxels, 3]
        # index from corrected coordinate2index: [B, N_voxels] e.g. [8, 32768]
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')

        # scatter grid features from points
        # Output shape: [B, C_dim, reso_grid^3]
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3)
        # Permute c to [B, C_dim, N_voxels] for scatter_mean's src argument
        c = c.permute(0, 2, 1) # Shape: [B, C_dim, N_voxels] e.g. [8, 32, 32768]

        # Expand index to match the dimensions of src (c) for scattering
        # Target index shape: [B, C_dim, N_voxels]
        # Current index shape: [B, N_voxels]
        # Add C_dim dimension and expand
        index = index.unsqueeze(1).expand_as(c) # Shape: [8, 32, 32768]

        # Scatter along dim=-1 (N_voxels dimension)
        fea_grid = scatter_mean(src=c, index=index, dim=-1, out=fea_grid, dim_size=self.reso_grid**3)

        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid


    def forward(self, x):
        # Input x shape: [B, C_in, G, G, G], e.g., [8, 1, 32, 32, 32]
        batch_size = x.size(0)
        device = x.device
        # Use spatial dimensions (G, G, G) starting from index 2
        grid_res = x.shape[2:5] # Tuple (G, G, G)
        n_voxel = grid_res[0] * grid_res[1] * grid_res[2] # G*G*G

        # Create coordinates for the GxGxG grid
        coord1 = torch.linspace(-0.5, 0.5, grid_res[0], device=device)
        coord2 = torch.linspace(-0.5, 0.5, grid_res[1], device=device)
        coord3 = torch.linspace(-0.5, 0.5, grid_res[2], device=device)

        # Create meshgrid and expand to batch size
        grid_coords = torch.stack(torch.meshgrid(coord1, coord2, coord3, indexing='ij'), dim=-1) # Shape [G, G, G, 3]
        p = grid_coords.unsqueeze(0).expand(batch_size, -1, -1, -1, -1) # Shape [B, G, G, G, 3]
        p = p.reshape(batch_size, n_voxel, 3) # Shape [B, n_voxel, 3], e.g., [8, 32768, 3]

        # Acquire voxel-wise feature
        # conv_in expects [B, C_in, G, G, G] - input x is already in this format
        c = self.actvn(self.conv_in(x)) # Output shape [B, c_dim, G, G, G]
        # Flatten spatial dimensions G*G*G = n_voxel
        c = c.view(batch_size, self.c_dim, n_voxel) # Shape [B, c_dim, n_voxel], e.g., [8, 32, 32768]
        c = c.permute(0, 2, 1) # Shape [B, n_voxel, c_dim], e.g., [8, 32768, 32]

        fea = {}
        if 'grid' in self.plane_type:
            # Pass the correctly shaped p and c
            fea['grid'] = self.generate_grid_features(p, c)
        else:
            # Pass the correctly shaped p and c
            if 'xz' in self.plane_type:
                fea['xz'] = self.generate_plane_features(p, c, plane='xz')
            if 'xy' in self.plane_type:
                fea['xy'] = self.generate_plane_features(p, c, plane='xy')
            if 'yz' in self.plane_type:
                fea['yz'] = self.generate_plane_features(p, c, plane='yz')

        return fea

class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, 
                 n_blocks=5, 
                 out_dim=1, 
                 leaky=False, 
                 sample_mode='bilinear', 
                 padding=0.1,
                 concat_feat=False,
                 no_xyz=False):
        super().__init__()
        
        self.concat_feat = concat_feat
        if concat_feat:
            c_dim *= 3
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.no_xyz = no_xyz
        self.hidden_size = hidden_size

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if not no_xyz:
            self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
    

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c


    def forward(self, p, c_plane, **kwargs):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                if 'grid' in plane_type:
                    c = self.sample_grid_feature(p, c_plane['grid'])
                if 'xz' in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane['xz'], plane='xz'))
                if 'xy' in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane['xy'], plane='xy'))
                if 'yz' in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane['yz'], plane='yz'))
                c = torch.cat(c, dim=1)
                c = c.transpose(1, 2)
            else:
                c = 0
                if 'grid' in plane_type:
                    c += self.sample_grid_feature(p, c_plane['grid'])
                if 'xz' in plane_type:
                    c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
                if 'xy' in plane_type:
                    c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
                if 'yz' in plane_type:
                    c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
                c = c.transpose(1, 2)

        p = p.float()

        if self.no_xyz:
            net = torch.zeros(p.size(0), p.size(1), self.hidden_size).to(p.device)
        else:
            net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

    def query_feature(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_grid_feature(p, c_plane['grid'])
            if 'xz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
            c = c.transpose(1, 2)
        return c

    def compute_out(self, p, c):
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


encoder_dict = {
    'voxel_simple_local': LocalVoxelEncoder,
}

decoder_dict = {
    'simple_local': LocalDecoder,
}




class TSDFEmbedder(nn.Module):
    """Embedder for concatenated 2D TSDF feature planes (xy, xz, yz) from GIGA's encoder."""
    def __init__(self, hidden_size: int = 256, c_dim: int = 32, input_resolution: int = 32):
        """
        Args:
            hidden_size: The target output embedding dimension.
            c_dim: The feature dimension of *each* input plane from the encoder.
            input_resolution: The spatial resolution (H=W) of the input feature planes.
        """
        super().__init__()
        # Input will be concatenation of 3 planes, so input channels = 3 * c_dim
        input_channels = 3 * c_dim
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Calculate the flattened size after convolutions
        # Input: input_resolution -> Output: input_resolution / 2 / 2 / 2
        final_res = input_resolution // 8
        flat_size = 256 * final_res * final_res
        self.proj = nn.Linear(flat_size, hidden_size)
        self.c_dim = c_dim
        self.input_resolution = input_resolution # Store for potential checks

    def forward(self, x_planes: Dict[str, torch.Tensor]):
        """
        Forward pass for the TSDF embedder using 2D feature planes.

        Args:
            x_planes: A dictionary containing the 2D feature planes,
                      e.g., {'xy': [B, C, H, W], 'xz': [B, C, H, W], 'yz': [B, C, H, W]}

        Returns:
            torch.Tensor: The final embedding vector [B, hidden_size].
        """
        # Ensure required planes are present and have expected shape
        required_planes = ['xy', 'xz', 'yz']
        if not all(plane in x_planes for plane in required_planes):
            raise ValueError(f"Missing one or more required planes ({required_planes}) in input dictionary.")

        # Check shapes (optional but good practice)
        B, C, H, W = x_planes['xy'].shape
        if C != self.c_dim or H != self.input_resolution or W != self.input_resolution:
             model_logger.info(f"Warning: Input plane shape ({B},{C},{H},{W}) doesn't match expected c_dim ({self.c_dim}) or resolution ({self.input_resolution}).")
             # Or raise an error:
             # raise ValueError(f"Input plane shape mismatch...")

        # Concatenate planes along the channel dimension
        x_cat = torch.cat([x_planes['xy'], x_planes['xz'], x_planes['yz']], dim=1) # Shape: [B, 3*C, H, W]

        x = F.relu(self.bn1(self.conv1(x_cat)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1) # Flatten
        x = self.proj(x)
        return x


class PrestoGIGA(nn.Module):
    """
    Unified model that combines PRESTO's diffusion transformer with GIGA's grasp affordance prediction
    """
    @dataclass
    class Config:
        input_size: int = 1000
        patch_size: int = 20
        in_channels: int = 7
        hidden_size: int = 256
        num_layer: int = 4
        num_heads: int = 16
        mlp_ratio: float = 4.0
        class_dropout_prob: float = 0.0
        cond_dim: int = 104
        learn_sigma: bool = True
        use_cond: bool = True
        use_pos_emb: bool = False
        dim_pos_emb: int = 3 * 2 * 32
        sin_emb_x: int = 0
        cat_emb_x: bool = False
        use_cond_token: bool = False
        use_cloud: bool = False
        
        grid_size: int = 40
        c_dim: int = 32
        use_grasp: bool = True
        use_tsdf: bool = True
        
        decoder_type: str = 'simple_local'
        decoder_kwargs: Dict[str, Any] = field(default_factory=lambda: {
            'dim': 3,
            'sample_mode': 'bilinear',
            'hidden_size': 32,
            'concat_feat': True
        })
        padding: float = 0.1
        
        use_amp: Optional[bool] = None
        use_joint_embeddings: bool = True
        
        encoder_type: Optional[str] = 'voxel_simple_local'
        encoder_kwargs: Dict[str, Any] = field(default_factory=lambda: {
            'plane_type': ['grid'],
            'plane_resolution': 40,
            'grid_resolution': 32,
            'unet': True,
            'unet3d': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        })

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        
        self._init_diffusion()
        self._init_grasp()
        
       
        
    def _init_diffusion(self):
        """Initialize diffusion model components from PRESTO"""
        cfg = self.cfg
        
        self.learn_sigma = cfg.learn_sigma
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.in_channels * 2 if cfg.learn_sigma else cfg.in_channels
        self.patch_size = cfg.patch_size
        self.num_heads = cfg.num_heads
        self.use_cond = cfg.use_cond
        self.use_cond_token = cfg.use_cond_token
        
        self.x_embedder = PatchEmbed(
            cfg.input_size, cfg.in_channels,
            cfg.patch_size, cfg.hidden_size,
            sin_emb=cfg.sin_emb_x,
            cat=cfg.cat_emb_x
        )
        self.t_embedder = TimestepEmbedder(cfg.hidden_size)
        
        if self.use_cond:
            if cfg.use_pos_emb:
                self.y_embedder = nn.Sequential(
                    SinusoidalPositionalEncoding(cfg.cond_dim, cfg.dim_pos_emb),
                    CondEmbedder(cfg.dim_pos_emb, cfg.hidden_size, cfg.class_dropout_prob)
                )
            else:
                self.y_embedder = CondEmbedder(
                    cfg.cond_dim, cfg.hidden_size, cfg.class_dropout_prob)
        
        num_patches = self.x_embedder.num_patches
        
        self.register_buffer('pos_embed',
                            torch.zeros(num_patches, cfg.hidden_size),
                            persistent=False)
                            
        if cfg.encoder_type:
            self.tsdf_encoder = encoder_dict[cfg.encoder_type](
                c_dim=cfg.c_dim, padding=cfg.padding,
                **vars(cfg.encoder_kwargs)
            )
        else:
            self.tsdf_encoder = None
        
        self.tsdf_embedder = TSDFEmbedder(cfg.hidden_size, cfg.c_dim)
        
        self.blocks = nn.ModuleList([
            DiTBlock(
                cfg.hidden_size, cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                use_cond=True
            )
            for _ in range(cfg.num_layer)
        ])
        
        self.final_layer = FinalLayer(cfg.hidden_size, cfg.patch_size, self.out_channels)
        
    def _init_grasp(self):
        """Initialize GIGA components (encoder, decoder, embedder)"""
        cfg = self.cfg # Use local cfg for brevity

        # Initialize rotation representation (used for decoder out_dim)
        self.rot_dim = 4 # Example: 6D rotation representation

        if cfg.encoder_type:
            # Instantiate the TSDF/Voxel Encoder
            encoder_cls = encoder_dict[cfg.encoder_type]
            self.tsdf_encoder = encoder_cls(
                c_dim=cfg.c_dim,
                padding=cfg.padding,
                **vars(cfg.encoder_kwargs)
            )
        else:
            self.tsdf_encoder = None

        # Instantiate the TSDF Embedder (for conditioning DiT)
        self.tsdf_embedder = TSDFEmbedder(
            hidden_size=cfg.hidden_size,
            c_dim=cfg.c_dim
        )

        # --- MODIFIED DECODER INITIALIZATION ---
        self.decoder_qual = None
        self.decoder_rot = None
        self.decoder_width = None
        self.decoder_tsdf = None # Optional: if you predict TSDF too

        if cfg.decoder_type and (cfg.use_grasp or cfg.use_tsdf):
            decoder_cls = decoder_dict[cfg.decoder_type]

            # Common arguments for all decoders
            common_decoder_kwargs = {
                'dim': 3,
                'c_dim': cfg.c_dim,
                **vars(cfg.decoder_kwargs) # Unpack specific decoder args
            }

            if cfg.use_grasp:
                # Decoder for Quality (output dim 1)
                self.decoder_qual = decoder_cls(
                    out_dim=1,
                    **common_decoder_kwargs
                )
                # Decoder for Rotation (output dim self.rot_dim)
                self.decoder_rot = decoder_cls(
                    out_dim=self.rot_dim,
                    **common_decoder_kwargs
                )
                # Decoder for Width (output dim 1)
                self.decoder_width = decoder_cls(
                    out_dim=1,
                    **common_decoder_kwargs
                )

            if cfg.use_tsdf:
                 # Optional: Decoder for TSDF prediction (output dim 1)
                 self.decoder_tsdf = decoder_cls(
                     out_dim=1,
                     **common_decoder_kwargs
                 )
        # --- END MODIFIED DECODER INITIALIZATION ---

        # Remove the single self.decoder initialization if it exists elsewhere
        # if hasattr(self, 'decoder'):
        #     del self.decoder

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize weights for linear and conv layers"""
        if isinstance(m, nn.Linear):
            # Use Xavier uniform initialization for linear layers
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)): # Assuming 3D convs might also be used
            # Use Kaiming normal initialization for conv layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def get_num_patches(self, x_shape):
        return x_shape[-1] // self.patch_size

    def set_pos_embed(self, grid_size):
        """
        Set positional embedding buffer
        """
        pos_embed = get_1d_sincos_pos_embed(
            self.cfg.hidden_size, grid_size, cls_token=False)
        self.pos_embed = torch.from_numpy(pos_embed).float().to(self.device)
            
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size
        x = einops.rearrange(x, '... s (p c) -> ... c (s p)',
                             c=self.out_channels)
        return x
        
    def encode_shared(self, x=None, tsdf=None, t=None, y=None):
        """
        Create a shared representation from either or both diffusion and TSDF inputs

        Args:
            x: [B, C, T] sequence data for diffusion
            tsdf: [B, 1, D, D, D] TSDF voxel grid (input to encoder)
            t: diffusion timesteps
            y: conditioning variables

        Returns:
            features: [B, N, D] transformer features
            c: [B, D] conditioning embedding
            tsdf_features: Dictionary containing output features from tsdf_encoder (e.g., {'xy': ..., 'xz': ..., 'yz': ...})
                           or None if tsdf is None.
        """
        model_logger.info(f"[PrestoGIGA.encode_shared] Input shapes: x={x.shape if x is not None else None}, "
              f"tsdf={tsdf.shape if tsdf is not None else None}, "
              f"t={t.shape if t is not None else None}, "
              f"y={y.shape if y is not None else None}")
        
        if x is not None:
            # Ensure pos_embed is initialized if needed (moved from original code block)
            print(x.shape, 'x')
            if not hasattr(self, 'pos_embed') or self.pos_embed is None:
                 print("valid")
                 self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, self.cfg.hidden_size), requires_grad=False)
                 pos_embed_1d = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.num_patches)
                 self.pos_embed.data.copy_(torch.from_numpy(pos_embed_1d).float().unsqueeze(0))

            x_embed = self.x_embedder(x) + self.pos_embed
            print(x_embed.shape, 'x_embed')
        else:
            x_embed = None

        tsdf_embed = None
        tsdf_features = None # Initialize tsdf_features

        if tsdf is not None:
            if self.tsdf_encoder is not None:
                # tsdf_encoder now returns a dictionary of features (planes or grid)
                tsdf_features = self.tsdf_encoder(tsdf) # e.g., {'xy': [B,C,H,W], 'xz': [B,C,H,W], 'yz': [B,C,H,W]}
                print(self.tsdf_encoder, 'tsdf_encoder')
                print(tsdf_features['xz'].shape, 'tsdf_features')
                # Pass the dictionary directly to the embedder
                tsdf_embed = self.tsdf_embedder(tsdf_features) # Expects dict, outputs [B, hidden_size]
                tsdf_embed = tsdf_embed.unsqueeze(1) # Shape: [B, 1, hidden_size]
                print(tsdf_embed.shape, 'tsdf_embed')
            else:
                # Handle case where there's no encoder but tsdf is provided (unlikely for this setup)
                model_logger.info("Warning: TSDF provided but no tsdf_encoder defined.")
                # tsdf_embed remains None

        # Create combined features for the transformer blocks
        if x_embed is not None and tsdf_embed is not None and self.cfg.use_joint_embeddings:
            # Concatenate sequence embedding and the single TSDF embedding token
            model_logger.info('Concatenating x_embed and tsdf_embed for joint embeddings')
            features = torch.cat([x_embed, tsdf_embed], dim=1)
            print(features.shape, 'features')
        elif x_embed is not None:
            features = x_embed
        elif tsdf_embed is not None:
            # If only TSDF is provided, expand its embedding to match sequence length expectations
            # Ensure pos_embed is initialized if needed (as above)
            if not hasattr(self, 'pos_embed') or self.pos_embed is None:
                 self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, self.cfg.hidden_size), requires_grad=False)
                 pos_embed_1d = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.num_patches)
                 self.pos_embed.data.copy_(torch.from_numpy(pos_embed_1d).float().unsqueeze(0))

            # Expand the single TSDF embedding token across the sequence dimension
            # Note: This assumes the transformer expects a sequence. Adjust if architecture changes.
            # We might need a different way to handle TSDF-only input depending on the transformer design.
            # For now, replicate the TSDF embedding and add positional encoding.
            features = tsdf_embed.expand(-1, self.x_embedder.num_patches, -1)
            features = features + self.pos_embed # Add positional encoding
        else:
            raise ValueError("Either x or tsdf must be provided")

        # --- Conditioning ---
        t_embed = self.t_embedder(t) if t is not None else None # [B, D]
        y_embed = self.y_embedder(y, train=self.training) if y is not None else None # [B, D]
        print(t.shape, 't')
        print(y.shape, 'y')
        print(t_embed.shape, 't_embed')
        print(y_embed.shape, 'y_embed')
        # Combine conditioning embeddings (handle None cases)
        if t_embed is not None and y_embed is not None:
            c = t_embed + y_embed
            print(c.shape, 'c')
        elif t_embed is not None:
            c = t_embed
        elif y_embed is not None:
            c = y_embed
        else:
            # If no time or class conditioning, create a zero embedding or handle as needed
            c = torch.zeros(features.shape[0], self.cfg.hidden_size, device=features.device, dtype=features.dtype)
            # Alternatively, could raise an error if conditioning is always expected

        # After TSDF encoder
        if tsdf is not None and self.tsdf_encoder is not None:
            model_logger.info(f"[PrestoGIGA.encode_shared] TSDF encoder output keys: {list(tsdf_features.keys())}")
            for k, v in tsdf_features.items():
                if isinstance(v, torch.Tensor):
                    model_logger.info(f"[PrestoGIGA.encode_shared] TSDF feature '{k}' shape: {v.shape}, "
                         f"range: [{v.min().item():.4f}, {v.max().item():.4f}]")
        
        # After TSDF embedding
        if 'tsdf_embed' in locals() and tsdf_embed is not None:
            model_logger.info(f"[PrestoGIGA.encode_shared] TSDF embedding shape: {tsdf_embed.shape}, "
                  f"range: [{tsdf_embed.min().item():.4f}, {tsdf_embed.max().item():.4f}]")
        
        # After sequence embedding
        if 'x_embed' in locals() and x_embed is not None:
            model_logger.info(f"[PrestoGIGA.encode_shared] Sequence embedding shape: {x_embed.shape}, "
                  f"range: [{x_embed.min().item():.4f}, {x_embed.max().item():.4f}]")
        
        # After combined features
        if 'features' in locals() and features is not None:
            model_logger.info(f"[PrestoGIGA.encode_shared] Combined features shape: {features.shape}, "
                  f"range: [{features.min().item():.4f}, {features.max().item():.4f}]")

        # At the end before return
        model_logger.info(f"[PrestoGIGA.encode_shared] Output shapes: features={features.shape if features is not None else None}, "
              f"c={c.shape if c is not None else None}, "
              f"tsdf_features={type(tsdf_features)}, has {len(tsdf_features) if tsdf_features else 0} keys")
        
        return features, c, tsdf_features
        
    
    def forward_grasp(self, tsdf_features: Dict[str, torch.Tensor], positions: torch.Tensor):
        """
        Process features through grasp output heads using GIGA's decoder.
        """
        model_logger.info(f"[PrestoGIGA.forward_grasp] Input shapes: positions={positions.shape}")
        model_logger.info(f"[PrestoGIGA.forward_grasp] tsdf_features keys: {list(tsdf_features.keys())}")
        
        if self.decoder_qual is None:
            raise RuntimeError("Grasp quality decoder is not initialized.")

        # The GIGA decoder expects the encoded features 'c' directly.
        # The dictionary tsdf_features holds these features.
        qual = self.decoder_qual(positions, tsdf_features) # Pass positions and the feature dict
        print(self.decoder_qual, 'decoder_qual', self.decoder_rot, 'decoder_rot', self.decoder_width, 'decoder_width')

        if self.decoder_rot is None:
            rot = torch.zeros(positions.shape[0], positions.shape[1], self.rot_dim, device=positions.device)
            warnings.warn("Grasp rotation decoder is not initialized.", RuntimeWarning)
        else:
            rot = self.decoder_rot(positions, tsdf_features) # Pass positions and the feature dict

        if self.decoder_width is None:
            width = torch.zeros(positions.shape[0], positions.shape[1], 1, device=positions.device)
            warnings.warn("Grasp width decoder is not initialized.", RuntimeWarning)
        else:
            width = self.decoder_width(positions, tsdf_features) # Pass positions and the feature dict

        # After quality prediction
        model_logger.info(f"[PrestoGIGA.forward_grasp] Quality output shape: {qual.shape}, "
              f"range: [{qual.min().item():.4f}, {qual.max().item():.4f}], "
              f"positive preds: {(qual > 0).float().mean().item():.4f}")
        
        # After rotation prediction
        model_logger.info(f"[PrestoGIGA.forward_grasp] Rotation output shape: {rot.shape}, "
              f"magnitude: {torch.norm(rot, dim=-1).mean().item():.4f}")
        
        # After width prediction
        model_logger.info(f"[PrestoGIGA.forward_grasp] Width output shape: {width.shape}, "
              f"range: [{width.min().item():.4f}, {width.max().item():.4f}]")
        
        return qual, rot, width
        
    def forward_tsdf(self, tsdf_features: Dict[str, torch.Tensor], positions: torch.Tensor):
        """
        Process features through TSDF output head using GIGA's decoder.

        Args:
            tsdf_features: Dictionary of encoded features from tsdf_encoder
                           (e.g., {'xy': ..., 'xz': ..., 'yz': ...}).
            positions: [B, M, 3] query positions for TSDF prediction.

        Returns:
            tsdf: [B, M, 1] TSDF values
        """
        model_logger.info(f"[PrestoGIGA.forward_tsdf] Input shapes: positions={positions.shape}")
        model_logger.info(f"[PrestoGIGA.forward_tsdf] tsdf_features keys: {list(tsdf_features.keys())}")
        print('here!!!!!')
        if self.decoder_tsdf is None:
            raise RuntimeError("TSDF decoder is not initialized.")

        # The GIGA decoder expects the encoded features 'c' directly.
        # The dictionary tsdf_features holds these features.
        tsdf = self.decoder_tsdf(positions, tsdf_features) # Pass positions and the feature dict

        # After TSDF prediction
        model_logger.info(f"[PrestoGIGA.forward_tsdf] TSDF output shape: {tsdf.shape}, "
              f"range: [{tsdf.min().item():.4f}, {tsdf.max().item():.4f}]")
        print(tsdf.shape, 'tsdf_FORWWARD')
        return tsdf
        
    def forward(self,
                sample: Optional[torch.FloatTensor] = None,
                timestep: Optional[Union[torch.Tensor, float, int]] = None,
                class_labels: Optional[torch.Tensor] = None,
                tsdf: Optional[torch.FloatTensor] = None,
                p: Optional[torch.FloatTensor] = None,
                p_tsdf: Optional[torch.FloatTensor] = None,
                mode: str = "joint", # "diffusion", "grasp", "tsdf", "joint"
                return_dict: bool = True):
        """
        Unified forward pass that supports diffusion, grasp, tsdf, or joint modes.

        Args:
            sample: [B, C, T] sequence data for diffusion input.
            timestep: Diffusion timesteps.
            class_labels: Conditioning variables (e.g., goal embedding).
            tsdf: [B, 1, D, D, D] TSDF voxel grid (input to encoder).
            p: [B, M, 3] query positions for grasp prediction.
            p_tsdf: [B, N, 3] query positions for TSDF prediction.
            mode: Specifies which parts of the model to run.
                  "diffusion": Only diffusion prediction.
                  "grasp": Only grasp prediction (requires tsdf and p).
                  "tsdf": Only TSDF prediction (requires tsdf and p_tsdf).
                  "joint": Runs diffusion and optionally grasp/tsdf if inputs provided.
            return_dict: Whether to return a dictionary.

        Returns:
            Dictionary with model outputs based on the mode.
        """
        model_logger.info(f"[PrestoGIGA.forward] Mode: {mode}, Input shapes: "
              f"sample={sample.shape if sample is not None else None}, "
              f"tsdf={tsdf.shape if tsdf is not None else None}, "
              f"p={p.shape if p is not None else None}, "
              f"p_tsdf={p_tsdf.shape if p_tsdf is not None else None}")
        
        # Safely get the use_amp setting from the config, default to None if not present
        use_amp_config = getattr(self.cfg, 'use_amp', False) # Default to False if not set
        amp_context = torch.cuda.amp.autocast(enabled=use_amp_config) if torch.cuda.is_available() else nullcontext()

        output = {}

        with amp_context:
            # 1. Encode inputs and get shared features
            # Pass diffusion inputs (sample, timestep, class_labels) and TSDF input
            # encode_shared returns transformer features, conditioning, and raw tsdf encoder output dict
            features, c, tsdf_features = self.encode_shared(x=sample, tsdf=tsdf, t=timestep, y=class_labels)

            # 2. Diffusion Prediction (if mode requires it)
            if mode in ["diffusion", "joint"] and sample is not None:
                # Process features through transformer blocks
                # Note: features might now include an extra token for tsdf_embed
                # Adjust FinalLayer input if necessary based on how features are structured
                num_extra_tokens = 1 if (tsdf is not None and self.cfg.use_joint_embeddings) else 0

                # Apply transformer blocks
                print(len(self.blocks), 'len(self.blocks)')
                for block in self.blocks:
                     features = block.forward(features, c) # features shape [B, N+num_extra, D]

                # Apply final layer for diffusion output
                # Pass only the sequence tokens (excluding potential TSDF token) to final_layer
                sequence_features = features[:, :-num_extra_tokens, :] if num_extra_tokens > 0 else features
                print(sequence_features.shape, 'sequence_features')
                # final_layer_output shape: [B, num_sequence_patches, patch_size * out_channels]
                final_layer_output = self.final_layer(sequence_features, c)
                print(final_layer_output.shape, 'final_layer_output')

                # --- Manually Unpatchify ---
                # B = batch size
                # N = num_sequence_patches
                # P = patch_size
                # C_out = output channels (self.out_dim, which is in_channels * 2 if learn_sigma else in_channels)
                B, N, _ = final_layer_output.shape
                P = self.x_embedder.patch_size
                C_out = self.out_channels # Get the output channel dimension defined in __init__

                # Reshape: [B, N, P * C_out] -> [B, N, P, C_out]
                x_reshaped = final_layer_output.view(B, N, P, C_out)
                print(x_reshaped.shape, 'x_reshaped')
                # Permute: [B, N, P, C_out] -> [B, C_out, N, P]
                x_permuted = x_reshaped.permute(0, 1, 2, 3)
                # Reshape: [B, C_out, N, P] -> [B, C_out, N * P] (where N * P = T, the original sequence length)
                diffusion_out_unpatchified = x_permuted.reshape(B, N * P, C_out)
                # --------------------------
                print(diffusion_out_unpatchified.shape, 'diffusion_out_unpatchified')
                output["diffusion_output"] = diffusion_out_unpatchified # Store the unpatchified output

            # 3. Grasp Prediction (if mode requires it and inputs available)
            if mode in ["grasp", "joint"] and tsdf_features is not None and p is not None:
                if not self.cfg.use_grasp:
                     model_logger.info("Warning: Grasp prediction requested but model cfg.use_grasp is False.")
                else:
                     qual, rot, width = self.forward_grasp(tsdf_features, p)
                     print(qual.shape, rot.shape, width.shape, 'qual, rot, width')
                     output["qual"] = qual
                     output["rot"] = rot
                     output["width"] = width

            # 4. TSDF Prediction (if mode requires it and inputs available)
          
            if mode in ["tsdf", "joint"] and tsdf_features is not None and p_tsdf is not None:
                 print('here!')
                 if not self.cfg.use_tsdf:
                     print('HERE!!!!')
                     model_logger.info("Warning: TSDF prediction requested but model cfg.use_tsdf is False.")
                 else:
                     tsdf_pred = self.forward_tsdf(tsdf_features, p_tsdf)
                     print(tsdf_pred.shape, 'tsdf_pred')
                     output["tsdf_pred"] = tsdf_pred

        # After encode_shared
        model_logger.info(f"[PrestoGIGA.forward] After encode_shared: "
              f"features={features.shape}, c={c.shape if c is not None else None}, "
              f"tsdf_features has {len(tsdf_features) if tsdf_features else 0} keys")
        
        # After transformer blocks
        model_logger.info(f"[PrestoGIGA.forward] After transformer blocks: features shape={features.shape}, "
              f"range: [{features.min().item():.4f}, {features.max().item():.4f}]")
        
        # After final layer
        if 'final_layer_output' in locals() and final_layer_output is not None:
            model_logger.info(f"[PrestoGIGA.forward] Final layer output shape: {final_layer_output.shape}, "
                  f"range: [{final_layer_output.min().item():.4f}, {final_layer_output.max().item():.4f}]")
        
        # After unpatchify
        if 'diffusion_out_unpatchified' in locals() and diffusion_out_unpatchified is not None:
            model_logger.info(f"[PrestoGIGA.forward] Diffusion output shape: {diffusion_out_unpatchified.shape}, "
                  f"range: [{diffusion_out_unpatchified.min().item():.4f}, {diffusion_out_unpatchified.max().item():.4f}]")
        
        # At the end, summary of output
        model_logger.info(f"[PrestoGIGA.forward] Output keys: {list(output.keys())}")
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                model_logger.info(f"[PrestoGIGA.forward] Output '{k}' shape: {v.shape}, "
                      f"range: [{v.min().item():.4f}, {v.max().item():.4f}]")
        
        if not return_dict:
            return tuple(output.values())
        else:
            print(output.keys(), "output")
            return output
    

    def grad_refine(self, tsdf, pos, bound_value=0.0125, lr=1e-6, num_step=1):
        """
        Gradient-based refinement of grasp positions using GIGA's approach
        
        Args:
            tsdf: [B, 1, D, D, D] TSDF voxel grid
            pos: [B, M, 3] initial positions
            bound_value: bound for position updates
            lr: learning rate
            num_step: number of optimization steps
            
        Returns:
            qual: [B, M, 1] refined grasp quality
            pos: [B, M, 3] refined positions
            rot: [B, M, 4] grasp rotation at refined positions
            width: [B, M, 1] grasp width at refined positions
        """
        pos_tmp = pos.clone()
        l_bound = pos - bound_value
        u_bound = pos + bound_value
        pos_tmp.requires_grad = True
        optimizer = torch.optim.SGD([pos_tmp], lr=lr)
        
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
            
        for _ in range(num_step):
            optimizer.zero_grad()
            results = self.forward(tsdf=tsdf, p=pos_tmp, mode="grasp")
            qual_out = results["qual"]
            loss = -qual_out.sum()
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            pos_tmp = torch.maximum(torch.minimum(pos_tmp, u_bound), l_bound)
            results = self.forward(tsdf=tsdf, p=pos_tmp, mode="grasp")
            qual_out = results["qual"]
            rot_out = results["rot"]
            width_out = results["width"]
            
        for p in self.parameters():
            p.requires_grad = True
            
        return qual_out, pos_tmp, rot_out, width_out
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
        
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def select_grasps(qual_vol, pos_vol, rot_vol, width_vol, threshold=0.9, max_filter_size=4):
    """
    Select grasps from volumes based on quality
    
    Args:
        qual_vol: [H, W, D] quality volume
        pos_vol: [H, W, D, 3] position volume
        rot_vol: [H, W, D, 4] rotation volume
        width_vol: [H, W, D, 1] width volume
        threshold: quality threshold
        max_filter_size: size of maximum filter
        
    Returns:
        grasps: List of selected grasp parameters
        scores: List of grasp qualities
    """
    from scipy import ndimage
    import numpy as np
    
    if torch.is_tensor(qual_vol):
        qual_vol = qual_vol.detach().cpu().numpy()
    if torch.is_tensor(pos_vol):
        pos_vol = pos_vol.detach().cpu().numpy()
    if torch.is_tensor(rot_vol):
        rot_vol = rot_vol.detach().cpu().numpy()
    if torch.is_tensor(width_vol):
        width_vol = width_vol.detach().cpu().numpy()
    
    mask = qual_vol > threshold
    
    if not np.any(mask):
        return [], []
    
    max_filtered = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    mask = (qual_vol >= max_filtered) & mask
    
    indices = np.where(mask)
    
    grasps = []
    scores = []
    
    for i, j, k in zip(*indices):
        pos = pos_vol[i, j, k]
        rot = rot_vol[i, j, k]
        width = width_vol[i, j, k, 0]
        qual = qual_vol[i, j, k]
        
        grasp = {
            'position': pos,
            'rotation': rot,
            'width': width
        }
        
        grasps.append(grasp)
        scores.append(qual)
    
    return grasps, scores


def predict_grasp_volumes(model, tsdf, resolution=40, device=None):
    """
    Predict grasp volumes from a TSDF
    
    Args:
        model: PrestoGIGA model
        tsdf: [1, 1, D, D, D] TSDF voxel grid
        resolution: resolution of output volumes
        device: device to run prediction on
        
    Returns:
        qual_vol: [H, W, D] quality volume
        rot_vol: [H, W, D, 4] rotation volume
        width_vol: [H, W, D] width volume
    """
    if device is None:
        device = model.device
        
    tsdf = tsdf.to(device)
    
    grid = torch.linspace(-0.5, 0.5, resolution, device=device)
    x, y, z = torch.meshgrid(grid, grid, grid, indexing='ij')
    query_points = torch.stack([x, y, z], dim=-1).view(1, -1, 3)
    
    with torch.no_grad():
        results = model.forward(tsdf=tsdf, p=query_points, mode="grasp")
        
    qual = results["qual"].view(resolution, resolution, resolution)
    rot = results["rot"].view(resolution, resolution, resolution, 4)
    width = results["width"].view(resolution, resolution, resolution, 1)
    
    return qual, rot, width, query_points.view(resolution, resolution, resolution, 3)
