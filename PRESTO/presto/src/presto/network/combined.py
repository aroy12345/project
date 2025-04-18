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
                 plane_resolution=512, grid_resolution=None, plane_type=['xz', 'xy', 'yz'], kernel_size=3, padding=0.1):
        super().__init__()
        self.actvn = F.relu
        if kernel_size == 1:
            self.conv_in = nn.Conv3d(1, c_dim, 1)
        else:
            self.conv_in = nn.Conv3d(1, c_dim, kernel_size, padding=1)

        if unet:
            # Convert unet_kwargs namespace to dict before unpacking
            unet_kwargs_dict = unet_kwargs.copy() # Use .copy() as it's already a dict
            self.unet = UNet(**unet_kwargs_dict) 
        else:
            self.unet = None

        if unet3d:
            print(unet3d_kwargs)
            # Also apply the fix here if unet3d_kwargs could be a namespace
            if unet3d_kwargs is not None:
                unet3d_kwargs_dict = vars(unet3d_kwargs)
            else:
                unet3d_kwargs_dict = {}
            self.unet3d = UNet3D(**unet3d_kwargs_dict)
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

        # Create coordinates for the GxGxG grid
        coord1 = torch.linspace(-0.5, 0.5, grid_res[0], device=device)
        coord2 = torch.linspace(-0.5, 0.5, grid_res[1], device=device)
        coord3 = torch.linspace(-0.5, 0.5, grid_res[2], device=device)

        # Create meshgrid and expand to batch size
        grid_coords = torch.stack(torch.meshgrid(coord1, coord2, coord3, indexing='ij'), dim=-1) # Shape [G, G, G, 3]
        p = grid_coords.unsqueeze(0).expand(batch_size, -1, -1, -1, -1) # Shape [B, G, G, G, 3]
        p = p.reshape(batch_size, grid_res[0] * grid_res[1] * grid_res[2], 3) # Shape [B, n_voxel, 3], e.g., [8, 32768, 3]

        # Acquire voxel-wise feature
        # conv_in expects [B, C_in, G, G, G] - input x is already in this format
        c = self.actvn(self.conv_in(x)) # Output shape [B, c_dim, G, G, G]
        # Flatten spatial dimensions G*G*G = n_voxel
        c = c.view(batch_size, self.c_dim, grid_res[0] * grid_res[1] * grid_res[2]) # Shape [B, c_dim, n_voxel], e.g., [8, 32, 32768]
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


# --- Base Config (Common Parameters) ---
@dataclass
class BaseModelConfig:
    hidden_size: int = 256
    padding: float = 0.1
    # GIGA/Encoder related
    encoder_type: Optional[str] = 'voxel_simple_local'
    encoder_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        'plane_type': ['xy', 'xz', 'yz'],
        'plane_resolution': 40,
        'grid_resolution': 32,
        'unet': True,
        'unet3d': True,
        'unet_kwargs': { 'depth': 3, 'merge_mode': 'concat', 'start_filts': 32 },
        'c_dim': 32 # Added c_dim here for consistency
    })
    c_dim: int = 32 # Feature dim per plane from encoder
    # Decoder related (used by GIGA, potentially indirectly by Presto via loaded GIGA)
    decoder_type: str = 'simple_local'
    decoder_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        'dim': 3, 'sample_mode': 'bilinear', 'hidden_size': 32, 'concat_feat': True
    })
    # Other shared params if any
    use_amp: Optional[bool] = None


# --- GIGA Model Config ---
@dataclass
class GIGAConfig(BaseModelConfig):
    # GIGA specific parameters (if any beyond BaseModelConfig)
    rot_dim: int = 4 # Example, adjust as needed
    use_tsdf_pred: bool = True # Control TSDF prediction head


# --- Presto Model Config ---
@dataclass
class PrestoConfig(BaseModelConfig):
    # Diffusion specific parameters
    input_size: int = 1000
    patch_size: int = 20
    in_channels: int = 7
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
    use_cloud: bool = False # Assuming this relates to input type, keep if needed
    use_joint_embeddings: bool = True # Keep for conditioning diffusion on TSDF


# --- GIGA Model ---
class GIGA(nn.Module):
    """
    GIGA model for grasp and TSDF affordance prediction.
    Uses a TSDF encoder and multiple decoder heads.
    """
    def __init__(self, config: GIGAConfig, **kwargs):
        super().__init__()
        self.cfg = config
        self._init_encoder()
        self._init_decoders()
        self.apply(self._init_weights) # Apply weight initialization

    def _init_encoder(self):
        """Initialize the TSDF encoder."""
        cfg = self.cfg
        if cfg.encoder_type:
            encoder_cls = encoder_dict[cfg.encoder_type]
            # Ensure c_dim is passed correctly if needed by the encoder's kwargs handling
            encoder_kwargs = cfg.encoder_kwargs.copy() # Use .copy() to avoid modifying the original config dict
            if 'c_dim' not in encoder_kwargs:
                 encoder_kwargs['c_dim'] = cfg.c_dim # Pass c_dim if not already in kwargs
            self.tsdf_encoder = encoder_cls(
                padding=cfg.padding,
                **encoder_kwargs
            )
            print(f"Initialized GIGA TSDF Encoder: {cfg.encoder_type}")
        else:
            self.tsdf_encoder = None
            print("Warning: GIGA model initialized without a TSDF encoder.")

    def _init_decoders(self):
        """Initialize the grasp and TSDF decoders."""
        cfg = self.cfg
        self.decoder_qual = None
        self.decoder_rot = None
        self.decoder_width = None
        self.decoder_tsdf = None

        if cfg.decoder_type and self.tsdf_encoder is not None: # Need encoder features
            decoder_cls = decoder_dict[cfg.decoder_type]
            # Use encoder's output feature dimension (c_dim) for decoder input
            # Ensure decoder_kwargs correctly reflects the expected c_dim from the encoder
            decoder_c_dim = cfg.c_dim # Assuming encoder output dim matches cfg.c_dim
            common_decoder_kwargs = {
                'dim': 3, # Spatial dimension
                'c_dim': decoder_c_dim,
                **cfg.decoder_kwargs.copy() # Use .copy() here as well if modifications might occur
            }
            # Update c_dim in common_decoder_kwargs explicitly if it was changed in encoder_kwargs
            common_decoder_kwargs['c_dim'] = self.tsdf_encoder.c_dim if hasattr(self.tsdf_encoder, 'c_dim') else cfg.c_dim


            # Grasp Decoders
            self.decoder_qual = decoder_cls(out_dim=1, **common_decoder_kwargs)
            self.decoder_rot = decoder_cls(out_dim=cfg.rot_dim, **common_decoder_kwargs)
            self.decoder_width = decoder_cls(out_dim=1, **common_decoder_kwargs)
            print(f"Initialized GIGA Grasp Decoders (Qual, Rot, Width) type: {cfg.decoder_type}")

            # Optional TSDF Decoder
            if cfg.use_tsdf_pred:
                self.decoder_tsdf = decoder_cls(out_dim=1, **common_decoder_kwargs)
                print(f"Initialized GIGA TSDF Decoder type: {cfg.decoder_type}")
        elif not self.tsdf_encoder:
             print("Warning: GIGA decoders not initialized because TSDF encoder is missing.")
        else:
             print(f"Warning: GIGA decoder type '{cfg.decoder_type}' not found or specified.")


    def _init_weights(self, m):
        """Initialize weights (same as original PrestoGIGA)."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)): # Include Conv3d if encoder uses it
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
             if m.bias is not None:
                 nn.init.constant_(m.bias, 0)
             if m.weight is not None:
                 nn.init.constant_(m.weight, 1.0)
        # Initialize DiTBlock parameters if needed (often done within the block itself)

    def encode_tsdf(self, tsdf: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encodes the input TSDF volume using the tsdf_encoder.

        Args:
            tsdf: [B, 1, D, D, D] TSDF voxel grid.

        Returns:
            A dictionary containing the encoded features (e.g., feature planes).
            Returns an empty dictionary if no encoder is present.
        """
        if self.tsdf_encoder is not None:
            tsdf_features = self.tsdf_encoder(tsdf)
            print(f"[GIGA.encode_tsdf] Input TSDF shape: {tsdf.shape}")
            for k, v in tsdf_features.items():
                 if isinstance(v, torch.Tensor):
                     print(f"[GIGA.encode_tsdf] Output feature '{k}' shape: {v.shape}")
            return tsdf_features
        else:
            print("[GIGA.encode_tsdf] No TSDF encoder available.")
            return {}

    def forward_affordance(self,
                           tsdf_features: Dict[str, torch.Tensor],
                           p_grasp: Optional[torch.Tensor] = None,
                           p_tsdf: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predicts grasp and/or TSDF affordances from encoded features and query points.

        Args:
            tsdf_features: Dictionary of encoded features from encode_tsdf.
            p_grasp: [B, M, 3] query positions for grasp prediction.
            p_tsdf: [B, N, 3] query positions for TSDF prediction.

        Returns:
            A dictionary containing the predicted affordances ('qual', 'rot', 'width', 'tsdf_pred').
        """
        output = {}

        if not tsdf_features:
            print("[GIGA.forward_affordance] Cannot predict affordances without TSDF features.")
            return output # Return empty if no features

        # Grasp predictions
        if p_grasp is not None:
            print(f"[GIGA.forward_affordance] Predicting grasp affordances for points shape: {p_grasp.shape}")
            if self.decoder_qual is not None:
                output['qual'] = self.decoder_qual(p_grasp, tsdf_features)
                print(f"[GIGA.forward_affordance] Predicted qual shape: {output['qual'].shape}")
            else:
                print("[GIGA.forward_affordance] Grasp quality decoder not available.")

            if self.decoder_rot is not None:
                output['rot'] = self.decoder_rot(p_grasp, tsdf_features)
                print(f"[GIGA.forward_affordance] Predicted rot shape: {output['rot'].shape}")
            else:
                print("[GIGA.forward_affordance] Grasp rotation decoder not available.")

            if self.decoder_width is not None:
                output['width'] = self.decoder_width(p_grasp, tsdf_features)
                print(f"[GIGA.forward_affordance] Predicted width shape: {output['width'].shape}")
            else:
                print("[GIGA.forward_affordance] Grasp width decoder not available.")

        # TSDF prediction
        if p_tsdf is not None and self.cfg.use_tsdf_pred:
            print(f"[GIGA.forward_affordance] Predicting TSDF for points shape: {p_tsdf.shape}")
            if self.decoder_tsdf is not None:
                output['tsdf_pred'] = self.decoder_tsdf(p_tsdf, tsdf_features)
                print(f"[GIGA.forward_affordance] Predicted tsdf_pred shape: {output['tsdf_pred'].shape}")
            else:
                print("[GIGA.forward_affordance] TSDF prediction decoder not available.")
        elif p_tsdf is not None and not self.cfg.use_tsdf_pred:
             print("[GIGA.forward_affordance] TSDF prediction requested but use_tsdf_pred is False in config.")

        print(self.decoder_qual, self.decoder_rot, self.decoder_width, self.decoder_tsdf)
        return output

    def forward(self,
                tsdf: torch.Tensor,
                p_grasp: Optional[torch.Tensor] = None,
                p_tsdf: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Main forward pass for the GIGA model. Encodes TSDF and predicts affordances.

        Args:
            tsdf: [B, 1, D, D, D] Input TSDF volume.
            p_grasp: [B, M, 3] Query points for grasp prediction.
            p_tsdf: [B, N, 3] Query points for TSDF prediction.

        Returns:
            Dictionary with affordance predictions.
        """
        tsdf_features = self.encode_tsdf(tsdf)
        affordance_output = self.forward_affordance(tsdf_features, p_grasp, p_tsdf)

        print(f"[GIGA.forward] Final output shape: {affordance_output['qual'].shape}")
        print(f"[GIGA.forward] Final output shape: {affordance_output['rot'].shape}")
        print(f"[GIGA.forward] Final output shape: {affordance_output['width'].shape}")
        print(f"[GIGA.forward] Final output shape: {affordance_output['tsdf_pred'].shape}")
        return affordance_output

    @property
    def device(self):
        # Safely get device from encoder or first decoder
        if self.tsdf_encoder is not None:
            return next(iter(self.tsdf_encoder.parameters())).device
        elif self.decoder_qual is not None:
            return next(iter(self.decoder_qual.parameters())).device
        else:
            # Fallback or raise error if no parameters
            try:
                return next(iter(self.parameters())).device
            except StopIteration:
                 print("Warning: GIGA model has no parameters to determine device.")
                 return torch.device("cpu") # Default fallback


# --- Presto Model ---
class Presto(nn.Module):
    """
    Presto diffusion model for trajectory prediction, conditioned on TSDF embedding.
    """
    def __init__(self, cfg: PrestoConfig):
        super().__init__()
        self.cfg = cfg

        # --- Initialize Diffusion Components ---
        self.learn_sigma = cfg.learn_sigma
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.in_channels * 2 if cfg.learn_sigma else cfg.in_channels
        self.patch_size = cfg.patch_size
        self.num_heads = cfg.num_heads
        self.use_cond = cfg.use_cond
        self.use_cond_token = cfg.use_cond_token # Keep if DiTBlock uses it

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
        else:
            self.y_embedder = None # Handle case where conditioning is off

        num_patches = self.x_embedder.num_patches
        # Initialize pos_embed buffer (will be filled later or loaded)
        self.register_buffer('pos_embed', torch.zeros(1, num_patches, cfg.hidden_size), persistent=False)


        # --- Initialize TSDF Encoder and Embedder (for conditioning) ---
        self.tsdf_encoder = None
        self.tsdf_embedder = None
        if cfg.encoder_type:
            encoder_cls = encoder_dict[cfg.encoder_type]
            # Ensure c_dim is passed correctly if needed by the encoder's kwargs handling
            encoder_kwargs = cfg.encoder_kwargs.copy() # Use .copy() to avoid modifying the original config dict
            if 'c_dim' not in encoder_kwargs:
                 encoder_kwargs['c_dim'] = cfg.c_dim # Pass c_dim if not already in kwargs
            self.tsdf_encoder = encoder_cls(
                padding=cfg.padding,
                **encoder_kwargs
            )
            # Determine the input resolution for the TSDF embedder from encoder config
            # Default to a reasonable value if not found
            tsdf_embedder_resolution = cfg.encoder_kwargs.get('plane_resolution', 32)
            self.tsdf_embedder = TSDFEmbedder(
                hidden_size=cfg.hidden_size,
                c_dim=cfg.c_dim, # Use the base c_dim config
                input_resolution=tsdf_embedder_resolution
            )
            print(f"Initialized Presto TSDF Encoder: {cfg.encoder_type}")
            print(f"Initialized Presto TSDF Embedder with input resolution: {tsdf_embedder_resolution}")
        else:
            print("Warning: Presto model initialized without TSDF encoder/embedder. Cannot use TSDF conditioning.")


        # --- Initialize Transformer Blocks ---
        self.blocks = nn.ModuleList([
            DiTBlock(
                cfg.hidden_size, cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                use_cond=True # DiTBlock expects conditioning signal 'c'
            )
            for _ in range(cfg.num_layer)
        ])

        # --- Initialize Final Layer ---
        self.final_layer = FinalLayer(cfg.hidden_size, cfg.patch_size, self.out_channels)

        self.apply(self._init_weights) # Apply weight initialization
        self._initialize_pos_embed() # Initialize positional embedding weights


    def _init_weights(self, m):
        """Initialize weights (same as original PrestoGIGA)."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)): # Include Conv3d if encoder uses it
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
             if m.bias is not None:
                 nn.init.constant_(m.bias, 0)
             if m.weight is not None:
                 nn.init.constant_(m.weight, 1.0)
        # Initialize DiTBlock parameters if needed (often done within the block itself)


    def _initialize_pos_embed(self):
        """Initialize the positional embedding buffer."""
        if hasattr(self, 'pos_embed') and self.pos_embed is not None:
             print("Initializing Presto positional embeddings...")
             pos_embed_1d = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.num_patches)
             self.pos_embed.data.copy_(torch.from_numpy(pos_embed_1d).float().unsqueeze(0))
             print(f"Positional embedding initialized with shape: {self.pos_embed.shape}")


    def encode_shared(self,
                      x: Optional[torch.Tensor] = None,
                      tsdf: Optional[torch.Tensor] = None,
                      t: Optional[torch.Tensor] = None,
                      y: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Encodes inputs (sequence, TSDF, time, condition) into features for DiT blocks.

        Args:
            x: [B, C_in, T] sequence data.
            tsdf: [B, 1, D, D, D] TSDF voxel grid.
            t: [B] diffusion timesteps.
            y: [B, cond_dim] conditioning variables.

        Returns:
            features: [B, N_seq + N_tsdf_token, D_hidden] Combined features for DiT blocks.
            c: [B, D_hidden] Combined conditioning vector for DiT blocks.
            tsdf_features: Dictionary from tsdf_encoder (passed through for potential use).
        """
        print(f"[Presto.encode_shared] Input shapes: x={x.shape if x is not None else None}, "
              f"tsdf={tsdf.shape if tsdf is not None else None}, "
              f"t={t.shape if t is not None else None}, "
              f"y={y.shape if y is not None else None}")

        # 1. Sequence Embedding (if x is provided)
        x_embed = None
        if x is not None:
            # Ensure pos_embed is correctly initialized and on the right device
            if self.pos_embed is None or self.pos_embed.shape[1] != self.x_embedder.num_patches:
                 print("Re-initializing positional embedding in encode_shared.")
                 self._initialize_pos_embed() # Re-initialize if needed

            # Add positional embedding
            # Ensure pos_embed is on the same device as x
            current_pos_embed = self.pos_embed.to(x.device)
            x_embed = self.x_embedder(x) + current_pos_embed
            print(f"[Presto.encode_shared] x_embed shape: {x_embed.shape}")
        else:
             print("[Presto.encode_shared] No sequence input (x) provided.")


        # 2. TSDF Embedding (if tsdf is provided and encoder/embedder exist)
        tsdf_embed_token = None
        tsdf_features = None
        if tsdf is not None and self.tsdf_encoder is not None and self.tsdf_embedder is not None:
            print("[Presto.encode_shared] Encoding TSDF...")
            tsdf_features = self.tsdf_encoder(tsdf) # Dict of planes, e.g., {'xy': [B,C,H,W], ...}
            print(f"[Presto.encode_shared] TSDF encoder output keys: {list(tsdf_features.keys())}")
            # Pass the dictionary directly to the embedder
            tsdf_embed = self.tsdf_embedder(tsdf_features) # Outputs [B, hidden_size]
            tsdf_embed_token = tsdf_embed.unsqueeze(1) # Shape: [B, 1, hidden_size]
            print(f"[Presto.encode_shared] tsdf_embed_token shape: {tsdf_embed_token.shape}")
        elif tsdf is not None:
            print("[Presto.encode_shared] TSDF input provided but encoder/embedder missing.")


        # 3. Combine Features for Transformer
        features = None
        num_extra_tokens = 0
        if x_embed is not None and tsdf_embed_token is not None and self.cfg.use_joint_embeddings:
            # Concatenate sequence embedding and the single TSDF embedding token
            features = torch.cat([x_embed, tsdf_embed_token], dim=1)
            num_extra_tokens = 1
            print(f"[Presto.encode_shared] Concatenated x_embed and tsdf_embed_token. Features shape: {features.shape}")
        elif x_embed is not None:
            features = x_embed
            print(f"[Presto.encode_shared] Using only x_embed. Features shape: {features.shape}")
        elif tsdf_embed_token is not None:
            # Handle TSDF-only input if necessary for the architecture.
            # This might require adjustments depending on how DiT blocks expect input.
            # For now, we assume 'x' is usually present for diffusion.
            print("[Presto.encode_shared] Warning: Only TSDF input provided to encode_shared. Feature handling might need review.")
            # If needed, replicate tsdf_embed_token and add pos encoding, similar to original PrestoGIGA
            # features = tsdf_embed_token.expand(-1, self.x_embedder.num_patches, -1) + self.pos_embed.to(tsdf_embed_token.device)
            features = tsdf_embed_token # Or maybe just pass the single token? Depends on DiT. Assuming x is required.
        else:
            # Raise error or handle case with no input features
             print("[Presto.encode_shared] Error: No input features (x or tsdf) to create transformer input.")
             # return None, None, None # Or raise error


        # 4. Conditioning Signal 'c'
        t_embed = self.t_embedder(t) if t is not None else None
        y_embed = self.y_embedder(y, train=self.training) if y is not None and self.y_embedder is not None else None

        print(f"[Presto.encode_shared] t_embed shape: {t_embed.shape if t_embed is not None else None}")
        print(f"[Presto.encode_shared] y_embed shape: {y_embed.shape if y_embed is not None else None}")

        # Combine conditioning embeddings
        if t_embed is not None and y_embed is not None:
            c = t_embed + y_embed
        elif t_embed is not None:
            c = t_embed
        elif y_embed is not None:
            c = y_embed
        else:
            # If no time or class conditioning, create a zero embedding or handle as needed
            # Ensure 'c' has the correct shape [B, hidden_size] if blocks expect it
            if features is not None: # Need batch size info
                 c = torch.zeros(features.shape[0], self.cfg.hidden_size, device=self.device, dtype=features.dtype)
                 print("[Presto.encode_shared] No t or y conditioning provided, using zero vector for c.")
            else:
                 c = None # Cannot determine batch size
                 print("[Presto.encode_shared] No t or y conditioning, and no features to determine batch size for zero vector c.")


        print(f"[Presto.encode_shared] Final 'features' shape: {features.shape if features is not None else None}")
        print(f"[Presto.encode_shared] Final 'c' shape: {c.shape if c is not None else None}")

        # Return tsdf_features as well, might be useful for debugging or other purposes
        return features, c, tsdf_features


    def forward_diffusion(self,
                          features: torch.Tensor,
                          c: torch.Tensor) -> torch.Tensor:
        """
        Processes features through the DiT blocks and final layer for diffusion prediction.

        Args:
            features: [B, N_seq (+ N_tsdf_token), D_hidden] Input features from encode_shared.
            c: [B, D_hidden] Conditioning vector from encode_shared.

        Returns:
            diffusion_out: [B, T, C_out] Unpatchified diffusion model output.
        """
        print(f"[Presto.forward_diffusion] Input features shape: {features.shape}, c shape: {c.shape}")

        # Determine if the extra TSDF token is present
        num_expected_seq_patches = self.x_embedder.num_patches
        num_extra_tokens = features.shape[1] - num_expected_seq_patches
        if num_extra_tokens not in [0, 1]: # Allow 0 or 1 extra token
             print(f"[Presto.forward_diffusion] Warning: Unexpected number of tokens ({features.shape[1]}) vs expected sequence patches ({num_expected_seq_patches}). Num extra: {num_extra_tokens}")
             # Adjust logic if more complex token structures are possible
             num_extra_tokens = max(0, num_extra_tokens) # Assume extra tokens are at the end


        # Apply transformer blocks
        print(f"[Presto.forward_diffusion] Applying {len(self.blocks)} DiT blocks...")
        h = features # Use 'h' for hidden state through blocks
        for i, block in enumerate(self.blocks):
            h = block.forward(h, c) # Pass features and conditioning
            print(f"[Presto.forward_diffusion] After block {i}, h shape: {h.shape}")


        # Apply final layer for diffusion output
        # Pass only the sequence tokens (excluding potential TSDF token) to final_layer
        if num_extra_tokens > 0:
            sequence_features = h[:, :-num_extra_tokens, :]
            print(f"[Presto.forward_diffusion] Extracted sequence features shape: {sequence_features.shape} (removed {num_extra_tokens} extra tokens)")
        else:
            sequence_features = h
            print(f"[Presto.forward_diffusion] Using all features as sequence features shape: {sequence_features.shape}")

        # final_layer_output shape: [B, num_sequence_patches, patch_size * out_channels]
        final_layer_output = self.final_layer(sequence_features, c)
        print(f"[Presto.forward_diffusion] Final layer output shape: {final_layer_output.shape}")


        # --- Manually Unpatchify ---
        B, N_seq_patches, _ = final_layer_output.shape
        P = self.patch_size
        C_out = self.out_channels # Output channels (e.g., 7 for sample, 14 for epsilon+variance)
        T_out = N_seq_patches * P # Expected total output sequence length

        # Reshape: [B, N_seq, P * C_out] -> [B, N_seq, P, C_out]
        x_reshaped = final_layer_output.view(B, N_seq_patches, P, C_out)
        # Permute: [B, N_seq, P, C_out] -> [B, C_out, N_seq, P]
        x_permuted = x_reshaped.permute(0, 3, 1, 2) # Correct permutation: B, C_out, N_seq, P
        # Reshape: [B, C_out, N_seq, P] -> [B, C_out, N_seq * P] = [B, C_out, T_out]
        diffusion_out_unpatchified = x_permuted.reshape(B, C_out, T_out)
        # Transpose to match typical trajectory format [B, T, C_out]
        diffusion_out = diffusion_out_unpatchified.permute(0, 2, 1)
        # --------------------------
        print(f"[Presto.forward_diffusion] Unpatchified output shape: {diffusion_out.shape}")

        return diffusion_out


    def forward(self,
                sample: torch.FloatTensor,
                timestep: Union[torch.Tensor, float, int],
                class_labels: Optional[torch.Tensor] = None,
                tsdf: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        """
        Main forward pass for the Presto model.

        Args:
            sample: [B, C_in, T] Input noisy sequence data.
            timestep: Diffusion timesteps.
            class_labels: Conditioning variables.
            tsdf: [B, 1, D, D, D] TSDF voxel grid for conditioning.

        Returns:
            torch.Tensor: The predicted diffusion output (e.g., noise or x0)
                          Shape: [B, T, C_out]
        """
        # 1. Encode inputs to get features and conditioning
        features, c, _ = self.encode_shared(x=sample, tsdf=tsdf, t=timestep, y=class_labels)

        if features is None or c is None:
             raise ValueError("Failed to generate features or conditioning in encode_shared.")

        # 2. Process through DiT blocks and final layer
        diffusion_output = self.forward_diffusion(features, c)

        print(f"[Presto.forward] Final output shape: {diffusion_output.shape}")
        return diffusion_output

    @property
    def device(self):
        return next(iter(self.parameters())).device

    # Keep unpatchify method if needed elsewhere, but forward_diffusion handles it now
    # def unpatchify(self, x): ...

    # Keep set_pos_embed if needed for external initialization, but _initialize_pos_embed handles internal setup
    # def set_pos_embed(self, grid_size): ...
