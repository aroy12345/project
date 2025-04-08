#!/usr/bin/env python3

from typing import Union, Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, asdict, field
from contextlib import nullcontext
import time

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

from presto.ConvONets.conv_onet.models import (
    decoder_dict, decoder, ConvolutionalOccupancyNetwork, 
    ConvolutionalOccupancyNetworkGeometry
)
from presto.ConvONets.encoder import encoder_dict


class TSDFEmbedder(nn.Module):
    """Encoder for TSDF voxel grids from GIGA"""
    def __init__(self, hidden_size: int = 256, c_dim: int = 32):
        super().__init__()
        # Input to this embedder is the output of tsdf_encoder, which has c_dim channels
        self.conv1 = nn.Conv3d(c_dim, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(64, c_dim, 3, stride=2, padding=1)

        # Calculate the flattened size after convolutions (assuming input 32x32x32 -> 2x2x2)
        # If the encoder output spatial dim changes, this needs update.
        flat_size = c_dim * 2 * 2 * 2
        self.proj = nn.Linear(flat_size, hidden_size)
        self.c_dim = c_dim
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        return x


class FeatureExtractor(nn.Module):
    """Module to extract features at 3D positions from transformer features"""
    def __init__(self, hidden_size: int, output_dim: int = None, use_mlp: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim or hidden_size
        self.use_mlp = use_mlp
        
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, self.output_dim)
            )
            
    def forward(self, features, positions, grid_size=40):
        """
        Extract features at arbitrary 3D positions from transformer features
        
        Args:
            features: [B, N, D] transformer features
            positions: [B, M, 3] query positions in normalized coordinates [-0.5, 0.5]
            grid_size: size of the voxel grid
        
        Returns:
            [B, M, output_dim] features at query positions
        """
        batch_size, num_patches, hidden_size = features.shape
        batch_size_q, num_queries, _ = positions.shape
        
        positions = (positions + 0.5) * grid_size
        
        positions = positions.long()
        
        flat_indices = positions[:, :, 0] * grid_size * grid_size + positions[:, :, 1] * grid_size + positions[:, :, 2]
        
        flat_indices = torch.clamp(flat_indices, 0, num_patches - 1)
        
        batch_indices = torch.arange(batch_size_q, device=features.device)[:, None].expand(-1, num_queries)
        extracted_features = features[batch_indices.flatten(), flat_indices.flatten()].view(batch_size_q, num_queries, hidden_size)
        
        if self.use_mlp:
            extracted_features = self.mlp(extracted_features)
            
        return extracted_features


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
        
        decoder_type: str = 'simple_fc'
        decoder_kwargs: Dict[str, Any] = field(default_factory=lambda: {
            'hidden_size': 32,
            'n_blocks': 5,
            'leaky': False,
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
        
        self.feature_extractor = FeatureExtractor(cfg.hidden_size, output_dim=cfg.c_dim)
        
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
            tsdf: [B, 1, D, D, D] TSDF voxel grid
            t: diffusion timesteps
            y: conditioning variables
            
        Returns:
            features: [B, N, D] transformer features
            c: [B, D] conditioning embedding
        """
        if x is not None:
            x_embed = self.x_embedder(x) + self.pos_embed
        else:
            x_embed = None
            
        if tsdf is not None:
            if self.tsdf_encoder is not None:
                tsdf_features = self.tsdf_encoder(tsdf)
                if 'grid' in tsdf_features:
                    tsdf_tensor = tsdf_features['grid']
                else:
                    raise KeyError("Expected 'grid' key in tsdf_encoder output, "
                                   f"but found keys: {tsdf_features.keys()}")
            else:
                tsdf_tensor = tsdf
            
            tsdf_embed = self.tsdf_embedder(tsdf_tensor)
            tsdf_embed = tsdf_embed.unsqueeze(1)
        else:
            tsdf_embed = None
            
        if x_embed is not None and tsdf_embed is not None and self.cfg.use_joint_embeddings:
            features = torch.cat([x_embed, tsdf_embed], dim=1)
        elif x_embed is not None:
            features = x_embed
        elif tsdf_embed is not None:
            features = tsdf_embed.expand(-1, self.x_embedder.num_patches, -1)
            features = features + self.pos_embed
        else:
            raise ValueError("Either x or tsdf must be provided")
            
        if t is not None:
            t_embed = self.t_embedder(t)
            
            if self.use_cond and y is not None:
                y_embed = self.y_embedder(y, self.training)
                c = t_embed + y_embed
            else:
                c = t_embed
        else:
            c = None

        if x_embed is not None:
            features = x_embed
        elif tsdf_embed is not None:
            features = tsdf_embed.expand(-1, self.x_embedder.num_patches, -1)
            features = features + self.pos_embed
        else:
            raise ValueError("Either x or tsdf must be provided")

        if c is not None and tsdf_embed is not None:
            c = c + tsdf_embed.squeeze(1)
        elif c is None and tsdf_embed is not None:
            c = tsdf_embed.squeeze(1)

        for block in self.blocks:
            features = block(features, c)

        return features, c, tsdf_features
        
    def forward_diffusion(self, features, c):
        """
        Process features through diffusion output head using PRESTO's components
        
        Args:
            features: [B, N, D] transformer features
            c: [B, D] conditioning embedding
            
        Returns:
            [B, C, T] diffusion output
        """
        x = self.final_layer(features, c)
        
        if self.use_cond_token:
            x = x[..., :-1, :]
            
        x = self.unpatchify(x)
        return x
        
    def forward_grasp(self, p, tsdf_features):
        """
        Predict grasp affordance using the GIGA decoder heads.

        Args:
            p (torch.Tensor): Query points [B, M, 3].
            tsdf_features (dict): Feature dictionary from the tsdf_encoder (e.g., {'grid': tensor}).

        Returns:
            qual, rot, width tensors.
        """
        batch_size = p.size(0)
        num_queries = p.size(1)

        # --- MODIFIED DECODER CALLS ---
        # Call each specific decoder head
        if self.decoder_qual is not None:
            qual = self.decoder_qual(p, tsdf_features) # [B, M, 1]
        else:
            qual = torch.zeros(batch_size, num_queries, 1, device=p.device)
            warnings.warn("Grasp quality decoder is not initialized.", RuntimeWarning)

        if self.decoder_rot is not None:
            rot = self.decoder_rot(p, tsdf_features) # [B, M, rot_dim]
        else:
            rot = torch.zeros(batch_size, num_queries, self.rot_dim, device=p.device)
            warnings.warn("Grasp rotation decoder is not initialized.", RuntimeWarning)

        if self.decoder_width is not None:
            width = self.decoder_width(p, tsdf_features) # [B, M, 1]
        else:
            width = torch.zeros(batch_size, num_queries, 1, device=p.device)
            warnings.warn("Grasp width decoder is not initialized.", RuntimeWarning)
        # --- END MODIFIED DECODER CALLS ---

        # Ensure outputs have the expected last dimension if necessary
        # (Decoders might already output [B, M, 1] or [B, M, rot_dim])
        # Example: if qual is [B, M], uncomment below
        # if qual.dim() == 2: qual = qual.unsqueeze(-1)
        # if width.dim() == 2: width = width.unsqueeze(-1)

        return qual, rot, width
        
    def forward_tsdf(self, features, positions):
        """
        Process features through TSDF output head using GIGA's decoder
        
        Args:
            features: [B, N, D] transformer features
            positions: [B, M, 3] query positions
            
        Returns:
            tsdf: [B, M, 1] TSDF values
        """
        point_features = self.feature_extractor(
            features, positions, self.cfg.grid_size)
        
        batch_size = point_features.shape[0]
        c = {
            'grid_feat': point_features.reshape(batch_size, self.cfg.c_dim, 1, 1, 1)
        }
        
        tsdf = self.decoder_tsdf(positions, c)
        
        return tsdf
        
    def forward(self, 
                sample: Optional[torch.FloatTensor] = None,
                timestep: Optional[Union[torch.Tensor, float, int]] = None,
                class_labels: Optional[torch.Tensor] = None,
                tsdf: Optional[torch.FloatTensor] = None,
                p: Optional[torch.FloatTensor] = None,
                p_tsdf: Optional[torch.FloatTensor] = None,
                mode: str = "joint",
                return_dict: bool = True):
        """
        Unified forward pass that supports diffusion, grasp, or joint modes
        
        Args:
            sample: [B, C, T] sequence data for diffusion
            timestep: diffusion timesteps
            class_labels: conditioning variables
            tsdf: [B, 1, D, D, D] TSDF voxel grid
            p: [B, M, 3] query positions for grasp prediction
            p_tsdf: [B, M, 3] query positions for TSDF prediction
            mode: one of "diffusion", "grasp", "joint"
            return_dict: whether to return a dictionary
            
        Returns:
            Dictionary with model outputs
        """
        # Safely get the use_amp setting from the config, default to None if not present
        use_amp_config = getattr(self.cfg, 'use_amp', None)

        amp_ctx = (nullcontext() if (use_amp_config is None)
                  else torch.cuda.amp.autocast(enabled=use_amp_config))
                  
        with amp_ctx:
            if timestep is not None:
                if not torch.is_tensor(timestep):
                    timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
                elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
                    pass
                timestep = timestep.to(sample.device if sample is not None else tsdf.device)
            
            features, c, tsdf_features = self.encode_shared(
                x=sample, tsdf=tsdf, t=timestep, y=class_labels)
            
            results = {}
            
            if mode in ["diffusion", "joint"] and sample is not None:
                diffusion_out = self.forward_diffusion(features, c)
                results["sample"] = diffusion_out
            
            if mode in ["grasp", "joint"] and p is not None:
                qual, rot, width = self.forward_grasp(p, tsdf_features)
                results["qual"] = qual
                results["rot"] = rot
                results["width"] = width
            
            if mode in ["grasp", "joint"] and p_tsdf is not None and self.cfg.use_tsdf:
                tsdf_out = self.forward_tsdf(features, p_tsdf)
                results["tsdf"] = tsdf_out
                
            if not return_dict:
                # If not returning dict, the original code returned the dict anyway.
                # We'll keep returning the full dict for consistency.
                # If a tuple output is desired for return_dict=False, this needs adjustment.
                return results

            # --- MODIFIED RETURN LOGIC ---
            # Always return the full results dictionary if return_dict is True,
            # making all computed outputs accessible.
            return results
            # --- END MODIFICATION ---

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


