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
             print(f"Warning: Input plane shape ({B},{C},{H},{W}) doesn't match expected c_dim ({self.c_dim}) or resolution ({self.input_resolution}).")
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
            tsdf: [B, 1, D, D, D] TSDF voxel grid (input to encoder)
            t: diffusion timesteps
            y: conditioning variables

        Returns:
            features: [B, N, D] transformer features
            c: [B, D] conditioning embedding
            tsdf_features: Dictionary containing output features from tsdf_encoder (e.g., {'xy': ..., 'xz': ..., 'yz': ...})
                           or None if tsdf is None.
        """
        if x is not None:
            # Ensure pos_embed is initialized if needed (moved from original code block)
            if not hasattr(self, 'pos_embed') or self.pos_embed is None:
                 self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, self.cfg.hidden_size), requires_grad=False)
                 pos_embed_1d = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.num_patches)
                 self.pos_embed.data.copy_(torch.from_numpy(pos_embed_1d).float().unsqueeze(0))

            x_embed = self.x_embedder(x) + self.pos_embed
        else:
            x_embed = None

        tsdf_embed = None
        tsdf_features = None # Initialize tsdf_features

        if tsdf is not None:
            if self.tsdf_encoder is not None:
                # tsdf_encoder now returns a dictionary of features (planes or grid)
                tsdf_features = self.tsdf_encoder(tsdf) # e.g., {'xy': [B,C,H,W], 'xz': [B,C,H,W], 'yz': [B,C,H,W]}

                # Pass the dictionary directly to the embedder
                tsdf_embed = self.tsdf_embedder(tsdf_features) # Expects dict, outputs [B, hidden_size]
                tsdf_embed = tsdf_embed.unsqueeze(1) # Shape: [B, 1, hidden_size]
            else:
                # Handle case where there's no encoder but tsdf is provided (unlikely for this setup)
                print("Warning: TSDF provided but no tsdf_encoder defined.")
                # tsdf_embed remains None

        # Create combined features for the transformer blocks
        if x_embed is not None and tsdf_embed is not None and self.cfg.use_joint_embeddings:
            # Concatenate sequence embedding and the single TSDF embedding token
            print('here')
            features = torch.cat([x_embed, tsdf_embed], dim=1)
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

        # Combine conditioning embeddings (handle None cases)
        if t_embed is not None and y_embed is not None:
            c = t_embed + y_embed
        elif t_embed is not None:
            c = t_embed
        elif y_embed is not None:
            c = y_embed
        else:
            # If no time or class conditioning, create a zero embedding or handle as needed
            c = torch.zeros(features.shape[0], self.cfg.hidden_size, device=features.device, dtype=features.dtype)
            # Alternatively, could raise an error if conditioning is always expected

        # Return the features for transformer blocks, conditioning, and raw encoder output
        return features, c, tsdf_features # Return tsdf_features dict
        
    def forward_diffusion(self, features, c):
        """
        Process features through diffusion output head using PRESTO's components

        Args:
            features: [B, N, D] transformer features (N = num_seq_tokens + num_other_tokens)
            c: [B, D] conditioning embedding

        Returns:
            [B, C, T] diffusion output
        """
        # Apply the final layer to all features
        processed_features = self.final_layer(features, c)
        # processed_features shape: [B, N, patch_size * out_channels]

        # Get the number of patches corresponding to the original sequence 'x'
        # This is determined by the PatchEmbed configuration (input_size, patch_size)
        num_sequence_patches = self.x_embedder.num_patches # e.g., 1000 // 20 = 50

        # Select only the features corresponding to the sequence patches
        # Assumes sequence patches are the first `num_sequence_patches` tokens
        # This holds if features were constructed like cat([x_embed, other_embeds], dim=1)
        sequence_features = processed_features[:, :num_sequence_patches, :]
        # sequence_features shape: [B, num_sequence_patches, patch_size * out_channels]

        # Note: The original code had a check for `self.use_cond_token` here.
        # If that feature is used (default is False), its interaction with token
        # selection needs careful review.
        # if self.use_cond_token:
        #     x = x[..., :-1, :] # Original logic might need adjustment

        # Unpatchify using only the sequence features
        x = self.unpatchify(sequence_features)
        # x shape: [B, out_channels, num_sequence_patches * patch_size]
        # e.g., [B, 14, 50 * 20] = [B, 14, 1000]
        return x
        
    def forward_grasp(self, tsdf_features: Dict[str, torch.Tensor], positions: torch.Tensor):
        """
        Process features through grasp output heads using GIGA's decoder.

        Args:
            tsdf_features: Dictionary of encoded features from tsdf_encoder
                           (e.g., {'xy': ..., 'xz': ..., 'yz': ...}).
            positions: [B, M, 3] query positions for grasp prediction.

        Returns:
            qual: [B, M, 1] grasp quality
            rot: [B, M, 4] grasp rotation
            width: [B, M, 1] grasp width
        """
        if self.decoder_qual is None:
            raise RuntimeError("Grasp quality decoder is not initialized.")

        # The GIGA decoder expects the encoded features 'c' directly.
        # The dictionary tsdf_features holds these features.
        qual = self.decoder_qual(positions, tsdf_features) # Pass positions and the feature dict

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
        if self.decoder_tsdf is None:
            raise RuntimeError("TSDF decoder is not initialized.")

        # The GIGA decoder expects the encoded features 'c' directly.
        # The dictionary tsdf_features holds these features.
        tsdf = self.decoder_tsdf(positions, tsdf_features) # Pass positions and the feature dict

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
                for block in self.blocks:
                     features = block(features, c) # features shape [B, N+num_extra, D]

                # Apply final layer for diffusion output
                # Pass only the sequence tokens (excluding potential TSDF token) to final_layer
                sequence_features = features[:, :-num_extra_tokens, :] if num_extra_tokens > 0 else features
                # final_layer_output shape: [B, num_sequence_patches, patch_size * out_channels]
                final_layer_output = self.final_layer(sequence_features, c)

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
                # Permute: [B, N, P, C_out] -> [B, C_out, N, P]
                x_permuted = x_reshaped.permute(0, 3, 1, 2)
                # Reshape: [B, C_out, N, P] -> [B, C_out, N * P] (where N * P = T, the original sequence length)
                diffusion_out_unpatchified = x_permuted.reshape(B, C_out, N * P)
                # --------------------------

                output["diffusion_output"] = diffusion_out_unpatchified # Store the unpatchified output

            # 3. Grasp Prediction (if mode requires it and inputs available)
            if mode in ["grasp", "joint"] and tsdf_features is not None and p is not None:
                if not self.cfg.use_grasp:
                     print("Warning: Grasp prediction requested but model cfg.use_grasp is False.")
                else:
                     qual, rot, width = self.forward_grasp(tsdf_features, p)
                     output["qual"] = qual
                     output["rot"] = rot
                     output["width"] = width

            # 4. TSDF Prediction (if mode requires it and inputs available)
            if mode in ["tsdf", "joint"] and tsdf_features is not None and p_tsdf is not None:
                 if not self.cfg.use_tsdf:
                     print("Warning: TSDF prediction requested but model cfg.use_tsdf is False.")
                 else:
                     tsdf_pred = self.forward_tsdf(tsdf_features, p_tsdf)
                     output["tsdf_pred"] = tsdf_pred


        if not return_dict:
             # Return tuple based on mode - adapt as needed
             if mode == "diffusion":
                 return output.get("diffusion_output")
             elif mode == "grasp":
                 return output.get("qual"), output.get("rot"), output.get("width")
             # ... handle other modes or return combined tuple
             else:
                 return tuple(output.values())
        else:
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

