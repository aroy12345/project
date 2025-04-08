import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from presto.ConvONets.encoder.unet import UNet
from presto.ConvONets.encoder.unet3d import UNet3D
from presto.ConvONets.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate


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

class VoxelEncoder(nn.Module):
    ''' 3D-convolutional encoder network for voxel input.

    Args:
        dim (int): input dimension
        c_dim (int): output dimension
    '''

    def __init__(self, dim=3, c_dim=128):
        super().__init__()
        self.actvn = F.relu

        self.conv_in = nn.Conv3d(1, 32, 3, padding=1)

        self.conv_0 = nn.Conv3d(32, 64, 3, padding=1, stride=2)
        self.conv_1 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv_2 = nn.Conv3d(128, 256, 3, padding=1, stride=2)
        self.conv_3 = nn.Conv3d(256, 512, 3, padding=1, stride=2)
        self.fc = nn.Linear(512 * 2 * 2 * 2, c_dim)

    def forward(self, x):
        batch_size = x.size(0)

        x = x.unsqueeze(1)
        net = self.conv_in(x)
        net = self.conv_0(self.actvn(net))
        net = self.conv_1(self.actvn(net))
        net = self.conv_2(self.actvn(net))
        net = self.conv_3(self.actvn(net))

        hidden = net.view(batch_size, 512 * 2 * 2 * 2)
        c = self.fc(self.actvn(hidden))

        return c
