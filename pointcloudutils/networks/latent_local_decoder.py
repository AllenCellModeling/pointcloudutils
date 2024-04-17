import torch.nn as nn
import torch.nn.functional as F
from cyto_dl.nn.point_cloud.graph_functions import (
    normalize_3d_coordinate,
    normalize_coordinate,
)


def _deconv_layer(in_channels, out_channels, k, s, p):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, k, s, p),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )

def _final_deconv_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        nn.Tanh()
    )
    

class LatentLocalDecoder(nn.Module):
    """Decoder.
        LocalDecoder with (V)AE on grid features.

    Args:
        num_features (int): dimensionality of each set of plane features
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        grid_feats_latent_dim (int): latent dimension size
    """

    def __init__(
        self,
        num_features=128,
        hidden_size=256,
        n_blocks=5,
        leaky=False,
        sample_mode="bilinear",
        padding=0.1,
        grid_feats_latent_dim=128,
        **base_kwargs,
    ):
        super().__init__()
        self.num_features = num_features

        self.grid_feats_latent_dim = grid_feats_latent_dim
        
        cnn_modules = [
            _deconv_layer(grid_feats_latent_dim, grid_feats_latent_dim//2, 4, 2, 1),          
            _deconv_layer(grid_feats_latent_dim//2, grid_feats_latent_dim//2, 4, 2, 1),            
            _deconv_layer(grid_feats_latent_dim//2, grid_feats_latent_dim//2, 4, 2, 1),              
            _final_deconv_block(256, num_features*3, 4, 2, 1)         
        ]

        self.pre_cnn = nn.Linear(grid_feats_latent_dim, grid_feats_latent_dim*4*4) 
        self.cnn_blocks = nn.Sequential(*cnn_modules)
                    
        self.n_blocks = n_blocks

        if self.num_features != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(self.num_features, hidden_size) for i in range(n_blocks)]
            )

        self.fc_p = nn.Linear(3, hidden_size)

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size) for i in range(n_blocks)]
        )

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

        self.th = nn.Tanh()

        

    def sample_plane_feature(self, p, c, plane="xz"):
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(
            c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode
        ).squeeze(-1)

        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(
            p.clone(), padding=self.padding
        )  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            F.grid_sample(
                c,
                vgrid,
                padding_mode="border",
                align_corners=True,
                mode=self.sample_mode,
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        return c


    def forward(self, p, z, **kwargs):
        
        if self.num_features != 0:
            pre_feats = self.pre_cnn(z)
            pre_feats = pre_feats.view(-1, self.grid_feats_latent_dim, 4, 4)
            xhat = self.cnn_blocks(pre_feats)
            xhat_xy = xhat[:,:self.num_features,:,:]
            xhat_yz = xhat[:,self.num_features:self.num_features*2,:,:]
            xhat_xz = xhat[:,self.num_features*2:,:,:]
                
            c_plane = {"xy":xhat_xy,"yz":xhat_yz,"xz":xhat_xz}
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)

        p = p.float()
        net = self.fc_p(p)
        for i in range(self.n_blocks):
            if self.num_features != 0:
                net = net + self.fc_c[i](c)
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        out = self.th(out)
        return out, c_plane


class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx
