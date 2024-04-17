# import torch
# from torch import nn

# from cyto_dl import utils
# from cyto_dl.nn.point_cloud.vnn import VNLinear, VNRotationMatrix
# from equiformer_pytorch import Equiformer

# log = utils.get_pylogger(__name__)


# def maxpool(x, dim=-1, keepdim=False):
#     out, _ = x.max(dim=dim, keepdim=keepdim)
#     return out


# def meanpool(x, dim=-1, keepdim=False):
#     out = x.mean(dim=dim, keepdim=keepdim)
#     return out


# class EquivTransformer(nn.Module):
#     def __init__(
#         self,
#         # encoder: dict,
#         x_label: str = "pcloud",
#         concat_feats: bool = False,
#         num_points: int = 256,
#         num_features: int = 128,
#     ):
#         super().__init__()
#         self.encoder = Equiformer(
#             num_tokens = 24,
#             dim = (4, 4, 2),               # dimensions per type, ascending, length must match number of degrees (num_degrees)
#             dim_head = (4, 4, 4),          # dimension per attention head
#             heads = (2, 2, 2),             # number of attention heads
#             num_linear_attn_heads = 0,     # number of global linear attention heads, can see all the neighbors
#             num_degrees = 3,               # number of degrees
#             depth = 4,                     # depth of equivariant transformer
#             attend_self = True,            # attending to self or not
#             reduce_dim_out = True,         # whether to reduce out to dimension of 1, say for predicting new coordinates for type 1 features
#             l2_dist_attention = False,      # set to False to try out MLP attention
#             reversible = True,
#         )
#         # self.concat_feats = concat_feats
#         self.x_label = x_label
#         # self.dim_feat = self.encoder.dim_feat
#         self.num_points = num_points
#         # self.num_features = num_features

#         self.pool = meanpool
#         # rotation module
#         self.rotation = VNRotationMatrix(self.num_points, dim=3, return_rotated=True)

#     def forward(self, x, get_rotation=False):
#         # if self.dim_feat > 0:
#         #     coors_out, feats_out = self.encoder(
#         #         x,
#         #     )
#         # else:
#         #     coors_out = self.encoder(x)
#         batch_size = x.shape[0]
#         num_points = x.shape[1]
#         mask = torch.ones(batch_size, num_points).type_as(x)
#         feats = x[:,:,-1]
#         coors = x[:,:,:3]
#         out = self.encoder(coors, feats, mask)
#         rot_eq_embed = out.type1
#         rot_inv_embed = out.type0

#         _, rot = self.rotation(rot_eq_embed)

#         rot = rot.mT

#         if get_rotation:
#             return {self.x_label: rot_inv_embed, "rotation": rot}

#         return {self.x_label: rot_inv_embed}
