# import logging
# from typing import Optional, Sequence, Union

# import numpy as np
# import torch
# import torch.nn as nn
# from omegaconf import DictConfig
# from cyto_dl.models.vae.image_vae import ImageVAE
# from monai.networks.layers.factories import Act, Norm

# Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
# logger = logging.getLogger("lightning")
# logger.propagate = False


# class _Scale(nn.Module):
#     def __init__(self, scale):
#         super().__init__()
#         self.scale = torch.tensor(scale)

#     def forward(self, x):
#         return x * self.scale.type_as(x)


# class VQVAE(ImageVAE):
#     def __init__(
#         self,
#         latent_dim: int,
#         spatial_dims: int,
#         num_embeddings: int,
#         commitment_cost: float,
#         decay: float,
#         in_shape: Sequence[int],
#         channels: Sequence[int],
#         strides: Sequence[int],
#         kernel_sizes: Sequence[int],
#         group: Optional[str] = None,
#         out_channels: int = None,
#         decoder_initial_shape: Optional[Sequence[int]] = None,
#         decoder_channels: Optional[Sequence[int]] = None,
#         decoder_strides: Optional[Sequence[int]] = None,
#         maximum_frequency: int = 8,
#         background_value: float = 0,
#         act: Optional[Union[Sequence[str], str]] = Act.PRELU,
#         norm: Union[Sequence[str], str] = Norm.INSTANCE,
#         dropout: Optional[Union[Sequence, str, float]] = None,
#         bias: bool = True,
#         prior: str = "gaussian",
#         last_act: Optional[str] = None,
#         last_scale: float = 1.0,
#         mask_input: bool = False,
#         mask_output: bool = False,
#         clip_min: Optional[int] = None,
#         clip_max: Optional[int] = None,
#         num_res_units: int = 2,
#         up_kernel_size: int = 3,
#         first_conv_padding_mode: str = "replicate",
#         encoder_padding: Optional[Union[int, Sequence[int]]] = None,
#         eps: float = 1e-8,
#         **base_kwargs,
#     ):

#         self.x_label = 'embedding'

#         super().__init__(
#             x_label=self.x_label,
#             latent_dim=latent_dim,
#             spatial_dims=spatial_dims,
#             in_shape=in_shape,
#             channels=channels,
#             strides=strides,
#             kernel_sizes=kernel_sizes,
#             group=group,
#             out_channels=out_channels,
#             decoder_initial_shape=decoder_initial_shape,
#             decoder_channels=decoder_channels,
#             decoder_strides=decoder_strides,
#             maximum_frequency=maximum_frequency,
#             background_value=background_value,
#             act=act,
#             norm=norm,
#             dropout=dropout,
#             bias=bias,
#             prior=prior,
#             last_act=last_act,
#             last_scale=last_scale,
#             mask_input=mask_input,
#             mask_output=mask_output,
#             clip_min=clip_min,
#             clip_max=clip_max,
#             num_res_units=num_res_units,
#             up_kernel_size=up_kernel_size,
#             first_conv_padding_mode=first_conv_padding_mode,
#             encoder_padding=encoder_padding,
#             eps=eps,
#             **base_kwargs,
#         )
#         self.num_embeddings = num_embeddings
#         self.commitment_cost = commitment_cost
#         self.decay = decay
#         self.vq_layer = nn.ModuleDict({self.x_label: VectorQuantizerEMA(latent_dim, self.num_embeddings, self.commitment_cost, self.decay)})

#     def forward(self, batch, decode=False, inference=True, return_params=False, **kwargs):
#         is_inference = inference or not self.training

#         z_params = self.encode(batch, **kwargs)
#         quantized, loss = self.vq_layer[self.x_label](z_params)

#         if not decode:
#             return quantized

#         xhat = self.decode(quantized)
#         if return_params:
#             return xhat, z, z_params

#         return xhat, z


# #EMA process
# class ExponentialMovingAverage(nn.Module):
#     """Maintains an exponential moving average for a value.
    
#       This module keeps track of a hidden exponential moving average that is
#       initialized as a vector of zeros which is then normalized to give the average.
#       This gives us a moving average which isn't biased towards either zero or the
#       initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)
      
#       Initially:
#           hidden_0 = 0
#       Then iteratively:
#           hidden_i = hidden_{i-1} - (hidden_{i-1} - value) * (1 - decay)
#           average_i = hidden_i / (1 - decay^i)
#     """
    
#     def __init__(self, init_value, decay):
#         super().__init__()
        
#         self.decay = decay
#         self.counter = 0
#         self.register_buffer("hidden", torch.zeros_like(init_value))
        
#     def forward(self, value):
#         self.counter += 1
#         self.hidden.sub_((self.hidden - value) * (1 - self.decay))
#         average = self.hidden / (1 - self.decay ** self.counter)
#         return average

# class Mish(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self,x):
#         return x*torch.tanh(F.softplus(x))

# class VectorQuantizerEMA(nn.Module):
#     """
#     VQ-VAE layer: Input any tensor to be quantized. 
#     Args:
#         embedding_dim (int): the dimensionality of the tensors in the
#           quantized space. Inputs to the modules must be in this format as well.
#         num_embeddings (int): the number of vectors in the quantized space.
#         commitment_cost (float): scalar which controls the weighting of the loss terms (see
#           equation 4 in the paper - this variable is Beta).
#     """
#     def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay,
#                epsilon=1e-5):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.num_embeddings = num_embeddings
#         self.commitment_cost = commitment_cost
#         self.epsilon = epsilon
        
#         # initialize embeddings as buffers
#         embeddings = torch.empty(self.num_embeddings, self.embedding_dim)
#         nn.init.xavier_uniform_(embeddings)
#         self.register_buffer("embeddings", embeddings)
#         self.ema_dw = ExponentialMovingAverage(self.embeddings, decay)
        
#         # also maintain ema_cluster_size， which record the size of each embedding
#         self.ema_cluster_size = ExponentialMovingAverage(torch.zeros((self.num_embeddings,)), decay)
        
#     def forward(self, x: dict):
#         x = x['embedding']
#         flat_x = x.reshape(-1, self.embedding_dim)
        
#         # Use index to find embeddings in the latent space
#         encoding_indices = self.get_code_indices(flat_x)
#         quantized = self.quantize(encoding_indices)
#         quantized = quantized.view_as(x) 
        
#         #EMA
#         with torch.no_grad():
#             encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
#             updated_ema_cluster_size = self.ema_cluster_size(torch.sum(encodings, dim=0))
#             n = torch.sum(updated_ema_cluster_size)
#             updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
#                                       (n + self.num_embeddings * self.epsilon) * n)
#             dw = torch.matmul(encodings.t(), flat_x) # sum encoding vectors of each cluster
#             updated_ema_dw = self.ema_dw(dw)
#             normalised_updated_ema_w = (
#               updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1))
#             self.embeddings.data = normalised_updated_ema_w
    

#         # commitment loss
#         e_latent_loss = F.mse_loss(x, quantized.detach())
#         loss = self.commitment_cost * e_latent_loss

#         # Straight Through Estimator
#         quantized = x + (quantized - x).detach()
    
#         return {self.x_label: quantized}, loss
    
#     def get_code_indices(self, flat_x):
#         # compute L2 distance
#         distances = (
#             torch.sum(flat_x ** 2, dim=1, keepdim=True) +
#             torch.sum(self.embeddings ** 2, dim=1) -
#             2. * torch.matmul(flat_x, self.embeddings.t())
#         ) # [N, M]
#         encoding_indices = torch.argmin(distances, dim=1) # [N,]
#         return encoding_indices
    
#     def quantize(self, encoding_indices):
#         """Returns embedding tensor for a batch of indices."""
#         return F.embedding(encoding_indices, self.embeddings)   