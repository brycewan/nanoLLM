import math

import torch
import torch.nn as nn
from torch import Tensor
import einops


class Linear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features:int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 以（out,in）方式存储，符合row-major memory在做y=xWT的时候
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()
        
        
    def reset_parameters(self) -> None:
        # Xavier initialization N(µ = 0, σ2=2/(din+dout) ), truncated at [−3σ, 3σ]
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
        
    def forward(self, x: Tensor) -> Tensor:
        return einops.einsum(x, self.weight, "b ... in, out in -> b ... out")
    
    
class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
            )
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # N(µ = 0, σ2=1)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        
    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]