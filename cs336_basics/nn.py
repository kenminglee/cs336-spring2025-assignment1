import math

import torch
from torch import nn
import einx
from jaxtyping import Float32

class Linear(nn.Module):

    def __init__(
        self, 
        in_features:int, 
        out_features:int, 
        device:torch.device | None=None, 
        dtype:torch.dtype | None=None
    ):
        super().__init__()
        weights = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std_dev = math.sqrt(2/(in_features+out_features))
        mean = 0
        weights = nn.init.trunc_normal_(weights, mean, std_dev, a=-3*std_dev, b=3*std_dev)
        self.w = nn.Parameter(weights, requires_grad=True)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return einx.dot("d_out [d_in], b... [d_in] -> b... d_out", self.w, x)

class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings:int, 
        embedding_dim: int, 
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        weights = torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device)
        weights = nn.init.trunc_normal_(weights, mean=0, std=1, a=-3, b=3)
        self.embed_mat = nn.Parameter(weights, requires_grad=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # unsqueeze, then perform lookup on embedding matrix
        return einx.get_at("[vocab] d_model, batch_size (seq_len [1]) -> batch_size seq_len d_model", self.embed_mat, token_ids)


class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device:torch.device|None=None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        gain = torch.ones((d_model,), dtype=dtype, device=device)
        self.g = nn.Parameter(gain, requires_grad=True)
        self.eps:float = eps
        self.d_model:int = d_model


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt((torch.pow(x, 2) + self.eps).sum(-1)/self.d_model)
        
        # x.shape = (batch, seq_len, d_model)
        # rms.shape = (batch, seq_len)
        # g.shape = (d_model, )
        result = (x / rms.unsqueeze(-1)) * self.g

        return result.to(in_dtype)

# SiLU/Swish activation function
def silu(x:torch.Tensor):
    return x * torch.sigmoid(x)

# Combination between SiLU and Gated Linear Units
class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int | None = None, device:torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        if not d_ff:
            d_ff:int = round(((8/3)*d_model)/64)
            d_ff *= 64 # ensure that d_ff is multiple of 64
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


