import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import math

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        # with sdpa_kernel([SDPBackend.MATH]):
        #     return F.scaled_dot_product_attention(
        #         query, key, value,
        #         attn_mask = mask,
        #         dropout_p = dropout.p if dropout is not None else 1.0, 
        #         is_causal = False, 
        #         scale = query.size(-1)
        #     )
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value) #, p_attn

class FlashAttention(Attention):
    """Compute Scaled Dot Product using Flash Attention
    """
    
    def forward(self, query, key, value, mask=None, dropout: nn.Dropout| None=None):
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
            return F.scaled_dot_product_attention(
                query, key, value,
                attn_mask = mask,
                dropout_p = dropout.p if dropout is not None else 1.0, 
                is_causal = False, 
                scale = query.size(-1)
            )
        # return super().forward(query, key, value, mask, dropout)