# python3.9 > site-packages > clip > model.py 参照.

from clip.model import ResidualAttentionBlock
from torch import nn
import torch

class MyTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, context_length: int, out_width: int = None, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.out_width = out_width if out_width is not None else width
        self.layers = layers
        self.context_length = context_length
        if attn_mask is "build_attention_mask":
            attn_mask = self.build_attention_mask()
        else:
            raise NotImplementedError("build_attention_maskにしよう.")
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
            , nn.Linear(width, out_width)
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
