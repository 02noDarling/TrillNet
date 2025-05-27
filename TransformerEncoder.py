import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from MaskedMultiheadSelfAttention import MaskedMultiheadSelfAttention
from config import *

class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, nhead):
        super(TransformerEncoderLayer, self).__init__()
        # self.self_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout, batch_first=batch_first)
        self.self_attn = MaskedMultiheadSelfAttention(model_dim, nhead)

        self.linear1 = nn.Linear(model_dim, 4*model_dim)
        self.linear2 = nn.Linear(model_dim*4, model_dim)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.activation = nn.ReLU()
    
    def forward(self, src, mask=None):
        # 多头自注意力层
        attn_output = self.self_attn(src, mask)
        src = src + attn_output
        src = self.norm1(src)

        # 前馈网络
        feedforward_output = self.linear1(src)
        feedforward_output = self.activation(feedforward_output)
        feedforward_output = self.linear2(feedforward_output)

        src = src + feedforward_output
        src = self.norm2(src)
        return src

class SwiGLU(nn.Module):
    """SwiGLU激活函数: SwiGLU(x) = Swish(xW + b) ⊗ (xV + c)"""
    def __init__(self, dim):
        super().__init__()
        hidden_dim = int(2 * dim * 4 / 3)  
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # down projection  
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # up projection
    
    def forward(self, x):
        # SwiGLU: swish(x @ w1) * (x @ w3) @ w2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LLaMaEncoderLayer(nn.Module):
    def __init__(self, model_dim, nhead):
        super(LLaMaEncoderLayer, self).__init__()
        
        # 注意力层
        self.self_attn = MaskedMultiheadSelfAttention(model_dim, nhead)
        
        # SwiGLU前馈网络
        self.feed_forward = SwiGLU(model_dim)
        
        # Pre-LayerNorm: 在注意力和前馈网络之前应用LayerNorm
        self.attention_norm = nn.LayerNorm(model_dim)
        self.ffn_norm = nn.LayerNorm(model_dim)
    
    def forward(self, src, mask=None):
        # Pre-LayerNorm + 多头自注意力 + 残差连接
        normalized_src = self.attention_norm(src)
        attn_output = self.self_attn(normalized_src, mask)
        src = src + attn_output
        
        # Pre-LayerNorm + SwiGLU前馈网络 + 残差连接
        normalized_src = self.ffn_norm(src)
        ffn_output = self.feed_forward(normalized_src)
        src = src + ffn_output
        
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, nums):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for i in range(nums)])
    
    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src
    
if __name__ == "__main__":
    encoder_layer = TransformerEncoderLayer(EMBED_DIM, NHEAD)
    encoder = TransformerEncoder(encoder_layer, ENCODER_NUMS)
    src = torch.randn(10, 5, EMBED_DIM)
    output = encoder(src)
    print(output.shape)