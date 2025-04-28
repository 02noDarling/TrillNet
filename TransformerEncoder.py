import torch
import torch.nn as nn
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