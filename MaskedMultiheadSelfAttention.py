import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MaskedMultiheadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        # 每个头的维度
        self.head_dim = embed_dim // num_heads

        # 定义 Q, K, V 的线性变换层
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # 输出的线性变换
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, mask=None):
        """
        query: [batch_size, seq_len, embed_dim]
        mask: [batch_size, seq_len, seq_len] - 掩码矩阵，0表示该位置的相关性需要被置为0
        """
        batch_size, seq_len, _ = query.size()

        # 计算 Q, K, V
        Q = self.query(query)  # [batch_size, seq_len, embed_dim]
        K = self.key(query)     # [batch_size, seq_len, embed_dim]
        V = self.value(query)   # [batch_size, seq_len, embed_dim]

        # 将 Q, K, V 切分成多个头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]

        # 计算 Q * K^T 得到注意力得分
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]

        # 如果有掩码，应用掩码
        if mask is not None:
            attention_scores = attention_scores + mask.unsqueeze(1)  # 掩码为0的位置相关性设为0

        # 缩放
        attention_scores = attention_scores / (self.head_dim ** 0.5)  # 缩放点积的结果

        # 计算软最大值，得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]

        # 计算加权值
        output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, head_dim]

        # 合并头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)  # [batch_size, seq_len, embed_dim]

        # 最后通过线性变换得到输出
        output = self.out_proj(output)  # [batch_size, seq_len, embed_dim]

        return output

# 测试代码
if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    embed_dim = 8
    num_heads = 2

    query = torch.randn(batch_size, seq_len, embed_dim)  # 随机输入
    mask = torch.ones(batch_size, seq_len, seq_len)  # 掩码矩阵，全1
    mask[:, 1, 2] = 0  # 将第 1 行第 2 列的位置设为 0

    # 实例化并运行 MaskedMultiheadSelfAttention
    attention = MaskedMultiheadSelfAttention(embed_dim, num_heads)
    output = attention(query, mask)
    print(output.shape)  # 应该是 [batch_size, seq_len, embed_dim]
