import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowBasedCrossAttention(nn.Module):
    def __init__(self, embed_dim, window_size):
        """
        单头 Self-Attention 层，带窗口掩码控制
        :param embed_dim: 输入特征的维度
        :param window_size: 允许的注意力窗口范围
        """
        super(WindowBasedCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size

        # 定义线性变换层（Q, K, V）
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, X, Y):
        """
        :param X: 输入数据，形状 [batch_size, seq_len, embed_dim]
        :return: Self-Attention 计算后的输出，形状 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = X.shape

        # 计算 Q, K, V
        Q = self.W_Q(Y)  # [batch_size, seq_len, embed_dim]
        K = self.W_K(X)  # [batch_size, seq_len, embed_dim]
        V = self.W_V(X)  # [batch_size, seq_len, embed_dim]

        # 计算 Q * K^T，得到注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, seq_len, seq_len]

        # 缩放
        scores = scores / (self.embed_dim ** 0.5)

        # 生成掩码矩阵 mask，满足 |i - j| <= w 的地方为 1，否则为 0
        mask = torch.ones((seq_len, seq_len), device=X.device)
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) > self.window_size:
                    mask[i][j] = 0

        # 应用掩码，将 mask=0 的地方的 score 设为 -1e10
        scores = scores.masked_fill(mask == 0, -1e10)

        # 计算 Softmax
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]

        # 计算加权值
        output = torch.matmul(attention_weights, V)  # [batch_size, seq_len, embed_dim]

        return output

# 测试代码
if __name__ == "__main__":
    batch_size = 2
    seq_len = 6
    embed_dim = 8
    window_size = 2  # 例如，w = 2 代表最多关注前后 2 个时间步

    X = torch.randn(batch_size, seq_len, embed_dim)  # 随机输入
    Y = torch.randn(batch_size, seq_len, embed_dim)  # 随机输入

    attention = WindowBasedCrossAttention(embed_dim, window_size)
    output = attention(X, Y)
    print(output.shape)  # 应该是 [batch_size, seq_len, embed_dim]
