import torch
import torch.nn as nn
from MaskedMultiheadSelfAttention import MaskedMultiheadSelfAttention
from WindowBasedCrossAttention import WindowBasedCrossAttention
from InformationAggregationLayer import InformationAggregationLayer

def get_graph(history, candidate_size, batch_size):
    # 初始化图的邻接矩阵
    array = torch.zeros(batch_size, candidate_size, candidate_size)
    # _, seq_x, seq_y = history.shape  # 获取历史张量的形状
    # print(f"History shape: {history.shape}")
    
    # 使用批量化操作构建邻接矩阵
    history_1 = history[:, :, :-1]  # 当前节点
    history_2 = history[:, :, 1:]   # 下一个节点
    mask = history_1 != history_2  # 计算是否不相等
    # 通过广播将相邻节点不相等的部分置为1，累加到邻接矩阵

    for b in range(batch_size):
        array[b, history_1[b, mask[b]], history_2[b, mask[b]]] += 1
    
    return array

def Mytransform(A, candidate_size, batch_size):
    # 对每个批次创建单位矩阵（加上自环）
    IN = torch.eye(candidate_size).unsqueeze(0).repeat(batch_size, 1, 1)  # 扩展到 batch_size
    A += IN  # 对邻接矩阵加上单位矩阵（自环）
    
    # 为每个批次计算度矩阵 D
    D = torch.zeros(batch_size, candidate_size, candidate_size)  # 创建度矩阵（batch_size 维度）
    row_sums = A.sum(dim=1)  # 计算每一行的和，避免使用显式的 for 循环
    row_sums_sqrt = row_sums.sqrt()  # 开根号
    D = torch.diag_embed(row_sums_sqrt)  # 构造对角矩阵 D
    
    # print(f"D:================>")
    # print(D)
    
    # 执行 D @ A @ D 操作
    return D @ A @ D  # 拉普拉斯矩阵变换

def generate_temporal_embeddings(max_time, embed_dim):
    d = embed_dim // 2  # 嵌入维度是 2d
    embeddings = []
    for t in range(max_time):
        e_t = torch.zeros(embed_dim)
        for k in range(d):
            denominator = 10000 ** (2 * k / embed_dim)
            e_t[2 * k] = torch.sin(torch.tensor(t / denominator))
            e_t[2 * k + 1] = torch.cos(torch.tensor(t / denominator))
        embeddings.append(e_t)
    return torch.stack(embeddings)

def fuse_spatial_temporal(pos_embeddings, history, temporal_embeddings, dim, type):
    if type ==0:
        batch_size, history_size, seq_len = history.shape
        fused_embeddings = torch.zeros(batch_size, history_size, seq_len, 2*dim)
        for batch in range(batch_size):
            for i in range(history_size):
                for j in range(seq_len):
                    fused_embeddings[batch][i][j] = pos_embeddings[batch][history[batch][i][j]] + temporal_embeddings[j]
        
        return fused_embeddings
    else:
        batch_size, seq_len = history.shape
        fused_embeddings = torch.zeros(batch_size, seq_len, 2*dim)
        for batch in range(batch_size):
            for i in range(seq_len):
                fused_embeddings[batch][i] = pos_embeddings[batch][history[batch][i]] + temporal_embeddings[i]
        
        return fused_embeddings

class TrillNet(nn.Module):
    def __init__(self, dim, num_heads = 2, window_size=2):
        super(TrillNet, self).__init__()
        self.dim = dim
        self.gcn = nn.Linear(dim, dim)  # 线性变换
        self.relu = nn.ReLU()  # ReLU激活函数

        self.num_heads = num_heads
        self.MaskedMultiheadSelfAttention = MaskedMultiheadSelfAttention(2*dim, num_heads)

        self.WindowBasedCrossAttention = WindowBasedCrossAttention(2*dim, window_size)

        self.InformationAggregationLayer = InformationAggregationLayer(dim)

    def forward(self, candidate_size, history, current_trajectory, E):
        batch_size = E.shape[0]
        A = get_graph(history, candidate_size, batch_size)  # 获取图的邻接矩阵
        A = Mytransform(A, candidate_size, batch_size)  # 进行图的拉普拉斯变换
        Eg = self.relu(self.gcn(A @ E))  # 应用线性层与激活函数
        pos_embeddings = torch.cat((Eg,E),dim=-1)
        temporal_embeddings = generate_temporal_embeddings(history.shape[2], 2*self.dim)
        fused_history_embeddings = fuse_spatial_temporal(pos_embeddings, history, temporal_embeddings, self.dim, 0)
        fused_current_embeddings = fuse_spatial_temporal(pos_embeddings, current_trajectory, temporal_embeddings, self.dim, 1)

         # 将 history 和 current 都转换为合适的形状 [batch_size * trajectory_size, seq_len, dim]
        fused_history_embeddings = fused_history_embeddings.view(batch_size * history.shape[1], history.shape[2], -1)
        fused_current_embeddings = fused_current_embeddings.view(batch_size, -1, self.dim * 2)

        mmsa_history_embeddings = self.MaskedMultiheadSelfAttention(fused_history_embeddings)
        mmsa_current_embeddings = self.MaskedMultiheadSelfAttention(fused_current_embeddings)
        
        # # 将历史轨迹的输出重新形状为原始形状 [batch_size, trajectory_size, seq_len, dim]
        # mmsa_history_embeddings = mmsa_history_embeddings.view(batch_size, history.shape[1], history.shape[2], 2*self.dim)
        # wca_history_embeddings = mmsa_history_embeddings.view(batch_size * history.shape[1], history.shape[2], -1)

        temp_current_embeddings = mmsa_current_embeddings.unsqueeze(1).repeat(1, history.shape[1], 1, 1)
        temp_current_embeddings = temp_current_embeddings.view(batch_size * history.shape[1], history.shape[2], -1)

        wca_history_embeddings = self.WindowBasedCrossAttention(mmsa_history_embeddings, temp_current_embeddings)
        wca_history_embeddings = wca_history_embeddings.view(batch_size, history.shape[1], history.shape[2], 2*self.dim)

        wca_current_embeddings = self.WindowBasedCrossAttention(mmsa_current_embeddings, mmsa_current_embeddings)

        final_embeddings = self.InformationAggregationLayer(wca_history_embeddings, wca_current_embeddings)


        # print("results:===================>")
        # print(f"History attention out shape: {wca_history_embeddings.shape}")
        # print(f"Current attention out shape: {wca_current_embeddings.shape}")
        # print("end==>")
        return final_embeddings

if __name__ == "__main__":    
    batch_size = 2  # 设置 batch_size 大于 1
    history_size = 2
    seq_len = 4
    candidate_size = 4
    embed = 9

    net = TrillNet(dim=embed)
    
    # 模拟输入数据
    history = torch.tensor(
        [
            [
                [0, 1, 2, 3],
                [1, 2, 3, 0]
            ],
            [
                [0, 1, 2, 3],
                [1, 2, 3, 0]
            ]
        ]
    )  # 假设有两个批次，history 是一个形状为 [batch_size, seq_x, seq_y, 2] 的张量
    current_trajectory = torch.tensor(
        [
            [2, 3, 0, 1],
            [2, 3, 0, 1]
        ]
    )  # 假设有两个批次，history 是一个形状为 [batch_size, seq_x, seq_y, 2] 的张量
    print(history.shape)
    E = torch.randn(batch_size, candidate_size, embed)  # 输入的特征矩阵，形状为 [batch_size, candidate_size, dim]
    final_embeddings = net(candidate_size, history, current_trajectory, E)  # 通过网络进行推理
    print("final_embeddings:==================>")
    print(final_embeddings.shape)
