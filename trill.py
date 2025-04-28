import torch
import torch.nn as nn
from config import *
from MaskedMultiheadSelfAttention import MaskedMultiheadSelfAttention
from WindowBasedCrossAttention import WindowBasedCrossAttention
from InformationAggregationLayer import InformationAggregationLayer
from TransformerEncoder import *

import torch

# def get_graph(history, candidate_size, batch_size):
#     # print(history.shape)
#     # 初始化图的邻接矩阵
#     array = torch.zeros(batch_size, candidate_size, candidate_size)
#     # _, seq_x, seq_y = history.shape  # 获取历史张量的形状
#     # print(f"History shape: {history.shape}")
#     # print(f"history_shape:{history.shape}")
#     # 使用批量化操作构建邻接矩阵
#     history_1 = history[:, :, :-1]  # 当前节点
#     history_2 = history[:, :, 1:]   # 下一个节点
#     mask = (history_1 != history_2) & (history_1 != candidate_size) & (history_2 != candidate_size)
#     # 通过广播将相邻节点不相等的部分置为1，累加到邻接矩阵

#     # for b in range(batch_size):
#     #     array[b, history_1[b, mask[b]], history_2[b, mask[b]]] += 1

#     for b in range(batch_size):
#         for i in range(history.shape[1]):
#             for j in range(history.shape[2]-1):
#                 if history[b][i][j] != candidate_size and history[b][i][j+1] != candidate_size and history[b][i][j] != history[b][i][j+1]:
#                     array[b][history[b][i][j]][history[b][i][j+1]] += 1
    
#     # print(array)
#     # print(array[0][2][4])
#     # exit(0)
#     return array

# def get_graph(history, candidate_size, batch_size):
#     """
#     构建加权全局转移图的邻接矩阵。
    
#     参数：
#     - history: 历史轨迹张量，形状为 (batch_size, num_trajectories, seq_len)
#     - candidate_size: 候选位置的数量（节点数量）
#     - batch_size: 批次大小
#     - current_idx: 当前轨迹的下标，用于计算时间权重
    
#     返回：
#     - array: 加权邻接矩阵，形状为 (batch_size, candidate_size, candidate_size)
#     """
#     # 初始化图的邻接矩阵
#     array = torch.zeros(batch_size, candidate_size, candidate_size)
    
#     # 获取历史轨迹的形状
#     _, num_trajectories, seq_len = history.shape  # history.shape = (batch_size, num_trajectories, seq_len)
    
#     # 计算每条历史轨迹的时间权重 w_j = 1 + e^(-0.01 * (当前轨迹下标 - 历史轨迹下标))
#     # 历史轨迹下标从 0 到 num_trajectories-1
#     history_indices = torch.arange(num_trajectories, dtype=torch.float32)  # [0, 1, ..., num_trajectories-1]
#     weights = 1 + torch.exp(-0.01 * (num_trajectories - 1 - history_indices))  # 形状为 (num_trajectories,)

    
#     # 使用批量化操作构建加权邻接矩阵
#     history_1 = history[:, :, :-1]  # 当前节点，形状为 (batch_size, num_trajectories, seq_len-1)
#     history_2 = history[:, :, 1:]   # 下一个节点，形状为 (batch_size, num_trajectories, seq_len-1)
    
#     # 创建掩码，确保只处理有效转移（节点不相等且不为 candidate_size）
#     mask = (history_1 != history_2) & (history_1 != candidate_size) & (history_2 != candidate_size)
#     # 形状为 (batch_size, num_trajectories, seq_len-1)
    
#     # 为每个批次构建加权邻接矩阵
#     for b in range(batch_size):
#         for j in range(num_trajectories):
#             # 获取当前轨迹的掩码
#             traj_mask = mask[b, j]  # 形状为 (seq_len-1,)
#             # 获取当前轨迹的转移对
#             src_nodes = history_1[b, j, traj_mask]  # 源节点
#             dst_nodes = history_2[b, j, traj_mask]  # 目标节点
#             # 累加加权转移频率
#             array[b, src_nodes, dst_nodes] += 1
    
#     print(array.shape)
#     print(array)
#     exit(0)
#     return array

def get_graph(history, candidate_size, batch_size):
    """
    构建加权全局转移图的邻接矩阵。
    
    参数：
    - history: 历史轨迹张量，形状为 (batch_size, num_trajectories, seq_len)
    - candidate_size: 候选位置的数量（节点数量）
    - batch_size: 批次大小
    - current_idx: 当前轨迹的下标，用于计算时间权重
    
    返回：
    - array: 加权邻接矩阵，形状为 (batch_size, candidate_size, candidate_size)
    """
    # 初始化图的邻接矩阵
    array = torch.zeros(batch_size, candidate_size, candidate_size, dtype=torch.float32)
    
    # 获取历史轨迹的形状
    _, num_trajectories, seq_len = history.shape  # history.shape = (batch_size, num_trajectories, seq_len)
    
    # 计算每条历史轨迹的时间权重 w_j = 1 + e^(-0.01 * (当前轨迹下标 - 历史轨迹下标))
    history_indices = torch.arange(num_trajectories, dtype=torch.float32)  # [0, 1, ..., num_trajectories-1]
    weights = 1 + torch.exp(-0.01 * (num_trajectories - 1 - history_indices))  # 形状为 (num_trajectories,)
    
    # 遍历批次、每条历史轨迹、每个时间步，构建加权邻接矩阵
    for b in range(batch_size):
        for i in range(num_trajectories):
            for j in range(seq_len - 1):
                src_node = history[b, i, j].item()  # 当前节点
                dst_node = history[b, i, j + 1].item()  # 下一个节点
                # 检查条件：节点不等于 candidate_size（无效位置），且相邻节点不相等
                if src_node != candidate_size and dst_node != candidate_size and src_node != dst_node:
                    array[b, src_node, dst_node] += weights[i]
    
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

# def generate_temporal_embeddings(max_time, embed_dim):
#     d = embed_dim // 2  # 嵌入维度是 2d
#     embeddings = []
#     for t in range(max_time):
#         e_t = torch.zeros(embed_dim)
#         for k in range(d):
#             denominator = 10000 ** (2 * k / embed_dim)
#             e_t[2 * k] = torch.sin(torch.tensor(t / denominator))
#             e_t[2 * k + 1] = torch.cos(torch.tensor(t / denominator))
#         embeddings.append(e_t)
#     return torch.stack(embeddings)

def generate_temporal_embeddings(max_time, embed_dim):
    # 创建时间序列 [0, 1, 2, ..., max_time-1]
    t = torch.arange(0, max_time, dtype=torch.float)
    
    # 创建索引 [0, 2, 4, ..., embed_dim-2]
    indices = torch.arange(0, embed_dim, 2, dtype=torch.float)
    
    # 计算分母 10000^(2k/d)
    denominators = 10000 ** (indices / embed_dim)
    
    # 扩展维度以便广播 [max_time, d]
    t = t.unsqueeze(1)  # [max_time, 1]
    denominators = denominators.unsqueeze(0)  # [1, d]
    
    # 计算 t/denominator [max_time, d]
    arguments = t / denominators
    
    # 计算正弦和余弦
    embeddings = torch.zeros(max_time, embed_dim)
    embeddings[:, 0::2] = torch.sin(arguments)  # 偶数位置
    embeddings[:, 1::2] = torch.cos(arguments)  # 奇数位置
    
    return embeddings

# def fuse_spatial_temporal(pos_embeddings, history, temporal_embeddings, dim, null_embedding, candidate_size, type):
#     if type ==0:
#         batch_size, history_size, seq_len = history.shape
#         fused_embeddings = torch.zeros(batch_size, history_size, seq_len, 2*dim)
#         for batch in range(batch_size):
#             for i in range(history_size):
#                 for j in range(seq_len):
#                     if history[batch][i][j] == candidate_size:
#                         fused_embeddings[batch][i][j] = null_embedding + temporal_embeddings[j]
#                     else:
#                         fused_embeddings[batch][i][j] = pos_embeddings[batch][history[batch][i][j]] + temporal_embeddings[j]
        
#         return fused_embeddings
#     else:
#         batch_size, seq_len = history.shape
#         fused_embeddings = torch.zeros(batch_size, seq_len, 2*dim)
#         for batch in range(batch_size):
#             for i in range(seq_len):
#                 if history[batch][i] == candidate_size:
#                     fused_embeddings[batch][i] = null_embedding + temporal_embeddings[i]
#                 else:
#                     fused_embeddings[batch][i] = pos_embeddings[batch][history[batch][i]] + temporal_embeddings[i]
        
#         return fused_embeddings

def fuse_spatial_temporal(pos_embeddings, history, temporal_embeddings, dim, null_embedding, candidate_size, type):
    if type == 0:
        batch_size, history_size, seq_len = history.shape
        
        # 创建索引掩码来区分是否为null_embedding
        is_null = (history == candidate_size)
        
        # 创建结果tensor
        fused_embeddings = torch.zeros(batch_size, history_size, seq_len, 2*dim, device=history.device)
        
        # 处理null embedding的情况
        # 扩展temporal_embeddings以便广播 [seq_len, 2*dim] -> [1, 1, seq_len, 2*dim]
        expanded_temporal = temporal_embeddings.unsqueeze(0).unsqueeze(0)
        
        # 对所有位置添加temporal embeddings (无论是否为null)
        null_plus_temporal = null_embedding.unsqueeze(0).unsqueeze(0).unsqueeze(0) + expanded_temporal
        
        # 使用掩码选择应用null embedding的位置
        null_mask = is_null.unsqueeze(-1).expand(-1, -1, -1, 2*dim)
        
        # 为非null位置准备embeddings
        # 收集所有需要的位置embeddings
        gathered_pos_embeddings = pos_embeddings[torch.arange(batch_size).unsqueeze(-1).unsqueeze(-1), 
                                                history.clamp(0, candidate_size-1)]
        
        # 添加temporal embeddings
        pos_plus_temporal = gathered_pos_embeddings + expanded_temporal
        
        # 使用掩码合并两种情况
        fused_embeddings = torch.where(null_mask, null_plus_temporal, pos_plus_temporal)
        
        return fused_embeddings
    else:
        batch_size, seq_len = history.shape
        
        # 创建掩码
        is_null = (history == candidate_size)
        
        # 创建结果tensor
        fused_embeddings = torch.zeros(batch_size, seq_len, 2*dim, device=history.device)
        
        # 扩展temporal_embeddings [seq_len, 2*dim] -> [1, seq_len, 2*dim]
        expanded_temporal = temporal_embeddings.unsqueeze(0)
        
        # null加上temporal的情况
        null_plus_temporal = null_embedding.unsqueeze(0).unsqueeze(0) + expanded_temporal
        
        # 使用掩码选择应用null embedding的位置
        null_mask = is_null.unsqueeze(-1).expand(-1, -1, 2*dim)
        
        # 收集所有需要的位置embeddings
        gathered_pos_embeddings = pos_embeddings[torch.arange(batch_size).unsqueeze(-1), 
                                               history.clamp(0, candidate_size-1)]
        
        # 添加temporal embeddings
        pos_plus_temporal = gathered_pos_embeddings + expanded_temporal
        
        # 使用掩码合并两种情况
        fused_embeddings = torch.where(null_mask, null_plus_temporal, pos_plus_temporal)
        
        return fused_embeddings
    

class TrillNet(nn.Module):
    def __init__(self, dim, num_heads = 2, window_size=2, candidate_size=4):
        super(TrillNet, self).__init__()
        self.dim = dim
        self.gcn = nn.Linear(dim, dim)  # 线性变换
        self.relu = nn.ReLU()  # ReLU激活函数

        self.num_heads = num_heads
        self.candidate_size = candidate_size

        self.embedding = nn.Parameter(torch.randn(1, candidate_size, dim))

        self.null_embedding = nn.Parameter(torch.rand(2*dim))

        # self.MaskedMultiheadSelfAttention = MaskedMultiheadSelfAttention(2*dim, num_heads)
        self.TransformerEncoderLayer = TransformerEncoderLayer(2*dim, num_heads)
        self.TransformerEncoder = TransformerEncoder(self.TransformerEncoderLayer, ENCODER_NUMS)

        self.WindowBasedCrossAttention = WindowBasedCrossAttention(2*dim, window_size)

        self.InformationAggregationLayer = InformationAggregationLayer(dim)


    def forward(self, history, current_trajectory):
        batch_size = history.shape[0]
        E = self.embedding.expand(batch_size, -1, -1)
        A = get_graph(history, self.candidate_size, batch_size)  # 获取图的邻接矩阵
        A = Mytransform(A, self.candidate_size, batch_size)  # 进行图的拉普拉斯变换
        A = A.to(DEVICE)
        E = E.to(DEVICE)
        Eg = self.relu(self.gcn(A @ E))  # 应用线性层与激活函数
        # pos_embeddings = torch.cat((Eg, E),dim=-1)
        pos_embeddings = torch.cat((E, E),dim=-1)
        temporal_embeddings = generate_temporal_embeddings(history.shape[2], 2*self.dim).to(DEVICE)
        fused_history_embeddings = fuse_spatial_temporal(pos_embeddings, history, temporal_embeddings, self.dim, self.null_embedding, self.candidate_size, 0)
        fused_current_embeddings = fuse_spatial_temporal(pos_embeddings, current_trajectory, temporal_embeddings, self.dim, self.null_embedding, self.candidate_size, 1)

         # 将 history 和 current 都转换为合适的形状 [batch_size * trajectory_size, seq_len, dim]
        fused_history_embeddings = fused_history_embeddings.view(batch_size * history.shape[1], history.shape[2], -1).to(DEVICE)
        fused_current_embeddings = fused_current_embeddings.view(batch_size, -1, self.dim * 2).to(DEVICE)

        def create_mask_optimized(history, candidate_size, device):
            # 直接创建掩码矩阵
            return torch.where(
                (history == candidate_size).unsqueeze(2),  # 条件: [batch, history, 1, seq]
                torch.full((history.shape[0], history.shape[1], history.shape[2], history.shape[2]), float('-inf'), device=device),  # 若为真
                torch.zeros(history.shape[0], history.shape[1], history.shape[2], history.shape[2], device=device)  # 若为假
            )
        fused_history_embeddings_mask =create_mask_optimized(history, self.candidate_size, DEVICE)
        
        # fused_history_embeddings_mask = torch.zeros(history.shape[0], history.shape[1], history.shape[2], history.shape[2]).to(DEVICE)
        # for i in range(history.shape[0]):
        #     for j in range(history.shape[1]):
        #         for k in range(history.shape[2]):
        #             if history[i][j][k] == self.candidate_size:
        #                 fused_history_embeddings_mask[i][j][:,k] = float('-inf')

        fused_history_embeddings_mask = fused_history_embeddings_mask.reshape(history.shape[0] * history.shape[1], history.shape[2], history.shape[2])

        # mmsa_history_embeddings = self.MaskedMultiheadSelfAttention(fused_history_embeddings, fused_history_embeddings_mask)
        mmsa_history_embeddings = self.TransformerEncoder(fused_history_embeddings, fused_history_embeddings_mask)

        def create_current_mask_optimized(current_trajectory, candidate_size, device):
            # 直接创建掩码矩阵
            return torch.where(
                (current_trajectory == candidate_size).unsqueeze(1),  # 条件: [batch, 1, seq]
                torch.full((current_trajectory.shape[0], current_trajectory.shape[1], current_trajectory.shape[1]), float('-inf'), device=device),  # 若为真
                torch.zeros(current_trajectory.shape[0], current_trajectory.shape[1], current_trajectory.shape[1], device=device)  # 若为假
            )
        fused_current_embeddings_mask = create_current_mask_optimized(current_trajectory, self.candidate_size, DEVICE)

        # fused_current_embeddings_mask = torch.zeros(current_trajectory.shape[0], current_trajectory.shape[1], current_trajectory.shape[1]).to(DEVICE)
        # for i in range(current_trajectory.shape[0]):
        #     for j in range(current_trajectory.shape[1]):
        #         if current_trajectory[i][j] == self.candidate_size:
        #             fused_current_embeddings_mask[i][:,j] = float('-inf')

        # mmsa_current_embeddings = self.MaskedMultiheadSelfAttention(fused_current_embeddings, fused_current_embeddings_mask)
        mmsa_current_embeddings = self.TransformerEncoder(fused_current_embeddings, fused_current_embeddings_mask)
        
        # # 将历史轨迹的输出重新形状为原始形状 [batch_size, trajectory_size, seq_len, dim]
        # mmsa_history_embeddings = mmsa_history_embeddings.view(batch_size, history.shape[1], history.shape[2], 2*self.dim)
        # wca_history_embeddings = mmsa_history_embeddings.view(batch_size * history.shape[1], history.shape[2], -1)

        temp_current_embeddings = mmsa_current_embeddings.unsqueeze(1).repeat(1, history.shape[1], 1, 1)
        temp_current_embeddings = temp_current_embeddings.view(batch_size * history.shape[1], history.shape[2], -1)

        wca_history_embeddings = self.WindowBasedCrossAttention(mmsa_history_embeddings, temp_current_embeddings)
        wca_history_embeddings = wca_history_embeddings.view(batch_size, history.shape[1], history.shape[2], 2*self.dim)

        final_embeddings = self.InformationAggregationLayer(wca_history_embeddings, mmsa_current_embeddings)


        # print("results:===================>")
        # print(f"History attention out shape: {wca_history_embeddings.shape}")
        # print(f"Current attention out shape: {wca_current_embeddings.shape}")
        # print("end==>")
        return final_embeddings @ self.embedding.transpose(-2, -1)

if __name__ == "__main__":    

    history_size = 10
    net = TrillNet(dim=EMBED_DIM, num_heads=NHEAD, window_size=WINDOW_SIZE, candidate_size=CANDIDATE_SIZE)
    
    # 模拟输入数据
    history =torch.randint(0, CANDIDATE_SIZE+1, (BATCH_SIZE, history_size, SEQ_LEN)) # 假设有两个批次，history 是一个形状为 [batch_size, history_size, seq_len] 的张量
    print(history.shape)
    current_trajectory = torch.randint(0, CANDIDATE_SIZE+1, (BATCH_SIZE, SEQ_LEN)) # 假设有两个批次，current_trajectory 是一个形状为 [batch_size, seq_len] 的张量
    print(current_trajectory.shape)

    p = net(history, current_trajectory)  # 通过网络进行推理
    print("prob:==================>")
    print(p.shape)
