import torch
import torch.nn as nn
import torch.nn.functional as F

class InformationAggregationLayer(nn.Module):
    def __init__(self, embed_dim):
        super(InformationAggregationLayer, self).__init__()
        self.embed_dim = embed_dim
        self.wc = nn.Linear(2*embed_dim, 2*embed_dim)
        self.wh = nn.Linear(2*embed_dim, 2*embed_dim)
        self.qt = nn.Linear(2*embed_dim, 1)
        self.wf = nn.Linear(4*embed_dim ,embed_dim)
    
    def forward(self, wca_history_embeddings, wca_current_embeddings):
        # batch_size, history_size, seq_len, _ = wca_history_embeddings.shape
        # alpha = torch.zeros(batch_size, seq_len, history_size)
        # for batch in range(batch_size):
        #     for i in range(seq_len):
        #         for j in range(history_size):
        #             t = self.qt(F.sigmoid(self.wc(wca_current_embeddings[batch][i].unsqueeze(0)) + self.wh(wca_history_embeddings[batch][j][i].unsqueeze(0))))
        #             alpha[batch][i][j] = t
        # # print(alpha)
        # final_embeddings = torch.zeros(batch_size, seq_len, self.embed_dim)

        # for batch in range(batch_size):
        #     for i in range(seq_len):
        #         sum = torch.zeros(1, 2*self.embed_dim)
        #         for j in range(history_size):
        #             sum += alpha[batch][i][j]*wca_history_embeddings[batch][j][i]
        #         final_embeddings[batch][i] = self.wf(torch.cat((wca_current_embeddings[batch][i].unsqueeze(0),sum),dim=-1)).squeeze(0)


        batch_size, history_size, seq_len, embed_dim = wca_history_embeddings.shape
        
        # 重塑张量以进行批量计算
        # [batch_size, seq_len, embed_dim] -> [batch_size*seq_len, 1, embed_dim]
        current_flat = wca_current_embeddings.view(batch_size * seq_len, 1, -1)
        
        # [batch_size, history_size, seq_len, embed_dim] -> [batch_size*seq_len, history_size, embed_dim]
        history_flat = wca_history_embeddings.permute(0, 2, 1, 3).contiguous().view(batch_size * seq_len, history_size, -1)
        
        # 计算注意力权重
        # [batch_size*seq_len, 1, embed_dim] -> [batch_size*seq_len, 1, hidden_dim]
        wc_out = self.wc(current_flat)
        
        # [batch_size*seq_len, history_size, embed_dim] -> [batch_size*seq_len, history_size, hidden_dim]
        wh_out = self.wh(history_flat)
        
        # 广播加法 [batch_size*seq_len, 1, hidden_dim] + [batch_size*seq_len, history_size, hidden_dim]
        # 结果: [batch_size*seq_len, history_size, hidden_dim]
        combined = F.sigmoid(wc_out + wh_out)
        
        # 计算注意力分数 [batch_size*seq_len, history_size, 1]
        alpha_flat = self.qt(combined)
        
        # 重新塑造为 [batch_size, seq_len, history_size]
        alpha = alpha_flat.view(batch_size, seq_len, history_size)
        
        # 计算注意力加权和
        # 扩展 alpha 以便进行广播 [batch_size, seq_len, history_size, 1]
        alpha_expanded = alpha.unsqueeze(-1)
        
        # 转置 history 使其形状为 [batch_size, seq_len, history_size, embed_dim]
        history_transposed = wca_history_embeddings.permute(0, 2, 1, 3)
        
        # 对历史嵌入进行加权 [batch_size, seq_len, history_size, embed_dim]
        weighted_history = alpha_expanded * history_transposed
        
        # 沿着历史维度求和 [batch_size, seq_len, embed_dim]
        history_sum = weighted_history.sum(dim=2)
        
        # 准备连接
        # 扩展当前嵌入的维度以便连接
        current_expanded = wca_current_embeddings.unsqueeze(2)
        history_sum_expanded = history_sum.unsqueeze(2)
        
        # 连接 [batch_size, seq_len, 1, embed_dim*2]
        concatenated = torch.cat((current_expanded, history_sum_expanded), dim=-1)
        
        # 应用最终的线性变换 [batch_size, seq_len, 1, final_dim]
        final_embeddings_expanded = self.wf(concatenated)
        
        # 去掉多余的维度 [batch_size, seq_len, final_dim]
        final_embeddings = final_embeddings_expanded.squeeze(2)
        
        return final_embeddings

if __name__ == "__main__":
    batch_size = 2
    history_size =2
    seq_len = 4
    embed_dim = 5

    wca_history_embeddings = torch.randn(batch_size, history_size, seq_len, 2*embed_dim)  # 随机输入
    wca_current_embeddings = torch.randn(batch_size, seq_len, 2*embed_dim)  # 随机输入
    
    net = InformationAggregationLayer(embed_dim)
    final_embeddings =net(wca_history_embeddings, wca_current_embeddings)
    print(final_embeddings)
