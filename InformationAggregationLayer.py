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
        batch_size, history_size, seq_len, _ = wca_history_embeddings.shape
        alpha = torch.zeros(batch_size, seq_len, history_size)
        for batch in range(batch_size):
            for i in range(seq_len):
                for j in range(history_size):
                    t = self.qt(F.sigmoid(self.wc(wca_current_embeddings[batch][i].unsqueeze(0)) + self.wh(wca_history_embeddings[batch][j][i].unsqueeze(0))))
                    alpha[batch][i][j] = t
        # print(alpha)
        final_embeddings = torch.zeros(batch_size, seq_len, self.embed_dim)

        for batch in range(batch_size):
            for i in range(seq_len):
                sum = torch.zeros(1, 2*self.embed_dim)
                for j in range(history_size):
                    sum += alpha[batch][i][j]*wca_history_embeddings[batch][j][i]
                final_embeddings[batch][i] = self.wf(torch.cat((wca_current_embeddings[batch][i].unsqueeze(0),sum),dim=-1)).squeeze(0)


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
