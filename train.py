import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from trill import TrillNet

class TrajectoryDataset(Dataset):
    def __init__(self, history, current_trajectory, E):
        self.history = history
        self.current_trajectory = current_trajectory
        self.E = E

    def __len__(self):
        return self.history.shape[0]  # batch_size

    def __getitem__(self, index):
        return self.history[index], self.current_trajectory[index], self.E
    
if __name__ == "__main__":
    batch_size = 10  # 设置 batch_size 大于 1
    history_size = 2
    seq_len = 4
    candidate_size = 4
    embed = 9
    Epoch = 200

    net = TrillNet(dim=embed)
    optimizer = optim.Adam(net.parameters(), lr=0.01)  # Adam优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    
    history = torch.randint(0, candidate_size, (batch_size, history_size, seq_len))
    current_trajectory = torch.randint(0, candidate_size, (batch_size, seq_len))
    
    for batch in range(batch_size):
        for i in range(history_size):
            for j in range(seq_len):
                history[batch][i][j] = j % candidate_size
    for batch in range(batch_size):
        for i in range(seq_len):
            current_trajectory[batch][i] = i % candidate_size

    print(history)
    print(current_trajectory)
    
    E = torch.randn(candidate_size, embed)  # 输入的特征矩阵，形状为 [1, candidate_size, dim]
    dataset = TrajectoryDataset(history, current_trajectory, E)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for epoch in range(Epoch):
        for batch_history, batch_trajectory, batch_E in dataloader:
            final_embeddings = net(candidate_size, batch_history, batch_trajectory, batch_E)  # 通过网络进行推理
            p = torch.matmul(final_embeddings, batch_E.transpose(-2,-1))
            
            train_loss = 0
            for batch in range(batch_history.shape[0]):
                for i in range(seq_len):
                    prob = p[batch][i]
                    target = batch_trajectory[batch][i]
                    loss = criterion(prob, target)
                    train_loss += loss
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            print(f"train_loss:{train_loss.item()}")
        print("==========================>")

    # 保存模型和E
    torch.save({
        'model_state_dict': net.state_dict(),
        'E': E.squeeze(0),  # 将E从[1, candidate_size, dim]转为[candidate_size, dim]
    }, 'trillnet_weights_and_E.pth')
