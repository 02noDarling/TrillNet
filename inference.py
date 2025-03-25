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
    history_size = 4
    seq_len = 5
    candidate_size = 4
    embed = 9

    # 加载模型和E
    checkpoint = torch.load('trillnet_weights_and_E.pth')
    net = TrillNet(dim=embed)  # 根据实际模型设置dim
    net.load_state_dict(checkpoint['model_state_dict'])
    E = checkpoint['E']  # 这里恢复E为[candidate_size, dim]
    
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
    
    dataset = TrajectoryDataset(history, current_trajectory, E)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for batch_history, batch_trajectory, batch_E in dataloader:
        final_embeddings = net(candidate_size, batch_history, batch_trajectory, batch_E)  # 通过网络进行推理
        p = torch.matmul(final_embeddings, batch_E.transpose(-2,-1))
        
        train_loss = 0
        for batch in range(batch_history.shape[0]):
            for i in range(seq_len):
                prob = p[batch][i]
                pred = torch.argmax(p[batch][i])
                target = batch_trajectory[batch][i]
                print(f"pred:{pred},target:{target}")
        

