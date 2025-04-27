import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from trill import TrillNet
from config import *
import os

class TrajectoryDataset(Dataset):
    def __init__(self, history, current_trajectory, E):
        self.history = history
        self.current_trajectory = current_trajectory
        self.E = E

    def __len__(self):
        return self.history.shape[0]  # batch_size

    def __getitem__(self, index):
        return self.history[index], self.current_trajectory[index], self.E

def batch_trajectory_recover(net, history, current_trajectory):
    p = net(history, current_trajectory)
    start, end = 16, 39
    batch_size = history.shape[0]
    for batch in range(batch_size):
        for i in range(start,end+1):
            if current_trajectory[batch][i] == CANDIDATE_SIZE:
                current_trajectory[batch][i] = torch.argmax(p[batch][i])
    return current_trajectory
    

if __name__ == "__main__":
    batch_size = 1
    history_size = 3
    net = TrillNet(dim=EMBED_DIM, num_heads=NHEAD, window_size=WINDOW_SIZE, candidate_size=CANDIDATE_SIZE)
    if os.path.exists("trillnet_weights.pth"):
        checkpoints = torch.load("trillnet_weights.pth", map_location=DEVICE)
        net.load_state_dict(checkpoints)
        print("YES!!!!")
    net = net.to(DEVICE)
    
    history = torch.randint(0, CANDIDATE_SIZE+1, (batch_size, history_size, SEQ_LEN))
    current_trajectory = torch.randint(0, CANDIDATE_SIZE+1, (batch_size, SEQ_LEN))

    current_trajectory = torch.tensor([[ 301,  1600,  484,  1600,  262, 1482,  969,  520,  982,  518, 1595,  663,
          906,  977,  876,  478, 1126,  611, 1331,  1600,  497,  392,  370, 1028,
          896, 1509, 1600,  867,  863,  1600, 1402,  616,  1600, 1600,  714, 1450,
           10,  250, 1360,   64,  876,  324, 1253,  322,  497,  160, 1193, 1058]])

    current_trajectory = batch_trajectory_recover(net, history, current_trajectory)
    print(current_trajectory)
    exit(0)
    
    # for batch in range(batch_size):
    #     for i in range(history_size):
    #         for j in range(seq_len):
    #             history[batch][i][j] = j % candidate_size
    # for batch in range(batch_size):
    #     for i in range(seq_len):
    #         current_trajectory[batch][i] = i % candidate_size

    # print(history)
    # print(current_trajectory)
    
    # dataset = TrajectoryDataset(history, current_trajectory, E)
    # dataloader = DataLoader(dataset, gitbatch_size=2, shuffle=True)
    
    # for batch_history, batch_trajectory, batch_E in dataloader:
    #     final_embeddings = net(candidate_size, batch_history, batch_trajectory, batch_E)  # 通过网络进行推理
    #     p = torch.matmul(final_embeddings, batch_E.transpose(-2,-1))
        
    #     train_loss = 0
    #     for batch in range(batch_history.shape[0]):
    #         for i in range(seq_len):
    #             prob = p[batch][i]
    #             pred = torch.argmax(p[batch][i])
    #             target = batch_trajectory[batch][i]
    #             print(f"pred:{pred},target:{target}")
        

