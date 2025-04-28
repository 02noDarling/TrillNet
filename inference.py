import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from trill import TrillNet
from config import *
import os
import csv
import random

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

    file_path = "/Users/hudaili/Desktop/VsCodeProjects/TrillNet/history_trajectories.csv"
    history = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            trajectory = []
            for item in row:
                item = int(item)
                if item == -1:
                    item =CANDIDATE_SIZE
                trajectory.append(item)
            history.append(trajectory)
    history = history[:-1]
    history = torch.tensor([history])
    current_trajectory = [
       -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,454,534,415,295,-1,178,58,61,180,-1,-1,-1,172,212,331,-1,330,-1,-1,406,287,-1,527,485,-1,-1,-1,-1,-1,-1,-1,-1
    ]
    for i,item in enumerate(current_trajectory):
        if item == -1:
            current_trajectory[i] =CANDIDATE_SIZE
    print(current_trajectory)

    label_current_trajectory = [current_trajectory.copy()]

    label_list = [ ]
    for i in range(48):
        if current_trajectory[i] != CANDIDATE_SIZE:
            label_list.append(i)
    k = 10
    label_list = random.sample(label_list, k)
    for item in label_list:
        current_trajectory[item] = CANDIDATE_SIZE
    print("去除了10个位置")
    print(current_trajectory)
    current_trajectory = torch.tensor([current_trajectory])

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    p = net(history, current_trajectory)
    
    batch_indices = torch.arange(p.shape[0], device=DEVICE).unsqueeze(1).expand(-1, len(label_list))
    item_indices = torch.tensor(label_list, device=DEVICE).unsqueeze(0).expand(p.shape[0], -1)
    
    # 一次性提取所有概率
    all_probs = p[batch_indices, item_indices]  # [batch_size, len(label_list)]
    
    # 一次性提取所有目标
    all_targets = torch.tensor(label_current_trajectory, device=DEVICE)[batch_indices, item_indices]
    
    # 计算所有损失
    all_probs = all_probs.squeeze(0)
    all_targets = all_targets.squeeze(0)

    print(all_targets)
    all_losses = criterion(all_probs, all_targets)
    # 计算平均损失
    mean_loss = all_losses.mean()
    print(f"mean_loss:{mean_loss.item()}")
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
        

