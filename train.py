import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from trill import TrillNet
from config import *
from data2list import data2list
import random
import numpy as np
import os
    
if __name__ == "__main__":
    # net = TrillNet(dim=EMBED_DIM, window_size=WINDOW_SIZE, candidate_size=CANDIDATE_SIZE)
    net = TrillNet(dim=EMBED_DIM, num_heads=NHEAD, window_size=WINDOW_SIZE, candidate_size=CANDIDATE_SIZE)
    if os.path.exists("trillnet_weights.pth"):
        checkpoints = torch.load("trillnet_weights.pth", map_location=DEVICE)
        net.load_state_dict(checkpoints)
        print("YES!!!!")
    net = net.to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=5e-5)  # Adam优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    
    input_dir = "final_user_augmented"
    users_trajectories = data2list(input_dir)
    for epoch in range(EPOCHS):
        for user_trajectories in users_trajectories:
            user_trajectories = np.array(user_trajectories)
            for i in range(3, len(user_trajectories)):
                history = user_trajectories[:i,:]
                label_current_trajectory = [user_trajectories[i]]
                history = torch.tensor(history).unsqueeze(0)
                current_trajectory = torch.tensor(label_current_trajectory)

                seq_len = current_trajectory.shape[1]

                label_list = [ ]
                for i in range(seq_len):
                    if current_trajectory[0][i] != CANDIDATE_SIZE:
                        label_list.append(i)
                k = 10
                label_list = random.sample(label_list, k)
                for item in label_list:
                    current_trajectory[0][item] = CANDIDATE_SIZE

                history = history.to(DEVICE)
                current_trajectory = current_trajectory.to(DEVICE)
                p = net(history, current_trajectory)  # 通过网络进行推理


                batch_indices = torch.arange(p.shape[0], device=DEVICE).unsqueeze(1).expand(-1, len(label_list))
                item_indices = torch.tensor(label_list, device=DEVICE).unsqueeze(0).expand(p.shape[0], -1)
                
                # 一次性提取所有概率
                all_probs = p[batch_indices, item_indices]  # [batch_size, len(label_list)]
                
                # 一次性提取所有目标
                all_targets = torch.tensor(label_current_trajectory, device=DEVICE)[batch_indices, item_indices]
                
                # 计算所有损失
                all_probs = all_probs.squeeze(0)
                all_targets = all_targets.squeeze(0)
                all_losses = criterion(all_probs, all_targets)
                # 计算平均损失
                mean_loss = all_losses.mean()

                # train_loss = 0
                # for batch in range(history.shape[0]):
                #     for item in label_list:
                #         prob = p[batch][item]
                #         target = label_current_trajectory[batch][item]
                #         target = torch.tensor(target).to(DEVICE)
                #         loss = criterion(prob, target)
                #         train_loss += loss
                # mean_loss = train_loss / len(label_list) 

                optimizer.zero_grad()
                mean_loss.backward()
                optimizer.step()
                print(f"train_loss:{mean_loss.item()}")
           # 保存模型和E
        torch.save(net.state_dict(), 'trillnet_weights.pth')
    exit(0)

    # history = torch.randint(0, candidate_size, (batch_size, history_size, seq_len))
    # current_trajectory = torch.randint(0, candidate_size, (batch_size, seq_len))
    
    # for batch in range(batch_size):
    #     for i in range(history_size):
    #         for j in range(seq_len):
    #             history[batch][i][j] = j % candidate_size
    # for batch in range(batch_size):
    #     for i in range(seq_len):
    #         current_trajectory[batch][i] = i % candidate_size

    # print(history)
    # print(current_trajectory)
    
    # dataset = TrajectoryDataset(history, current_trajectory)
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # for epoch in range(Epoch):
    #     for batch_history, batch_trajectory in dataloader:
    #         p = net(batch_history, batch_trajectory)  # 通过网络进行推理
            
    #         train_loss = 0
    #         for batch in range(batch_history.shape[0]):
    #             for i in range(seq_len):
    #                 prob = p[batch][i]
    #                 target = batch_trajectory[batch][i]
    #                 loss = criterion(prob, target)
    #                 train_loss += loss
            
    #         optimizer.zero_grad()
    #         train_loss.backward()
    #         optimizer.step()
    #         print(f"train_loss:{train_loss.item()}")
    #     print("==========================>")

    # # 保存模型和E
    # torch.save(net.state_dict(), 'trillnet_weights.pth')
