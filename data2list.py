import os
from config import *

def data2list(dir_path):
    users_trajectories = []

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)

        user_trajectories = []
        with open(file_path, encoding='utf-8') as f:
            lines = f.readlines()

        # print("原始行数:", len(lines))

        for line in lines:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            parts = line.split(',')
            for i, item in enumerate(parts):
                parts[i] = int(item)
            for i,item in enumerate(parts):
                if item == -1:
                    parts[i] = CANDIDATE_SIZE
            user_trajectories.append(parts)
        
        users_trajectories.append(user_trajectories)
    
    return users_trajectories

if __name__ == "__main__":
    dir_path = "data"
    users_trajectories = data2list(dir_path)
    print(users_trajectories)



    