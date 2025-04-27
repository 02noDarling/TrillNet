from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import os
import numpy as np
from config import *
from inference import *
import random

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# ID 到经纬度转换
def id_to_coord(id):
    if id is None or id < 0 or id >= 1600:
        return None
    py = id // 40  # 纬度索引
    px = id % 40   # 经度索引
    lat = 39.75 + py * 0.00875
    lng = 116.15 + px * 0.01125
    return [lat, lng]

def id_to_random_coord(id):
    lon_min_total, lon_max_total = 116.15, 116.60
    lat_min_total, lat_max_total = 39.75, 40.10
    num_grids = 40
    grid_width = (lon_max_total - lon_min_total) / num_grids
    grid_height = (lat_max_total - lat_min_total) / num_grids

    def location_id_to_random_point(location_id):
        if location_id == -1:
            return None

        py = location_id // num_grids
        px = location_id % num_grids

        lon_min = lon_min_total + px * grid_width
        lat_min = lat_min_total + py * grid_height

        lon = lon_min + random.random() * grid_width
        lat = lat_min + random.random() * grid_height

        return (lat, lon)
    return location_id_to_random_point(id)

def coord_to_id(posx, posy):
    pos_x_list = np.linspace(116.15, 116.6, 41)
    pos_y_list = np.linspace(39.75, 40.1, 41)
    px = py = -1
    for i in range(len(pos_x_list)-1):
        if posx <= pos_x_list[i+1]:
            px = i
            break

    for i in range(len(pos_y_list)-1):
        if posy <= pos_y_list[i+1]:
            py = i
            break
    
    return py * 40 + px

# 读取历史轨迹
def read_trajectories_from_csv(file_path='history.csv'):
    if not os.path.exists(file_path):
        print(f"CSV file {file_path} not found")
        return []
    trajectories = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            try:
                trajectory = [float(x) if x != '-1' else None for x in row]
                trajectories.append(trajectory)
            except ValueError:
                print(f"Invalid row skipped: {row}")
    return trajectories

@app.route('/get_history', methods=['GET'])
def get_history():
    # trajectories = read_trajectories_from_csv('history.csv')
    # coord_trajectories = [
    #     [id_to_coord(id) for id in trajectory]
    #     for trajectory in trajectories
    # ]

    # print("Converted trajectories:", coord_trajectories)
    # return jsonify({'trajectories': coord_trajectories})

    coord_trajectories = read_trajectories_from_csv('history.csv')
    new_coord_trajectories = []
    for trajectory in coord_trajectories:
        new_trajectory = []
        for i in range(0, len(trajectory), 2):
            if trajectory[i] != None:
                new_trajectory.append([trajectory[i+1], trajectory[i]])
            else:
                new_trajectory.append(None)
        new_coord_trajectories.append(new_trajectory)
    print("Converted trajectories:", new_coord_trajectories)
    return jsonify({'trajectories': new_coord_trajectories})

@app.route('/convert_trajectory', methods=['POST'])
def convert_trajectory():
    data = request.json
    trajectory = data['trajectory']

    print(trajectory)

    points = [{'coord': id_to_coord(id) if id != -1 else None, 'inferred': False} for id in trajectory]
    points = [{'coord': [trajectory[id+1], trajectory[id]] if trajectory[id] != -1 else None, 'inferred': False} for id in range(0, len(trajectory), 2)]
    print(points)
    return jsonify({'points': points})

def trajectory_to_standard(trajectory):
    standard_trajectory = []
    for i in range(0, len(trajectory), 2):
        if trajectory[i] == -1:
            standard_trajectory.append(CANDIDATE_SIZE)
        else:
            standard_trajectory.append(coord_to_id(trajectory[i], trajectory[i+1]))

    # print("转换完成==========》")
    # print(len(trajectory))
    # print(standard_trajectory)
    return standard_trajectory

def get_history_from_csv(file_path='history.csv'):
    if not os.path.exists(file_path):
        print(f"CSV file {file_path} not found")
        return []
    trajectories = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            try:
                trajectory = [float(x) if x != '-1' else -1 for x in row]
                trajectories.append(trajectory)
            except ValueError:
                print(f"Invalid row skipped: {row}")
    return trajectories

def infer(current_trajectory):
    net = TrillNet(dim=EMBED_DIM, num_heads=NHEAD, window_size=WINDOW_SIZE, candidate_size=CANDIDATE_SIZE)
    if os.path.exists("trillnet_weights.pth"):
        checkpoints = torch.load("trillnet_weights.pth", map_location=DEVICE)
        net.load_state_dict(checkpoints)
        print("已经成功加载权重了!!!")
    net = net.to(DEVICE)
    history = get_history_from_csv()
    print(history[0])
    for i in range(len(history)):
        history[i] = trajectory_to_standard(history[i])
    
    output_file = "history_trajectories.csv"
    with open(output_file, 'w', newline='',encoding='utf-8') as file:
        writer = csv.writer(file)
        for history_trajectory in history:
            for i in range(len(history_trajectory)):
                if history_trajectory[i] ==CANDIDATE_SIZE:
                    history_trajectory[i] = -1
            writer.writerow(history_trajectory)


    print("历史轨迹如下")
    for history_trajectory in history:
        print(history_trajectory)


    # print(history)
    # current_trajectory = trajectory_to_standard(current_trajectory)

    # history = torch.randint(0, CANDIDATE_SIZE+1, (batch_size, history_size, SEQ_LEN))
    # current_trajectory = torch.randint(0, CANDIDATE_SIZE+1, (batch_size, SEQ_LEN))
    history = torch.tensor([history])
    current_trajectory = torch.tensor([current_trajectory])

    print(history.shape)
    print(current_trajectory.shape)

    current_trajectory = batch_trajectory_recover(net, history, current_trajectory)
    print("预测轨迹如下！！！！！")
    print(current_trajectory)
    current_trajectory = current_trajectory.tolist()[0]
    return current_trajectory


@app.route('/complete_trajectory', methods=['POST'])
def complete_trajectory():
    data = request.json
    trajectory = data['trajectory']
    current_trajectory = trajectory_to_standard(trajectory)
    complete_current_trajectory = infer(current_trajectory)

    completed_points = []
    counts = 0
    for i in range(16, 40):
        if current_trajectory[i] == CANDIDATE_SIZE:
            completed_points.append({'coord': id_to_random_coord(complete_current_trajectory[i]), 'inferred': True})
            counts += 1
        else:
            completed_points.append({'coord': [trajectory[i*2+1], trajectory[i*2]], 'inferred': False})
    
    print(f"数量：{counts}")
    return jsonify({'points': completed_points})

    completed_points = []
    last_valid_coord = None
    for i, id in enumerate(trajectory):
        if id == -1:
            # 线性插值：使用前后有效点的平均值
            next_valid_coord = None
            for j in range(i + 1, len(trajectory)):
                if trajectory[j] != -1:
                    next_valid_coord = id_to_coord(trajectory[j])
                    break
            if last_valid_coord and next_valid_coord:
                coord = [
                    (last_valid_coord[0] + next_valid_coord[0]) / 2,
                    (last_valid_coord[1] + next_valid_coord[1]) / 2
                ]
            else:
                coord = [39.90 + i * 0.01, 116.40 + i * 0.01]  # 回退默认值
            completed_points.append({'coord': coord, 'inferred': True})
        else:
            coord = id_to_coord(id)
            completed_points.append({'coord': coord, 'inferred': False})
            last_valid_coord = coord
    return jsonify({'points': completed_points})

if __name__ == '__main__':
    app.run(debug=True)