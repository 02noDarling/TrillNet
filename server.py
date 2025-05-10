from flask import Flask, request, jsonify, session
from flask_cors import CORS
import csv
import os
import numpy as np
from config import *
from inference import *
import random
import sqlite3
import json
from flask import send_from_directory

app = Flask(__name__)
app.secret_key = '3d3f8b3e9e7f4c2a1b0d8e7f6a5c4b3a'  # 必须设置密钥
CORS(app, supports_credentials=True, origins=["http://localhost:5000/"])

# 数据库初始化
def init_db():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    # 创建用户表
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
    # 创建实验表
    c.execute('''CREATE TABLE IF NOT EXISTS experiments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  name TEXT NOT NULL,
                  history_trajectory TEXT,
                  current_trajectory TEXT,
                  predicted_trajectory TEXT,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

init_db()

@app.route('/login')
def serve_login():
    return send_from_directory('static', 'login.html')

@app.route('/experiment_ui')
def serve_experiment_ui():
    return send_from_directory('static', 'experiment_ui.html')

@app.route('/trajectory')
def serve_trajectory():
    return send_from_directory('static', 'trajectory.html')

# 用户认证相关
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')  # 实际应哈希存储
    try:
        conn = sqlite3.connect('data.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                 (username, password))
        conn.commit()
        return jsonify({'success': True})
    except sqlite3.IntegrityError:
        return jsonify({'error': '用户名已存在'}), 400

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?", 
             (username, password))
    user = c.fetchone()
    if user:
        print("here")
        print(user[0])
        session['user_id'] = user[0]
        session['username'] = username
        print(session)
        return jsonify({'success': True, 'message': '登录成功'})
    return jsonify({'success': False, 'message': '用户名或密码错误'})

@app.route('/check_login', methods=['GET'])
def check_login():
    print(session)
    if 'user_id' in session:
        return jsonify({'success': True, 'username': session['username']})
    return jsonify({'success': False})

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True})

# # 实验管理相关
# @app.route('/create_experiment', methods=['POST'])
# def create_experiment():
#     if 'user_id' not in session:
#         return jsonify({'error': '未登录'}), 401
#     exp_name = request.form['name']
#     file = request.files['file']
#     if file:
#         file_path = f'uploads/{exp_name}.csv'
#         file.save(file_path)
#         conn = sqlite3.connect('data.db')
#         c = conn.cursor()
#         try:
#             c.execute('''INSERT INTO experiments 
#                          (user_id, name, history_path) 
#                          VALUES (?, ?, ?)''',
#                      (session['user_id'], exp_name, file_path))
#             conn.commit()
#             return jsonify({'success': True})
#         except sqlite3.IntegrityError:
#             return jsonify({'error': '实验名称重复'}), 400
# # 创建实验

@app.route('/create_experiment', methods=['POST'])
def create_experiment():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '未登录'}), 401
    data = request.json
    name = data.get('name')
    history_trajectory = data.get('historyTrajectory')
    # if not name or not history_trajectory or len(history_trajectory) != 48:
    #     return jsonify({'success': False, 'message': '实验名称和48个点的历史轨迹必填'})
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("SELECT id FROM experiments WHERE user_id = ? AND name = ?", (session['user_id'], name))
    if c.fetchone():
        conn.close()
        return jsonify({'success': False, 'message': '实验名称已存在'})
    c.execute("INSERT INTO experiments (user_id, name, history_trajectory) VALUES (?, ?, ?)",
              (session['user_id'], name, json.dumps(history_trajectory)))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': '实验创建成功'})

# @app.route('/get_experiments')
# def get_experiments():
#     print("HHHHH")
#     print(session)
#     if 'user_id' not in session:
#         return jsonify({'error': '未登录'}), 401
#     conn = sqlite3.connect('data.db')
#     c = conn.cursor()
#     c.execute('''SELECT id, name, current_trajectory, predicted_trajectory 
#                  FROM experiments WHERE user_id=?''', 
#              (session['user_id'],))
#     experiments = []
#     for row in c.fetchall():
#         experiments.append({
#             'id': row[0],
#             'name': row[1],
#             'has_current': bool(row[2]),
#             'has_predicted': bool(row[3])
#         })
#     print(experiments)
#     return jsonify({'experiments': experiments})

# 获取实验列表
@app.route('/get_experiments', methods=['GET'])
def get_experiments():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '未登录'}), 401
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("SELECT id, name, history_trajectory, current_trajectory, predicted_trajectory FROM experiments WHERE user_id = ?", (session['user_id'],))
    experiments = [{'id': row[0], 'name': row[1], 'historyTrajectory': json.loads(row[2]) if row[2] else None,
                    'currentTrajectory': json.loads(row[3]) if row[3] else None,
                    'predictedTrajectory': json.loads(row[4]) if row[4] else None} for row in c.fetchall()]
    conn.close()
    return jsonify({'success': True, 'experiments': experiments})

@app.route('/delete_experiment', methods=['POST'])
def delete_experiment():
    if 'username' not in session:
        return jsonify({'success': False, 'message': '未登录'}), 401
    data = request.json
    expId = data.get('expId')

    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM experiments WHERE id = ?", (expId,))
    if cursor.rowcount == 0:
        conn.close()
        return jsonify({'success': False, 'message': '实验不存在或无权限删除'})
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': '实验删除成功'})

# 原有轨迹处理函数保持不变，补充保存功能
@app.route('/save_trajectory', methods=['POST'])
def save_trajectory():
    data = request.json
    exp_id = data.get('exp_id')
    trajectory = data.get('trajectory')
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('''UPDATE experiments 
                 SET current_trajectory=?
                 WHERE id=?''', 
              (str(trajectory), exp_id))
    conn.commit()
    return jsonify({'success': True})

@app.route('/export_experiment', methods=['POST'])
def export_experiment():
    if 'user_id' not in session:
        return jsonify({'error': '未登录'}), 401
        
    exp_id = request.json.get('exp_id')
    
    # 连接数据库
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    
    try:
        # 查询实验数据
        c.execute('''SELECT * FROM experiments 
                     WHERE id=? AND user_id=?''',
                     (exp_id, session['user_id']))
        experiment = c.fetchone()

        history_trajectories = str(get_history_from_database(exp_id))
        
        if not experiment:
            return jsonify({'error': '实验不存在或无权访问'}), 404
            
        # 构建数据字典
        data = {
            '实验ID': experiment[0],
            '用户ID': experiment[1],
            '实验名称': experiment[2],
            '历史轨迹路径': history_trajectories,
            '当前轨迹': experiment[4],
            '预测轨迹': experiment[5]
        }
        
        # 构建文本内容
        txt_content = "\n".join(
            [f"{key}: {value}" for key, value in data.items()]
        )
        
        return jsonify({
            'success': True,
            'filename': f"experiment_id_{experiment[0]}_name_{experiment[2]}.txt",
            'content': txt_content
        })
        
    except Exception as e:
        print(f"导出失败: {str(e)}")
        return jsonify({'error': '服务器错误'}), 500

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

@app.route('/get_history', methods=['POST'])
def get_history():
    # trajectories = read_trajectories_from_csv('history.csv')
    # coord_trajectories = [
    #     [id_to_coord(id) for id in trajectory]
    #     for trajectory in trajectories
    # ]

    # print("Converted trajectories:", coord_trajectories)
    # return jsonify({'trajectories': coord_trajectories})
    data = request.json
    expId = data['expId']
    print("YYYYYYYYYYYY")
    print(expId)

    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("SELECT history_trajectory FROM experiments WHERE id = ?", (expId,))
    history_trajectories  = c.fetchone()[0]
    history_trajectories = json.loads(history_trajectories)
    print(len(history_trajectories))

    new_coord_trajectories = []
    for trajectory in history_trajectories:
        print(len(trajectory))
        new_trajectory = []
        for i in range(len(trajectory)):
            if trajectory[i][0] != -1:
                point = trajectory[i]
                new_trajectory.append([point[1], point[0]])
            else:
                new_trajectory.append(None)
        new_coord_trajectories.append(new_trajectory)


    # coord_trajectories = read_trajectories_from_csv('history.csv')
    # new_coord_trajectories = []
    # for trajectory in coord_trajectories:
    #     new_trajectory = []
    #     for i in range(0, len(trajectory), 2):
    #         if trajectory[i] != None:
    #             new_trajectory.append([trajectory[i+1], trajectory[i]])
    #         else:
    #             new_trajectory.append(None)
    #     new_coord_trajectories.append(new_trajectory)
    print("Converted trajectories:")
    for trajectory in new_coord_trajectories:
        print(trajectory)
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

def get_history_from_database(expId):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("SELECT history_trajectory FROM experiments WHERE id = ?", (expId,))
    history_trajectories  = c.fetchone()[0]
    history_trajectories = json.loads(history_trajectories)
    all_trajectories = []
    for trajectory in history_trajectories:
        new_trajectory = []
        for item in trajectory:
            new_trajectory.append(item[0])
            new_trajectory.append(item[1])
        all_trajectories.append(new_trajectory)
    # print(all_trajectories)
    return all_trajectories

def infer(current_trajectory, expId):
    net = TrillNet(dim=EMBED_DIM, num_heads=NHEAD, window_size=WINDOW_SIZE, candidate_size=CANDIDATE_SIZE)
    if os.path.exists("trillnet_weights.pth"):
        checkpoints = torch.load("trillnet_weights.pth", map_location=DEVICE)
        net.load_state_dict(checkpoints)
        print("已经成功加载权重了!!!")
    net = net.to(DEVICE)

    # history = get_history_from_csv()
    history = get_history_from_database(expId)
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
    expId = data['expId']

    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('''UPDATE experiments 
                 SET current_trajectory=?
                 WHERE id=?''', 
              (str(trajectory), expId))
    conn.commit()

    print("现有轨迹")
    print(trajectory)
    current_trajectory = trajectory_to_standard(trajectory)
    complete_current_trajectory = infer(current_trajectory, expId)

    completed_points = []
    recovery_trajectory = [-1 for i in range(96)]
    counts = 0
    for i in range(16, 40):
        if current_trajectory[i] == CANDIDATE_SIZE:
            point = id_to_random_coord(complete_current_trajectory[i])
            completed_points.append({'coord': point, 'inferred': True})
            counts += 1
            recovery_trajectory[2*i] = point[1]
            recovery_trajectory[2*i+1] = point[0]
        else:
            completed_points.append({'coord': [trajectory[i*2+1], trajectory[i*2]], 'inferred': False})
            recovery_trajectory[2*i] = trajectory[2*i]
            recovery_trajectory[2*i+1] = trajectory[2*i+1]
    
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('''UPDATE experiments 
                 SET predicted_trajectory=?
                 WHERE id=?''', 
              (str(recovery_trajectory), expId))
    conn.commit()

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

@app.route('/tiles/<int:z>/<int:x>/<int:y>.png')
def serve_tile(z, x, y):
    return send_from_directory('static/tiles', f'{z}/{x}/{y}.png')

if __name__ == '__main__':
    app.run(debug=True)