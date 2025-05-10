import csv
import os
import random

# 计算曼哈顿距离
def manhattan_distance(id1, id2, num_grids=40):
    px1, py1 = id1 % num_grids, id1 // num_grids
    px2, py2 = id2 % num_grids, id2 // num_grids
    return abs(px1 - px2) + abs(py1 - py2)

# 生成新的位置ID，要求与原ID的曼哈顿距离不超过10
def generate_new_id(original_id, prob, num_grids=40, max_distance=0):
    if random.random() > prob:
        return original_id  # 如果不变，则返回原ID

    # 计算原始位置的行列坐标
    px, py = original_id % num_grids, original_id // num_grids
    possible_new_ids = []

    # 查找与原始ID曼哈顿距离不超过max_distance的所有新ID
    for dx in range(-max_distance, max_distance + 1):
        for dy in range(-max_distance, max_distance + 1):
            nx, ny = px + dx, py + dy
            if 0 <= nx < num_grids and 0 <= ny < num_grids:
                new_id = ny * num_grids + nx
                if manhattan_distance(original_id, new_id, num_grids) <= max_distance:
                    possible_new_ids.append(new_id)

    # 随机选取一个新的ID
    return random.choice(possible_new_ids) if possible_new_ids else original_id

# 处理CSV文件，更新每一行的位置ID
def update_trajectory(csv_file, prob=0.3, num_grids=40, max_distance=10):
    updated_trajectories = []

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            updated_row = []
            for loc in row:
                loc = int(loc) if loc != "" else -1  # 处理空字符串为-1
                if loc != -1:  # 只有非-1的ID才会进行更新
                    new_loc = generate_new_id(loc, prob, num_grids, max_distance)
                    updated_row.append(str(new_loc))
                else:
                    updated_row.append("-1")
            updated_trajectories.append(updated_row)

    return updated_trajectories

# 保存更新后的轨迹到新CSV文件
def save_updated_trajectories(updated_trajectories, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(updated_trajectories)

# 处理指定目录下的所有CSV文件
def process_directory(input_directory, output_directory, prob=0.3, distance=10):
    # 遍历目录下所有的CSV文件
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            input_file = os.path.join(input_directory, filename)
            output_file = os.path.join(output_directory, filename)

            print(f"处理文件: {input_file}")

            updated_trajectories = update_trajectory(input_file, prob, max_distance=distance)
            save_updated_trajectories(updated_trajectories, output_file)

            print(f"更新后的数据已保存到 {output_file}")

# 使用示例
if __name__ == "__main__":
    input_directory = r"D:\VsCodeProjects\TrillNet\final_user_augmented"  # 输入的CSV文件目录
    output_directory = r"D:\VsCodeProjects\TrillNet\final_user_augmented_strength"  # 输出的更新后的CSV文件目录

    prob = 0.15  # 30%的概率更新位置ID
    distance = 5

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    process_directory(input_directory, output_directory, prob, distance)
    print("所有文件处理完成")
