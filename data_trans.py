import os
import csv
import random
input_dir = "/Users/hudaili/Desktop/VsCodeProjects/TrillNet/final_user_augmented"
save_dir = "/Users/hudaili/Desktop/VsCodeProjects/TrillNet/final_user_augmented_coord"
os.makedirs(save_dir, exist_ok=True)


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

    return (lon, lat)

for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)

    trajectories = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            trajectory = []
            for item in row:
                item = int(item)
                if item == -1:
                    trajectory.append(-1)
                    trajectory.append(-1)
                else:
                    lon, lat = location_id_to_random_point(item)
                    trajectory.append(lon)
                    trajectory.append(lat)
            trajectories.append(trajectory)
    output_file_path = os.path.join(save_dir, file_name)
    with open(output_file_path, 'w', newline='',encoding='utf-8') as file:
        writer = csv.writer(file)
        for trajectory in trajectories:
            writer.writerow(trajectory)
