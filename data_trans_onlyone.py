import os
import csv
import random


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

trajectory = []
new_trajectory = []
for item in trajectory:
    if item == -1:
        new_trajectory.append(-1)
        new_trajectory.append(-1)
    else:
        lon, lat = location_id_to_random_point(item)
        new_trajectory.append(lon)
        new_trajectory.append(lat)
for i,item in enumerate(trajectory):
    if i != len(trajectory)-1:
        print(item,end="")
        print(',',end="")
    else:
        print(item)
