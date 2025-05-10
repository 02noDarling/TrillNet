import os
import argparse
import math
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def deg2num(lat_deg, lon_deg, zoom):
    """将经纬度转换为瓦片坐标 (x, y)"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def download_tile(x, y, z, output_dir, max_retries=3):
    """下载单个瓦片（含重试机制）"""
    url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    tile_path = os.path.join(output_dir, f"{z}/{x}/{y}.png")
    
    if os.path.exists(tile_path):
        print(f"跳过已存在的瓦片: {z}/{x}/{y}")
        return True

    os.makedirs(os.path.dirname(tile_path), exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=20)  # 增加超时时间
            if response.status_code == 200:
                with open(tile_path, 'wb') as f:
                    f.write(response.content)
                print(f"下载成功: {z}/{x}/{y}")
                return True
            else:
                print(f"尝试 {attempt+1}/{max_retries}: 下载失败（状态码 {response.status_code}）: {z}/{x}/{y}")
        except Exception as e:
            print(f"尝试 {attempt+1}/{max_retries}: 下载失败（{str(e)}）: {z}/{x}/{y}")
        
        time.sleep(2 ** attempt)  # 指数退避策略
    
    print(f"全部尝试失败: {z}/{x}/{y}")
    return False

def main():
    parser = argparse.ArgumentParser(description='下载OpenStreetMap瓦片')
    parser.add_argument('--lat', nargs=2, type=float, required=True, help='纬度范围（最小 最大）')
    parser.add_argument('--lon', nargs=2, type=float, required=True, help='经度范围（最小 最大）')
    parser.add_argument('--zoom', type=str, required=True, help='缩放级别（如 11-13）')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    args = parser.parse_args()

    # 解析缩放级别范围
    if '-' in args.zoom:
        min_zoom, max_zoom = map(int, args.zoom.split('-'))
    else:
        min_zoom = max_zoom = int(args.zoom)

    # 遍历所有缩放级别
    for z in range(min_zoom, max_zoom + 1):
        # 计算瓦片坐标范围
        x_min, y_max = deg2num(args.lat[0], args.lon[0], z)
        x_max, y_min = deg2num(args.lat[1], args.lon[1], z)

        # 生成所有瓦片坐标
        tiles = [(x, y, z) for x in range(x_min, x_max + 1) 
                        for y in range(y_min, y_max + 1)]

        # 使用更保守的并发控制
        with ThreadPoolExecutor(max_workers=4) as executor:  # 减少并发数
            futures = [executor.submit(download_tile, x, y, z, args.output_dir) for x, y, z in tiles]
            
            # 添加进度监控
            for future in as_completed(futures):
                future.result()  # 触发异常检查
                time.sleep(0.5)  # 增加请求间隔

if __name__ == "__main__":
    main()