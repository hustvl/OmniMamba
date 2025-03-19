import os
from datetime import datetime
from typing import *




def find_latest_model_bin(base_path):
    # 1. 找到时间最晚的子文件夹
    subfolders = [os.path.join(base_path, folder) for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
    if not subfolders:
        raise ValueError("指定路径下没有子文件夹")
    
    latest_folder = max(subfolders, key=lambda folder: extract_timestamp_from_name(os.path.basename(folder)))

    # 2. 找到最新子文件夹中最大的 checkpoint 文件夹
    checkpoint_folders = [
        folder for folder in os.listdir(latest_folder)
        if os.path.isdir(os.path.join(latest_folder, folder)) and folder.startswith("checkpoint-")
    ]
    if not checkpoint_folders:
        raise ValueError(f"路径 {latest_folder} 下没有以 'checkpoint-xxxx' 命名的文件夹")
    
    max_checkpoint = max(
        checkpoint_folders,
        key=lambda folder: int(folder.split("-")[1])
    )

    # 3. 获取最大 checkpoint 文件夹中的 pytorch_model.bin 文件路径
    model_bin_path = os.path.join(latest_folder, max_checkpoint, "pytorch_model.bin")
    if not os.path.isfile(model_bin_path):
        raise FileNotFoundError(f"文件 {model_bin_path} 不存在")
    
    return model_bin_path

def extract_timestamp_from_name(folder_name):
    try:
        timestamp_str = "_".join(folder_name.split("_")[1:])  # 提取日期时间部分
        return datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        raise ValueError(f"文件夹名称格式无效：{folder_name}")
