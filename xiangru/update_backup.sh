#!/bin/bash

# 设置源文件夹和目标文件夹
source_folder="/home/xiangru/Verifier_Development/complete_verifier/nerf"
destination_folder="/home/xiangru/File_Delivery/xiangru"

# 确保目标文件夹存在
mkdir -p "$destination_folder"

# 使用 rsync 复制文件，排除 data 文件夹
rsync -av --progress \
    --exclude '__pycache__' \
    --exclude 'data' \
    --exclude 'data2' \
    "$source_folder"/ "$destination_folder"/