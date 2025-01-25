import pandas as pd
import ast

# 读取 CSV 文件
import csv

# 读取CSV文件
input_file = '/mnt/HardDisk/datasets/SOT/GOT-10K/train_ann.csv'
output_file = '/mnt/HardDisk/datasets/SOT/GOT-10K/train_ann_filter.csv'

def bbox_condition(bbox):
    # 计算面积 (宽度 * 高度)
    width = bbox[2]
    height = bbox[3] 
    area = width * height
    # print(area)
    # 检查条件
    return area >= 10 and all(i >= 0 for i in bbox)

# 打开输入和输出文件
with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    
    # 写入标题行
    writer.writeheader()
    
    for row in reader:
        bbox = eval(row['bbox'])
             
        # 过滤条件
        if bbox_condition(bbox):
            writer.writerow(row)
        else:
            print(bbox)

print("过滤后的DataFrame已保存到 'filtered_file.csv'")

# 保存处理后的数据
