import argparse
from abc import ABC
from collections import defaultdict
import fire
import os
import pandas as pd
import imagesize
import numpy as np
from tqdm import tqdm
import ast

header = [
    "sequence_id",
    "track_id",
    "frame_index",
    "img_path",
    "bbox",
    "frame_shape",
    "dataset",
    "presence",
    "near_corner",
]

import cv2


def show_data_frame(data_root, df):
    for index, row in df.iterrows():
        img_path = row["img_path"]
        img = cv2.imread(os.path.join(data_root, img_path))
        x, y, w, h = row["bbox"]
        x1, y1 = int(x), int(y)  # 左上角坐标
        x2, y2 = int(x + w), int(y + h)  # 右下角坐标
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制绿色矩形框
        cv2.imshow("Image with BBox", img)
        cv2.waitKey(0)  # 等待按键


def parser_got_10k_sub_seq(data_root, sub_seq_name):
    all_ext = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(os.path.join(data_root, sub_seq_name)):
        for file in filenames:
            name, ext = os.path.splitext(os.path.basename(file))
            all_ext[ext].append(os.path.join(sub_seq_name, file))
    max_len_key = max(all_ext, key=lambda k: len(all_ext[k]))
    if len(all_ext[max_len_key]) < 20:
        print(
            f"ignore sub folder {sub_seq_name} cus max length ext {max_len_key}, has only {len(all_ext[max_len_key]) } files"
        )
        return None
    all_img_list = all_ext[max_len_key]

    gt_file = os.path.join(data_root, sub_seq_name, "groundtruth.txt")
    cover_file = os.path.join(data_root, sub_seq_name, "cover.label")
    absence_file = os.path.join(data_root, sub_seq_name, "absence.label")
    cut_by_img_file = os.path.join(data_root, sub_seq_name, "cut_by_image.label")

    imgs_data = {
        int(os.path.splitext(os.path.basename(img_file))[0]): {
            "id": int(os.path.splitext(os.path.basename(img_file))[0]),
            "path": img_file,
        }
        for img_file in all_img_list
    }
    ids = list(imgs_data.keys())
    ids.sort()

    df = pd.DataFrame({"frame_index": ids})
    df["img_path"] = [imgs_data[id]["path"] for id in df["frame_index"]]

    df["frame_shape"] = [list(map(int, imagesize.get(os.path.join(data_root, imgpath)))) for imgpath in df["img_path"]]

    all_gt_items = []
    with open(gt_file, "r") as f:
        all_gt_items = [list(map(int, map(float, l.split(",")))) for l in f.readlines()]
    df["bbox"] = all_gt_items

    all_absence_flags = []
    with open(absence_file, "r") as f:
        all_absence_flags = [int(l) for l in f.readlines()]
    df["presence"] = all_absence_flags

    all_cut_by_img_flags = []
    with open(cut_by_img_file, "r") as f:
        all_cut_by_img_flags = [int(l) for l in f.readlines()]
    df["near_corner"] = all_cut_by_img_flags

    return df


def main(dataset_name, dataset_folder, output_csv_file):
    print(f"processing dataset for {dataset_name}")

    subfolders = []
    for dirpath, dirnames, filenames in os.walk(dataset_folder):
        for dirname in dirnames:
            subfolders.append(dirname)
    df = None
    for i, sub_dir_name in tqdm(enumerate(subfolders), total=len(subfolders)):
        subdf = parser_got_10k_sub_seq(dataset_folder, sub_dir_name)
        if subdf is None:
            continue
        subdf["dataset"] = "got10k"
        subdf["sequence_id"] = np.full(len(subdf), i)
        subdf["track_id"] = np.full(len(subdf), i)

        # show_data_frame(dataset_folder, subdf)
        # break


        if df is None:
            df = subdf
        else:
            df = pd.concat([df, subdf])


    df.to_csv(output_csv_file, index=False)


if __name__ == "__main__":
    fire.Fire(main)
