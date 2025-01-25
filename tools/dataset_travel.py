import warnings
from typing import Dict, Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from model_training.dataset import get_tracking_datasets
from model_training.train.fear_lightning_model import FEARLightningModel
from model_training.train.trainer import get_trainer
from model_training.utils import prepare_experiment, create_logger
import cv2
import numpy as np


logger = create_logger(__name__)
warnings.filterwarnings("ignore")


@hydra.main(config_name="fear_tracker", config_path="../model_training/config")
def run_experiment(hydra_config: DictConfig) -> None:
    config = prepare_experiment(hydra_config)
    model = instantiate(config["model"])
    train_dataset, val_dataset = get_tracking_datasets(config)
    for item in train_dataset:
        image = item["TRACKER_TARGET_TEMPLATE_IMAGE_KEY"].numpy()  # 转换为 numpy 数组
        bbox = item["TRACKER_TEMPLATE_BBOX_KEY"].numpy()  # 转换为 numpy 数组
        print(image.shape)
       # 调整图像通道顺序 (C, H, W) -> (H, W, C)
        image = np.transpose(image, (1, 2, 0))
        print(image.shape)
        image = np.ascontiguousarray(image)

        # 将图像从浮点数 [0, 1] 转换为整数 [0, 255]
        # image = (image * 255).astype(np.uint8)

        # 确保图像布局是连续的

        # 绘制 bbox
        x, y, w, h = bbox  # 假设 bbox 是 [x, y, w, h] 格式
        x1, y1 = int(x), int(y)  # 左上角坐标
        x2, y2 = int(x + w), int(y + h)  # 右下角坐标
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制绿色矩形框
        cv2.imshow("Image with BBox", image)
        cv2.waitKey(0)  # 等待按键

if __name__ == "__main__":
    run_experiment()
