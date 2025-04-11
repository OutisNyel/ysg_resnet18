# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import cv2
from functools import lru_cache
import os
from PIL import Image
from preprocess import Preprocess as P
import random
import sqlite3
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


DB_PATH = 'data/fundus.db'
IMAGE_ROOT = 'P:/Datasets/ysg/ysg/Training_Dataset'
SQL = '''
SELECT
    left_path,
    right_path,
    diabetic,
    glaucoma,
    cataract,
    age_related_macular_degeneration,
    hypertensive_retinopathy,
    myopia,
    other_diseases
FROM 
    fundus
WHERE 
    is_test = 0 AND low_quality_image = 0;
'''


class FundusDataset(Dataset):
    def __init__(self,
                 is_train,
                 args,
                 db_path=DB_PATH,
                 image_root=IMAGE_ROOT,
                 sql=SQL
                 ):
        self.is_train = is_train
        self.args = args
        self.db_path = db_path
        self.image_root = image_root
        self.sql = sql

        self.data = self._load_data_from_db()
        self.transform = self.build_transform(self.is_train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        left_path = self.data[index][0]
        right_path = self.data[index][1]

        left_eye = cv2.imread(os.path.join(self.image_root, left_path))
        right_eye = cv2.imread(os.path.join(self.image_root, right_path))
        # 随机交换左右眼
        if random.random() < 0.5:
            t = left_eye
            left_eye = right_eye
            right_eye = t

        left_eye = P.center(left_eye)
        right_eye = P.center(right_eye)
        left_eye = P.reflective_pad(left_eye)
        right_eye = P.reflective_pad(right_eye)
        left_eye = P.random_rotate_flip(left_eye, random.randint(0, 4))
        right_eye = P.random_rotate_flip(right_eye, random.randint(0, 4))
        left_eye = P.clahe(left_eye)
        right_eye = P.clahe(right_eye)
        left_eye = P.resize(left_eye)
        right_eye = P.resize(right_eye)
        left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)
        right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)
        left_eye = Image.fromarray(left_eye)
        right_eye = Image.fromarray(right_eye)
        left_eye = self.transform(left_eye)
        right_eye = self.transform(right_eye)

        label = torch.tensor(self.data[index][2:], dtype=torch.float32)  # 将标签转换为张量
        return (left_eye, right_eye), label  # 返回处理后的图像和标签

    def build_transform(self, is_train):
        '''return a transform.
        train: PIL.Image/torch.Tensor
        eval: torch.Tensor'''
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        # train transform
        if is_train == 'train':
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                is_training=True,  # 启用训练模式的数据增强
                input_size=self.args.input_size, # 假如不指定输入大小，默认缩放到 224^2
                color_jitter=self.args.color_jitter,  # 颜色抖动强度, 默认 None
                auto_augment=self.args.aa,   # 自动增强策略, 默认 rand-m9-mstd0.5-inc1, override color_jitter
                re_prob=self.args.reprob,    # 随机擦除概率, 默认 0.25
                re_mode=self.args.remode,    # 随机擦除模式, 默认 pixel
                re_count=self.args.recount,  # 随机擦除次数, 默认 1
                mean=mean,  # 传入的标准化均值
                std=std,    # 传入的标准化标准差
            )
            return transform

        # eval transform
        t = []
        # 将 PIL.Image 转为 torch.Tensor
        t.append(transforms.ToTensor())
        # 标准化（均值/标准差）
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)

    @lru_cache(maxsize=None)
    def _load_data_from_db(self):
        db = sqlite3.connect(self.db_path)
        cur = db.cursor()
        cur.execute(self.sql)
        data = cur.fetchall()
        db.close()
        return data
