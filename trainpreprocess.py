"""
学習前々処理
preprocess.pyのtrainpreprocessを実行する
得られたx,tを保存
"""

import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import *
import shutil
import os
import subprocess
import cv2
import time
import torchvision.transforms.functional as F
from torchvision.io import read_image

from preprocess import train_preprocess

def main():
    videos_path = "short_raw_video"
    x, t = train_preprocess(videos_path)
    x = torch.div(x.to(dtype=torch.float32), 255)
    t = t.to(dtype=torch.float32)
    print("前処理が終わりました。")
    print(x.size())
    print(t.size())
    print(x.device)
    print(t.device)
    print(x.dtype)
    print(t.dtype)
    torch.save(x, "pt_file/train_preprocess_input_data.pt")
    torch.save(t, "pt_file/train_preprocess_input_label.pt")

if __name__ == '__main__':
    main()