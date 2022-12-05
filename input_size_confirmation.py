"""
input_tensor_fileにあるテンソルのサイズをリストで表示する
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

def main():
    data_path = "input_tensor_file"
    tensor_list = sorted(os.listdir(data_path))
    for data in tensor_list:
        tensor_path = os.path.join(data_path, data)
        tensor = torch.load(tensor_path)
        print("・{}".format(tensor_path))
        print("size : {}\ndtype : {}\ndevice : {}\n".format(tensor.size(), tensor.dtype, tensor.device))

if __name__ == "__main__":
    main()