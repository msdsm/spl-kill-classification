"""
spla_kill_summarization
train file

train_preprocessからデータ得る
netに流す
教師データと正解データの差分から逆伝搬
最適化
パラメータ変更

学習終了後にパラメータ保存

再学習する場合は最初に保存したぱらめーたを持ってくるようにして、
最後に保存するぱらめーたファイルを別名にする

何回目に学習したぱらめーたふぁいるなのかわかるように
ぱらめーたファイルは日付などにする

多くなることを想定してぱらめーたファイルを格納したディレクトリを用意しておく
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
    print(os.listdir("./data/kill"))
    print(os.listdir("./data/notkill"))
    print(os.listdir("./data/notlabel"))
    print(os.listdir("./short_raw_video"))
    print(os.listdir("./trained_raw_video"))
    raw_videos_path = "./short_raw_video"
    x, t = train_preprocess(raw_videos_path)
    print(x.size())
    print(t.size())
    print(torch.count_nonzero(x))
    print(torch.count_nonzero(t))
    print(os.listdir("./data/kill"))
    print(os.listdir("./data/notkill"))
    print(os.listdir("./data/notlabel"))
    print(os.listdir("./short_raw_video"))
    print(os.listdir("./trained_raw_video"))

if __name__ == '__main__':
    main()