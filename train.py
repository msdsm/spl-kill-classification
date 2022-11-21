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
import torch.nn as nn
import torch.optim as optim # 最適化手法を使える
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
import sys
from myeco import myECO



def main():
    parameters_saved_path = "./pth_file/20221121.pth"
    # parameters_load_path = "./pth_file/20221120.pth"
    device = torch.device("cuda:0") # GPU指定
    net = myECO()
    net = net.to(device)

    ####################超重要####################
    # net.load_state_dict(torch.load(parameters_load_path)) # load
    ##############################################

    criterion = nn.BCELoss() # 最終的な出力がシグモイドのとき使用可能
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.005)


    videos_path = "short_raw_video"
    x, t = train_preprocess(videos_path)
    x = torch.div(x.to(dtype=torch.float32), 255)
    t = t.to(dtype=torch.float32)
    print("学習前処理が終わりました。")
    print(x.size())
    print(t.size())
    print(x.device)
    print(t.device)
    print(x.dtype)
    print(t.dtype)
    batch_size = x.size()[0]

    EPOCH = 10
    for train_time in range(EPOCH):
        print("{}回目の学習を開始します。".format(train_time+1))
        train_start_time = time.perf_counter()
        optimizer.zero_grad()
        outputs = net(x)
        loss = criterion(outputs, t)
        loss.backward() # 逆伝搬
        optimizer.step() # パラメータ更新
        print("{}回目の学習を終了しました。".format(train_time+1))
        train_end_time = time.perf.counter()
        print("経過時間 : {}".format(train_end_time-train_start_time))

    # 学習データに対する適合率を調べる
    outputs = net(x)
    loss = criterion(outputs, t) # labelを勝手にone-hot-vectorにしてくれるらしい
    print("学習データに対する適合率を調べます。")
    print("学習データのバッチごとの平均を取ります。バッチサイズは{}です。".format(x.size()[0]))
    print("損失関数の値 : {}".format(loss.item()))
    right_cnt = sum((outputs>0.5) == t).item()
    print("正答率　: {}".format(right_cnt / batch_szie))
    print(right_cnt)
    print((outputs>0.5)==t)

    ################超重要###############
    torch.save(net.state_dict(), parameters_saved_path) # 保存
    #####################################


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print("実行時間 : {}".format(end-start))