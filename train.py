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
from moviepy.editor import *
import shutil
import os
import subprocess
import cv2
import time

import sys
from myeco import myECO



def main():
    parameters_saved_path = "./trained_model_file/20221204.pth"
    parameters_load_path = "./trained_model_file/20221204.pth"
    # device = torch.device("cuda:0,1,2,3") # GPU指定
    net = myECO()
    # net = net.to(device)
    # net = nn.DataParallel(net, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    net = net.to("cuda")
    net.train()

    ####################超重要####################
    net.load_state_dict(torch.load(parameters_load_path)) # load
    ##############################################

    criterion = nn.BCELoss() # 最終的な出力がシグモイドのとき使用可能
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.005)

    x = torch.load("input_tensor_file/data_20221203.pt")
    t = torch.load("input_tensor_file/label_20221203.pt")
    x = x.to("cuda")
    t = t.to("cuda")
    print(x.size())
    print(t.size())
    print(x.device)
    print(t.device)
    print(x.dtype)
    print(t.dtype)
    batch_size = x.size()[0]
    print(batch_size)

    max_batch_size = 5

    """
    x2 = torch.zeros(max_batch_size, 24, 3, 180, 320)
    x2 = x2.to("cuda")
    t2 = torch.zeros(max_batch_size, 1).to("cuda")
    print(x2.size(), x2.dtype, x2.device)
    print(t2.size(), t2.dtype, t2.device)


    for i in range(max_batch_size):
        x2[i] = x.clone()[i]

    print("batch0の学習開始")
    optimizer.zero_grad()
    tmp = net(x2)
    print("batch0の順伝搬終了")
    loss = criterion(tmp, t2)
    loss.backward()
    optimizer.step()

    # 適合率調べる
    tmp = net(x2)
    loss = criterion(tmp, t2)
    print(loss.item())
    """
    x2 = torch.zeros(max_batch_size, 24, 3, 180, 320)
    x2 = x2.to("cuda")
    t2 = torch.zeros(max_batch_size, 1)
    t2 = t2.to("cuda")
    q = batch_size // max_batch_size
    r = batch_size % max_batch_size
    x3 = torch.zeros(r, 24, 3, 180, 320).to("cuda")
    t3 = torch.zeros(r, 1).to("cuda")
    EPOCH = 20
    print("バッチサイズ{}のテンソルを{}回入力して最後にバッチサイズ{}のテンソルを入力します".format(max_batch_size, q, r))
    print("EPOCHは{}です".format(EPOCH))
    for train_time in range(EPOCH):
        print("{}回目の学習を開始します。".format(train_time+1))
        for train_batch_time in range(q):
            print("{} : {}".format(train_time, train_batch_time))
            for i in range(max_batch_size):
                x2[i] = x.clone()[train_batch_time*max_batch_size+i]
                t2[i] = t.clone()[train_batch_time*max_batch_size+i]
            optimizer.zero_grad()
            outputs = net(x2)
            loss = criterion(outputs, t2)
            loss.backward() # 逆伝搬
            optimizer.step() # パラメータ更新
        
        for i in range(r):
            x3[i] = x.clone()[q*max_batch_size + i]
            t3[i] = t.clone()[q*max_batch_size + i]
        optimizer.zero_grad()
        outputs = net(x3)
        loss = criterion(outputs, t3)
        loss.backward()
        optimizer.step()
        print("{}回目の学習を終了しました。".format(train_time+1))

        # 学習データに対する適合率を調べる
        for train_batch_time in range(q):
            for i in range(max_batch_size):
                x2[i] = x.clone()[train_batch_time*max_batch_size+i]
                t2[i] = t.clone()[train_batch_time*max_batch_size+i]
            outputs = net(x2)
            loss = criterion(outputs, t2) # labelを勝手にone-hot-vectorにしてくれるらしい
            print("EPOCH:{}, train_batch_time:{}, 学習データに対する適合率を調べます。".format(train_time, train_batch_time))
            print("学習データのバッチごとの平均を取ります。バッチサイズは{}です。".format(x2.size()[0]))
            print("損失関数の値 : {}".format(loss.item()))
            right_cnt = sum((outputs>0.5) == t2).item()
            print("正答率　: {}".format(right_cnt / max_batch_size))
            print(right_cnt)
            print((outputs>0.5)==t2)
        for i in range(r):
            x3[i] = x.clone()[q*max_batch_size + i]
            t3[i] = t.clone()[q*max_batch_size + i]
        outputs = net(x3)
        loss = criterion(outputs, t3)
        print("EPOCH:{}, r:{}, 学習データに対する適合率を調べます。".format(train_time, r))
        print("学習データのバッチごとの平均を取ります。バッチサイズは{}です。".format(x3.size()[0]))
        print("損失関数の値 : {}".format(loss.item()))
        right_cnt = sum((outputs>0.5) == t3).item()
        print("正答率　: {}".format(right_cnt / x3.size()[0]))
        print(right_cnt)
        print((outputs>0.5)==t3)


    ################超重要###############
    torch.save(net.state_dict(), parameters_saved_path) # 保存
    #####################################


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print("実行時間 : {}".format(end-start))