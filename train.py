"""
学習ファイル

学習前処理で生成したテンソルを使用して学習を行う
学習後にモデルのパラメータ保存
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
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
    device = "cuda:0"
    parameters_saved_path = "./trained_model_file/20220115.pth"
    parameters_load_path = "./trained_model_file/20221231.pth"
    net = myECO()
    net = net.to(device)
    net.train()

    ####################超重要####################
    net.load_state_dict(torch.load(parameters_load_path)) # load
    ##############################################

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.005)


    data_list=[]

    data_path = "./manual_data/manual_notkill_input_file"
    for filename in os.listdir(data_path):
        data_list.append(os.path.join(data_path, filename))

    data_path = "./manual_data/manual_kill_input_file"
    for filename in os.listdir(data_path):
        data_list.append(os.path.join(data_path, filename))

    x_list = []
    t_list = []
    for data in data_list:
        tensor_file_path = data
        print(tensor_file_path)
        x, t = torch.load(tensor_file_path)
        x = x.to(device)
        t = t.to(device)
        x_list.append(x)
        t_list.append(t)

    input_x = torch.cat(x_list, axis=0)
    input_t = torch.cat(t_list, axis=0)
    input_x = input_x.to(device)
    input_t = input_t.to(device)

    input_x = input_x.to(torch.float32)

    print(input_x.size(), input_t.size()) # torch.Size([360, 24, 3, 180, 320]) torch.Size([360, 1])

    input_t = torch.reshape(input_t, (input_t.size()[0],)) # torch.Size([360]) こっち使う

    print(input_t.size())

    train = TensorDataset(input_x, input_t)
    	
    train_loader = DataLoader(train, batch_size=10, shuffle=True)

    

    for epoch in range(500): #学習回数500回
        total_loss = 0
        cnt = 0
        for train_x, train_y in train_loader:
            cnt+=1
            print(epoch, cnt)
            print(train_x.size(), train_x.device, train_x.dtype)
            optimizer.zero_grad()
            output = net(train_x)
            loss = criterion(output, train_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() #loss.data[0]で記述するとPyTorch0.5以上ではエラーが返る
        
        if (epoch+1)%60 == 0:
            print(epoch+1, total_loss)

    ################超重要###############
    torch.save(net.state_dict(), parameters_saved_path) # 保存
    #####################################


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print("実行時間 : {}".format(end-start))