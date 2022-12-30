"""
推論ファイル
resultの中からlabel.ptが存在しないディレクトリを探索
見つけたらディレクトリを出力する
input.ptをロードして順伝搬
label.ptをそのディレクトリ内に保存する


eval2はevalと何が違う？
忘れたけどこれでうまく行く
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
    parameters_load_path = "./trained_model_file/20221230-2.pth"
    net = myECO()
    net = net.to("cuda")
    net.train()
    ####################超重要####################
    net.load_state_dict(torch.load(parameters_load_path)) # load
    ##############################################

    criterion = nn.BCELoss() # 最終的な出力がシグモイドのとき使用可能
    max_batch_size = 5
    x2 = torch.zeros(max_batch_size, 24, 3, 180, 320).to("cuda")
    t = torch.zeros(0, 1).to("cuda")
    # result内loop
    result_path = "./result"
    for name in sorted(os.listdir(result_path)):
        directory_path = os.path.join(result_path, name)
        label_flag = False
        input_flag = False
        for file_name in sorted(os.listdir(directory_path)):
            if file_name == "input.pt":
                input_flag = True
            if file_name == "label.pt":
                label_flag = True
            
        if(not label_flag and input_flag): # label.ptはないけどinput.ptはある
            # 推論を行ってlabel.ptを生成する
            input_data_path = os.path.join(directory_path, "input.pt")
            x = torch.load(input_data_path)
            print(x.size())
            batch_size = x.size()[0]
            q = batch_size // max_batch_size
            r = batch_size % max_batch_size
            for eval_batch_time in range(q):
                print("name : {}, eval_batch_time : {}".format(name, eval_batch_time))
                for i in range(max_batch_size):
                    x2[i] = x.clone()[eval_batch_time*max_batch_size+i]    
                outputs = net(x2)          
                t = torch.cat([t, outputs], axis=0)
            print("name : {}, r : {}".format(name, r))

            x2 = torch.zeros(r, 24, 3, 180, 320).to("cuda")
            for i in range(r):
                x2[i] = x.clone()[q*max_batch_size + i]
            outputs = net(x2)
            t = torch.cat([t, outputs], axis=0)
        
            print(t.size())
            print(t)
            saved_label_path = os.path.join(directory_path, "label.pt")
            torch.save(t, saved_label_path)



if __name__ == "__main__":
    main()