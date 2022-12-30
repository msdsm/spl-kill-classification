"""
各labelのディレクトリからlabelとinputをtensorとして保存
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

def train_preprocess(raw_videos_path):
    print("学習時前処理を開始します。与えられた入力は{}です。".format(raw_videos_path))
    mp4_list = os.listdir(raw_videos_path)
    batch_size = 20
    if len(mp4_list) < batch_size: 
        return False
    frame = 120
    channel = 3
    height = 1080
    width = 1920
    input_data = torch.zeros([batch_size, frame//5, channel, height//6, width//6], dtype=torch.uint8, device='cuda:0')
    input_label = torch.zeros([batch_size, 1], dtype=torch.uint8, device='cuda:0') # 0か1のみ
    jpg_tensor = torch.zeros([3, 1080//6, 1920//6], dtype=torch.uint8, device='cuda:0')
    jpg_tensor_sum = torch.zeros([3, 1080//6, 1920//6], dtype=torch.uint8, device='cuda:0')
    tmp_tensor = torch.zeros([3, 1080, 1920], dtype=torch.uint8, device='cuda:0')
    tmp_list = sorted(mp4_list)
    target = tmp_list[0:20]

    tmp_name, tmp_ext = os.path.splitext(target[0])
    saved_path = os.path.join("./manual_notkill_input_file",tmp_name+".pt")


    print("処理を行う対象は{}です。".format(target))
    for batch_size_idx,file_name in enumerate(target):
        # ファイル、拡張子に分割
        name, ext = os.path.splitext(file_name)

        video_path = os.path.join(raw_videos_path, file_name)

        # mp4以外無視
        if ext != ".mp4":
            continue
        
        # mp4ファイル(save_tmp_path)をjpgに変換
        dst_directory_path = os.path.join(raw_videos_path, name)
        # 作成
        if not os.path.exists(dst_directory_path):
            os.mkdir(dst_directory_path)
        # ffmpegを実行させてjpgに変換 scaleでサイズ指定
        cmd = 'ffmpeg -i \"{}\" -vf scale=1920:1080 \"{}/image_%05d.jpg\"'.format(video_path, dst_directory_path)
        subprocess.call(cmd, shell=True)
        # jpgごとのループ つまりframe
        frame_loop_cnt = 0
        label_flag = False
        for jpg_file_name in sorted(os.listdir(dst_directory_path)):
            frame_loop_cnt += 1
            jpg_file_path = os.path.join(dst_directory_path, jpg_file_name)
            tmp_tensor = read_image(path=jpg_file_path).to('cuda:0')
            # jpg_tensorを初期化
            jpg_tensor.mul_(0)
            # jpg_tensorへ圧縮
            jpg_tensor = F.resize(img=tmp_tensor, size=(1080//6, 1920//6)).clone()
            if frame_loop_cnt%5 == 0:
                # input_dataにセットする
                input_data[batch_size_idx][frame_loop_cnt//5 - 1] = jpg_tensor_sum.clone()
                # jpg_tensor_sum初期化
                jpg_tensor_sum.mul_(0)
            else :
                # 加算
                jpg_tensor_sum.add_(jpg_tensor.div(5, rounding_mode='floor'))
        # jpgファイルごとのループ終わり
        shutil.rmtree(dst_directory_path)
    # ここでmp4_listのloop終わり

    if raw_videos_path == "./manual_kill":
        input_label = torch.ones([batch_size, 1], dtype=torch.uint8, device='cuda:0')
        for file_name in target:
            shutil.move(os.path.join(raw_videos_path, file_name), "trained_manual_kill/") # 使ったmp4移動
    else : 
        input_label = torch.zeros([batch_size, 1], dtype=torch.uint8, device='cuda:0')
        for file_name in target:
            shutil.move(os.path.join(raw_videos_path, file_name), "trained_manual_notkill/") # 使ったmp4移動

    print("テンソルを生成しました。")
    print("入力テンソルのサイズは{}です。".format(input_data.size()))
    print("教師データテンソルのサイズは{}です。".format(input_label.size()))
    print("入力テンソルのデバイスは{}です。".format(input_data.device))
    print("教師データテンソルのデバイスは{}です。".format(input_label.device))
    

    torch.save((input_data, input_label), saved_path)


    return True

def main():
    while True:    
        flag = train_preprocess("./manual_notkill")
        if flag == False:
            break

if __name__ == '__main__':
    main()