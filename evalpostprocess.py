"""
切り抜き生成ファイル
推論後処理
resultの中からlabel.ptは存在するがhighlight.mp4とempty.txtは存在しないディレクトリを探す
あったらディレクトリ名を出力する
label.ptをロードする
label.ptとmp4ファイルから切り抜き生成してhighlight.mp4という名前で保存
多分CPUでok
そのためにtensor -> numpy
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

def eval_postprocess(result_path):
    ret = []
    for name in sorted(os.listdir(result_path)):
        directory_path = os.path.join(result_path, name)
        label_flag = False
        highlight_flag = False
        mp4_flag = False
        empty_flag = False
        for file_name in sorted(os.listdir(directory_path)):
            if file_name == "highlight.mp4":
                highlight_flag = True
            if file_name == "label.pt":
                label_flag = True
            if file_name == name+'.mp4':
                mp4_flag = True
            if file_name == "empty.txt":
                empty_flag = True
        # print(highlight_flag, label_flag)

        if not highlight_flag and label_flag and mp4_flag and not empty_flag:
            print(name)
            label_path = os.path.join(directory_path, "label.pt")
            highlight_path = os.path.join(directory_path, "highlight.mp4") # 保存先
            video_path = os.path.join(directory_path, name+".mp4")
            t = torch.load(label_path)
            np_t = t.to("cpu").detach().numpy().copy()
            print(np_t)
            cut_time = []
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            time = 0
            unit_frames = 120
            for idx, tmp_num in enumerate(np_t):
                if tmp_num[0] > 0.5: # 出力が(0,1)であることを想定　{0,1}の２値しか取らない場合でもok
                    cut_time.append(idx*(unit_frames//fps))
            tmp_video_path = os.path.join(directory_path,"tmp_cut_video") # 切り抜いた動画の保存先
            if not os.path.exists(tmp_video_path):
                os.mkdir(tmp_video_path)
            # 切り抜く
            highlight_list = []
            for index, start_time in enumerate(cut_time):
                end_time = start_time + unit_frames//fps
                print(start_time, end_time)
                unit_video = VideoFileClip(video_path).subclip(start_time, end_time) # 4秒動画完成
                tmp_cut_video_path = str(index).zfill(10)+'.mp4'
                save_tmp_path = os.path.join(tmp_video_path, tmp_cut_video_path)
                unit_video.write_videofile(
                    save_tmp_path,
                    codec='libx264', 
                    audio_codec='aac', 
                    temp_audiofile='temp-audio.m4a', 
                    remove_temp=True
                ) # 保存
                highlight_list.append(VideoFileClip(save_tmp_path))
            # 切り抜き終わり
            if len(highlight_list) != 0: # 空でないなら(切り抜く場所があるなら)
                # highlight生成
                highlight = concatenate_videoclips(highlight_list)
                highlight.write_videofile(
                    highlight_path,
                    codec='libx264', 
                    audio_codec='aac', 
                    temp_audiofile='temp-audio.m4a', 
                    remove_temp=True
                ) # 切り抜き動画保存
            else : # 空の場合は切り抜く箇所がないということを示す空のファイルを生成する
                empty_file_path = os.path.join(directory_path, "empty.txt")
                f = open(empty_file_path, 'w')
                f.write('')
                f.close()

            shutil.rmtree(tmp_video_path) # きりぬいた4秒動画まとめて削除
        # if文終わり
    # resultのloop終わり
    return ret

def main():
    result_path = "./result"
    generated_list = eval_postprocess(result_path)
    print(generated_list)

if __name__ == "__main__":
    main()