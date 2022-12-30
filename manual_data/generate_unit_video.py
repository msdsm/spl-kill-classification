"""
前処理
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
#from .utils import *




def generate_unit_video(raw_videos_path): # "raw_video"を与える
    """
    4秒動画自動生成
    入力 : long videoを入れたディレクトリのパス
    出力 : 4秒動画に分割したmp4の名前のlist
    処理内容 : mp4をすべて取得して、4秒動画に分割
    同ディレクトリ内に名前が衝突しないように生成
    もとの長い動画を把握しておく必要はなし
    """
    ret = []
    print("4秒動画自動生成を開始します。与えられた入力は{}です。".format(raw_videos_path))
    # 1. 入力をうけとりすべてのmp4ファイルを120フレームに分割しjpg変換
    mp4_list = os.listdir(raw_videos_path)
    for file_name in mp4_list:
        # ファイル、拡張子に分割
        name, ext = os.path.splitext(file_name)
        ret.append(name)
        # mp4以外無視
        if ext != ".mp4":
            continue
        
        videopath = os.path.join(raw_videos_path, file_name)
        # print(videopath)
        cap = cv2.VideoCapture(videopath)
        #print(f"width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}") # width: 1920.0
        #print(f"height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}") # height: 1080.0
        #print(f"fps: {cap.get(cv2.CAP_PROP_FPS)}") # fps: 30
        #print(f"frame_count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}") # フレーム数 frame_count: 1800.0 1分の場合
        #print(f"length: {cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)} s")　#再生時間 length: 21.3 s
        #print("fourcc: " + int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("utf-8"))　#　コーデックの情報 fourcc: avc1

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # print(fps, frame_count)
        time = 0
        unit_frames = 120
        cut_time = []
        while time*fps + unit_frames < frame_count:
            start = time
            cut_time.append(start)
            time += unit_frames//fps
        # print(cut_time)
        
        # 切り抜く
        for index, start_time in enumerate(cut_time):
            end_time = start_time + unit_frames//fps
            print(start_time, end_time)
            unit_video = VideoFileClip(videopath).subclip(start_time, end_time) # 4秒動画完成
            tmp_cut_video_path = name+str(index).zfill(10)+'.mp4'
            save_tmp_path = os.path.join("./", tmp_cut_video_path)
            unit_video.write_videofile(
                save_tmp_path,
                codec='libx264', 
                audio_codec='aac', 
                temp_audiofile='temp-audio.m4a', 
                remove_temp=True
            ) # 保存
    # ここでmp4_listのloop終わり
    return ret


def main():
    videos_path = "raw_video_for_manual"
    ret = generate_unit_video(videos_path)
    print(ret)

if __name__ == '__main__':
    main()
