"""
推論前処理
preprocess.pyのeval_preprocessを実行する
得られたxを保存する
mp4をresultへ移動する
xの名前をmp4と一致させる
result下にmp4の名前のディレクトリを生成する
そのディレクトリ以下にmp4, ptファイルを保存する

推論したら出力されたラベルをlabel.ptという名前で保存

推論後はループを回す際にlabel.ptがあるかどうかで判別
label.ptがあるなら推論をすでにしているということなので、切り抜き生成する
label.ptをloadしてmp4を切り抜いて保存する
以上の処理はローカルのCPUで可能？
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



def eval_preprocess(raw_videos_path): # "eval_raw_video"を与える
    """
    推論前の前処理
    """

    print("推論前処理を開始します。与えられた入力は{}です。".format(raw_videos_path))
    # 1. 入力をうけとりすべてのmp4ファイルを120フレームに分割しjpg変換
    mp4_list = os.listdir(raw_videos_path)
    # ファイルごとのループ
    for file_name in mp4_list:
        # ファイル、拡張子に分割
        name, ext = os.path.splitext(file_name)

        # mp4以外無視
        if ext != ".mp4":
            continue
        
        # raw_videoの切り抜かれた複数の4秒動画をjpg化したディレクトリらの保存先
        if not os.path.exists(os.path.join("./data/notlabel", name)):
            os.mkdir(os.path.join("./data/notlabel", name))
        
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
        tmp_video_path = "tmp_cut_video" # 切り抜いた動画の保存先
        if not os.path.exists(tmp_video_path):
            os.mkdir(tmp_video_path)
        
        # 切り抜く
        for index, start_time in enumerate(cut_time):
            end_time = start_time + unit_frames//fps
            print(start_time, end_time)
            unit_video = VideoFileClip(videopath).subclip(start_time, end_time) # 4秒動画完成
            tmp_cut_video_path = str(index).zfill(10)+'.mp4'
            save_tmp_path = os.path.join(tmp_video_path, tmp_cut_video_path)
            unit_video.write_videofile(
                save_tmp_path,
                codec='libx264', 
                audio_codec='aac', 
                temp_audiofile='temp-audio.m4a', 
                remove_temp=True
            ) # 保存

            # ここからmp4ファイル(save_tmp_pathにあるやつ)をjpgに変換する
            # ファイル、拡張子に分割
            tmp_cut_video_name, tmp_cut_video_ext = os.path.splitext(tmp_cut_video_path)
            dst_directory_path = os.path.join(os.path.join("./data/notlabel", name), tmp_cut_video_name)
            # なければ作成
            if not os.path.exists(dst_directory_path):
                os.mkdir(dst_directory_path)
             # ffmpegを実行させてjpgに変換 scaleでサイズ指定
            cmd = 'ffmpeg -i \"{}\" -vf scale=1920:1080 \"{}/image_%05d.jpg\"'.format(
                save_tmp_path, dst_directory_path)
            subprocess.call(cmd, shell=True)
            # jpgに変換後,4秒動画もういらないので削除
            os.remove(save_tmp_path)
        os.rmdir(tmp_video_path) # 空でなければ削除されない 4秒動画保存先削除
    # ここでmp4_listのloop終わり
    # data/notlabel/mp4ファイルの名前/　にjpg作成完了
    """jpg作成スキップ終わり"""
    print("4秒ごとに切り抜いてjpgに変換しました。")
    
    
    data_path = './data'
    data_notlabel_path = './data/notlabel'
    notlabel_video_list = sorted(os.listdir(data_notlabel_path))
    # print(notlabel_video_list) # mp4のファイル名
    batch_size = 0
    frame = 120
    channel = 3
    height = 1080
    width = 1920
    for mp4_list_name in notlabel_video_list:
        mp4_list_name_path = os.path.join(data_notlabel_path, mp4_list_name)
        batch_size += len(os.listdir(mp4_list_name_path))
    # print(batch_size)
    input_data = torch.zeros([batch_size, frame//5, channel, height//6, width//6], dtype=torch.uint8, device='cuda:0')
    # print(input_data.size())
    # print(input_data.device)
    batch_size_idx = -1
    jpg_tensor = torch.zeros([3, 1080//6, 1920//6], dtype=torch.uint8, device='cuda:0')
    jpg_tensor_sum = torch.zeros([3, 1080//6, 1920//6], dtype=torch.uint8, device='cuda:0')
    tmp_tensor = torch.zeros([3, 1080, 1920], dtype=torch.uint8, device='cuda:0')
    # print(jpg_tensor.size(), jpg_tensor.device)
    # print(jpg_tensor_sum.size(), jpg_tensor_sum.device)
    # print(tmp_tensor.size(), tmp_tensor.device)
    for mp4_list_name in notlabel_video_list:
        # print(mp4_list_name)
        mp4_list_name_path = os.path.join(data_notlabel_path, mp4_list_name)
        # 4秒動画ごとのループ
        for unit_video_name in sorted(os.listdir(mp4_list_name_path)):   
            batch_size_idx += 1
            # 4秒動画をjpgに変換したディレクトリ内
            # print(unit_video_name)
            unit_video_path = os.path.join(mp4_list_name_path, unit_video_name)
            tmp_jpg_list = sorted(os.listdir(unit_video_path))
            # print(tmp_jpg_list)

            # jpgごとのループ つまりframe
            frame_loop_cnt = 0
            for jpg_file_name in tmp_jpg_list:
                frame_loop_cnt += 1
                jpg_file_path = os.path.join(unit_video_path, jpg_file_name)
                # print(jpg_file_path)
                # sta = time.perf_counter()
                tmp_tensor = read_image(path=jpg_file_path).to('cuda:0')
                # end = time.perf_counter()
                # print("img -> tensor : {}".format(end-sta))
                # print(tmp_tensor.size()) # 3*1080*1920
                # print(tmp_tensor.device)

                # jpg_tensorを初期化
                # sta = time.perf_counter()
                jpg_tensor.mul_(0)
                # end = time.perf_counter()
                # print("chw loop {}".format(end-sta))

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
        # ４秒動画ごと（バッチ）ごとのループ終わり
    # mp4ごとのループ終わり input_data完成

    # notlabelからjpgすべて削除
    shutil.rmtree(data_notlabel_path)
    os.mkdir(data_notlabel_path)

    print("テンソルを生成しました。")
    print("入力テンソルのサイズは{}です。".format(input_data.size()))
    print("入力テンソルのデバイスは{}です。".format(input_data.device))

    # mp4ファイル移動
    for mp4_file in os.listdir(raw_videos_path): # 中身は絶対に１つ
        mp4_path = os.path.join(raw_videos_path, mp4_file)
        name, ext = os.path.splitext(mp4_file)
        result_path = os.path.join("./result", name)
        os.mkdir(result_path)
        shutil.move(mp4_path, result_path)
        tensor_path = os.path.join(result_path, "input.pt")
        torch.save(input_data, tensor_path)
    return input_data


def main():
    # 動画のパスを指定する
    videos_path = "eval_raw_video"

    x = eval_preprocess(videos_path)
    print("推論前処理が終わりました。")
    print(x.size())
    print(x.device)
    print(x.dtype)

if __name__ == '__main__':
    main()