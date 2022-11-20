"""
前処理
train_preprocess : 学習時の前処理

test_preprocess : 推論時の前処理
trainと同じく与えられた長いmp4ファイルを区切ってテンソル作るだけ
mp4を4秒ごとに区切って新しく一時的なディレクトリに保存しておく
jpgは保存せずにすべて削除する
返り値として必要な情報 : 入力テンソル、教師データテンソル、バッチサイズごとにその4秒動画のパス

test_postprocess : 推論後の後処理（どうがせいせい)
出力結果のテンソルからlabelが1となっているバッチに対応する4秒動画を持ってくる
全部昇順にくっつけて出力する
一時的に保存している4秒動画すべて削除する

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




def train_preprocess(raw_videos_path): # "raw_video"を与える
    """
    学習前の前処理
    入力:長いmp4を格納したディレクトリのパス
    "./short_raw_video" or "long_raw_video"

    出力:tuple(tensor.size(n,f,c,h,w), tensor.size(n,1))
    """

    print("学習時前処理を開始します。与えられた入力は{}です。".format(raw_videos_path))
    # 1. 入力をうけとりすべてのmp4ファイルを120フレームに分割しjpg変換
    mp4_list = os.listdir(raw_videos_path)
    """ jpg作成スキップする場合ここから"""
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
    data_kill_path = './data/kill'
    data_notkill_path = './data/notkill'
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
    input_label = torch.zeros([batch_size, 1], dtype=torch.uint8, device='cuda:0') # 0か1のみ
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
            label_flag = False
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

                # label特定する
                tmp_flag = True
                tmp_val = tmp_tensor[0][995][705]
                if tmp_val < 100 and tmp_val >=0:
                    for iii in range(20):
                        for jjj in range(20):
                            if not (tmp_tensor[0][995+iii][705+jjj]==tmp_val and tmp_tensor[1][995+iii][705+jjj]==tmp_val and tmp_tensor[2][995+iii][705+jjj]==tmp_val):
                                tmp_flag = False
                    if tmp_flag:
                        label_flag = True
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
            if label_flag:
                input_label[batch_size_idx][0] = 1
                # print(batch_size_idx)
            else :
                input_label[batch_size_idx][0] = 0
        # ４秒動画ごと（バッチ）ごとのループ終わり
    # mp4ごとのループ終わり input_data完成
    # print(input_data.size()) # torch.Size([21, 24, 3, 180, 320])
    # print(input_label.size()) # torch.Size([21, 1])
    # print(torch.count_nonzero(input_data)) # tensor(86958727, device='cuda:0')
    # print(torch.count_nonzero(input_label)) # tensor(2, device='cuda:0')

    # notlabelからkill or notkill へ移動 ディレクトリの名前が衝突しないようにmp4name+4秒動画ディレクトリ名とする
    batch_size_idx = -1
    for mp4_list_name in notlabel_video_list:
        mp4_list_name_path = os.path.join(data_notlabel_path, mp4_list_name)
        # 4秒動画ごとのループ
        for unit_video_name in sorted(os.listdir(mp4_list_name_path)):
            batch_size_idx += 1
            if input_label[batch_size_idx][0] == 1:
                # killへ移動
                # ./data/notlabel/"mp4name"/"4秒動画名"/ -> ./data/kill/"mp4name"_"4秒動画"
                unit_video_path = os.path.join(mp4_list_name_path, unit_video_name)
                new_path = os.path.join(data_kill_path, mp4_list_name+'_'+unit_video_name)
                shutil.move(unit_video_path, new_path)
            else :
                # notkillへ移動
                # ./data/notlabel/"mp4name"/"4秒動画名"/ -> ./data/notkill/"mp4name"_"4秒動画"
                unit_video_path = os.path.join(mp4_list_name_path, unit_video_name)
                new_path = os.path.join(data_notkill_path, mp4_list_name+'_'+unit_video_name)
                shutil.move(unit_video_path, new_path)
    
    # notlabel以下のディレクトリ(mp4ごとに生成したディレクトリ)削除
    for mp4_list_name in notlabel_video_list:
        mp4_list_name_path = os.path.join(data_notlabel_path, mp4_list_name)
        os.rmdir(mp4_list_name_path)

    # mp4ファイル移動
    for mp4_file in os.listdir(raw_videos_path):
        mp4_path = os.path.join(raw_videos_path, mp4_file)
        shutil.move(mp4_path, "./trained_raw_video")

    print("テンソルを生成しました。")
    print("入力テンソルのサイズは{}です。".format(input_data.size()))
    print("教師データテンソルのサイズは{}です。".format(input_label.size()))
    ret = input_data, input_label

    # print(len(ret)) # 2
    # print(ret[0].size()) # torch.Size([21, 24, 3, 180, 320])
    # print(ret[1].size()) # torch.Size([21, 1])
    return ret




    

def test_preprocess():
    """
    推論前の前処理
    """



def main():
    path = "./short_raw_video"
    x, t = train_preprocess(path)
    print(x.size())
    print(t.size())
    """
    dataset = train_preprocess(path)
    print(dataset.size())
    """


if __name__ == "__main__":
    main()