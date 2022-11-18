"""
jpg -> numpy 
numpy -> tensor
tensor.Size(N,f,c,h,w) # 画像ディレクトリの数,そのディレクトリ内の枚数,3,1080,1280
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import *
import os
import subprocess
import cv2
#from .utils import *


def train_preprocess(raw_videos_path): # "raw_video"を与える
    """
    学習前の前処理

    入力:raw_video(mp4)を複数格納したディレクトリの名前(str) (./raw_video)
    1. すべてのmp4ファイルを４秒(120フレーム)ごとに区切って、jpgに変換する　jpgをまとめたディレクトリ名はmp4のやつ再利用

    2. 処理したmp4ファイルをすべてraw_videoからtrained_raw_videoへ移す
    3. label付
    4. data内に保存したディレクトリらをラベルに合わせてkill/notkillへうつす
    5. video_listからTensorに変換
    6. 使用した画像はすべてdataから削除（容量を気にして保存しない）
    出力: tensor?
    """
    # 1. 入力をうけとりすべてのmp4ファイルを120フレームに分割しjpg変換
    mp4_list = os.listdir(raw_videos_path)
    # print(mp4_list) # ['dummy.mp4', 'dummy2.mp4']
    # ファイルごとのループ
    for file_name in mp4_list:
        # ファイル、拡張子に分割
        name, ext = os.path.splitext(file_name)

        # mp4以外無視
        if ext != ".mp4":
            continue
        
        # raw_videoの切り抜かれた複数の4秒動画をjpg化したディレクトリらの保存先
        if not os.path.exists(os.path.join("./dummy_data/dummy_notlabel", name)):
            os.mkdir(os.path.join("./dummy_data/dummy_notlabel", name))
        
        videopath = os.path.join(raw_videos_path, file_name)
        # print(videopath)
        cap = cv2.VideoCapture(videopath)
        """
        print(f"width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}") # width: 1920.0
        print(f"height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}") # height: 1080.0
        print(f"fps: {cap.get(cv2.CAP_PROP_FPS)}") # fps: 30
        print(f"frame_count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}") # フレーム数 frame_count: 1800.0 1分の場合
        print(f"length: {cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)} s")　#再生時間 length: 21.3 s
        print("fourcc: " + int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("utf-8"))　#　コーデックの情報 fourcc: avc1
        """
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
            dst_directory_path = os.path.join(os.path.join("./dummy_data/dummy_notlabel", name), tmp_cut_video_name)
            # なければ作成
            if not os.path.exists(dst_directory_path):
                os.mkdir(dst_directory_path)
             # ffmpegを実行させてjpgに変換 scaleでサイズ指定
            cmd = 'ffmpeg -i \"{}\" -vf scale=1920:1080 \"{}/image_%05d.jpg\"'.format(
                save_tmp_path, dst_directory_path)
            subprocess.call(cmd, shell=True)
            # jpgに変換後,4秒動画もういらない
            os.remove(save_tmp_path)
        os.rmdir(tmp_video_path) # 空でなければ削除されない 4秒動画保存先削除
    
    # ここでmp4_listのloop終わり
    # data/notlabel/mp4ファイルの名前/　にjpg作成完了

    """
    このあとかくこと
    ラベル特定
    ラベルに合わせてディレクトリをnotlabelからkill or notkillへ移動する これ必要なくないか
    120枚のjpgをまとめた複数のディレクトリ(Nこ)からtensor.Size(N,120,3,1080,1280)作成 ラベルも対応させる
    jpgすべて削除
    notlabel/　にある複数のディレクトリすべて削除
    データセットを返り値とする
    必要な情報はテンソル、教師データ
    あとはなんだろうか
    入力として与えた教師データはraw_videoと対応させて保存するべきではなかろうか
    """





    

def test_preprocess():
    """
    推論前の前処理
    """



def main():
    path = "./dummy_raw_video"
    train_preprocess(path)
    """
    dataset = train_preprocess(path)
    print(dataset.size())
    """


if __name__ == "__main__":
    main()