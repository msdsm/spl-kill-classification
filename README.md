# spl-kill-classification

## 注意事項
### trainpreprocessいらない


## 暫定の最終バージョン
- 以下を追加
- 出力層を2にして誤差関数設定
- pytorchのdataloaderを使用
- シャッフルを行って均等に学習可能

## 実行順序
1. 長い動画をmanual_data/raw_video_for_manualに保存
2. manual_data下でgenerate_unit_video.pyを実行


## デバイス
- kawauso


## 構造
### ディレクトリ
- eval_raw_video
- manual_data
- result
- trained_model_file
- trained_raw_video
### ファイル
- eco.py
- eval.py
- evalpostprocess.py
- evalpreprocess.py
- myeco.py
- train.py
- trainpreprocess.py

## eco.py
- networkの基本的な層


## myeco.py
- eco.pyにある層を結合させてnetwork定義

## train.py
- 学習

## evalpreprocess.py
- 推論前処理

## eval.py
- 推論

## evalpostprocess.py
- 推論後処理
1. resultの中からhiglight.mp4とempty.txtはないがlabel.ptはあるディレクトリを探索して出力
2. labelをもとに切り抜き生成



