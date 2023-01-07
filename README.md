# spl-kill-classification


## 実行順序
1. 長い動画をmanual_data/raw_video_for_manualに保存
2. manual_data下でgenerate_unit_video.pyを実行
    - その結果, manual_data/unit_video_for_manualに保存される
3. manual_data/unit_video_for_manualの4秒動画を見て手動でmanual_data/manual_killまたはmanual_data/manual_notkillへ移動
4. manual_data下でgenerate_tensor.pyを実行
    - その結果, manual_data/manual_kill_input_file, manual_data/manual_notkill_input_fileにテンソルがptファイルとして保存される
5. 最上位ディレクトリ下でtrain.pyを実行
    - 学習結果はtrained_model_fileに保存される(保存先は実行前に変更しておく必要がある)
6. 長い動画をeval_raw_videoに保存
7. evalpreprocess.pyを実行
8. eval.pyを実行
9. evalpostprocess.pyを実行
    - result内に切り抜き動画と入力動画がセットで保存される


## 構造
### ディレクトリ
- eval_raw_video
- manual_data
    - generate_tensor.py
    - generate_unit_video.py
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



