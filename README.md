# spl-kill-classification
- preprocess.py


## preprocess.py
- 前処理ファイル
### train_preprocess()
1. raw_videoから存在するmp4ファイルすべて持ってくる
2. 4秒ごとに区切る
3. jpg化してdata/notlabe/"4秒動画名”に保存する
4. 得られたすべての4秒動画(120枚のjpg)に対してTensorつくる(この個数がバッチサイズ)
5. 120フレームは24フレームに圧縮（平均）
6. 1080×1920は180×320に圧縮
7. ラベリング行う
8. ラベルに合わせてnotlabelからkill or not killへ移動
9. mp4ファイルをraw_videoからtrained_raw_videoへ移動
10. 入力テンソルと教師データテンソルを返す
- 入力テンソル : Tensor.Size(N, 24, 3, 180, 320)
- 教師データ : Tensor.Size(N, 1)