# manual_data

## 構成
### ディレクトリ
- manual_kill
- manual_kill_input_file
- manual_notkill_input_file
- manual_notkill
- raw_video_for_manual
- trained_manual_kill
- trained_manual_notkill
- unit_video_for_manual
### ファイル
- generate_tensor.py
- generate_unit_video.py


## generate_unit_video.py
- raw_video_for_manualにある長い動画を4秒にわける
- 名前が衝突しないようにunit_video_for_manualに保存

## manual_notkill, manual_kill
- 手作業でunit_video_for_manualからkill,notkillへ移動させる

## generate_tensor.py
- manual_notkill, manual_killからバッチサイズが20になるように選択して入力テンソル作成して保存
- 保存先はmanual_kill_input_file,manual_notkill_input_file
- テンソルを生成したら4秒動画はtrained_manual_notkill, trained_manual_killへ移動

## tensorについて
- ptファイルは入力データ、教師データのテンソルのタップル
- x, t = torch.load("パス")のようにする