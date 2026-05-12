# 次回やること

## 直近のセッションでやったこと（2026-05-12 後半）

### JSON 出力スキーマを v2 に拡張（`demo_video.py`）

`_wrist.json` を「手首だけ」から「手の主要情報まるごと」に拡張。`schema_version: 2`。

**ヘッダに追加:**
- `schema_version`, `n_frames_processed`, `detector`
- `camera: {focal_length_px, principal_point_px, model}` — pinhole, +x right / +y down / +z forward, 主点は画像中心
- `units: {...}` — 各フィールドの単位と座標系の説明

**各 hand の per-frame レコードに追加:**
- `bbox` [x1,y1,x2,y2] — 検出器出力（ViTPose の手キーポイント外接矩形、フル画像 px）
- `joints_2d` (21,2) — MANO 21関節をフル画像にピクセル投影。`joints_2d[0] == wrist_2d`
- `joints_3d` (21,3) — カメラ座標 (`cam_t` 加算済み、左手は x-mirror 適用、MANO 単位 ≈ メートル)
- `cam_t` (3,) — 弱透視→透視変換後のカメラ座標系並進
- `mano: {global_orient (1,3,3), hand_pose (15,3,3), betas (10,)}` — 生の HaMeR 予測（**右手正規化フレーム**、左手は x-mirror **未適用**）

実装メモ:
- `process_frame` の中で `out['pred_mano_params']` / `pred_keypoints_3d` / `pred_cam_t_full` をそのままシリアライズ
- `process_video` で `focal_length_px = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(W,H)` を計算してヘッダに渡す
- `bbox` は `detect_hand_bboxes` の出力配列を `bbox_idx` で順送り（dataloader 内のバッチイテレーションと同じ順序）
- 検出器名は `hamer_api.hamer()` → `process_video(detector_name=...)` 経由で伝搬

### `Hamer` クラス API 追加（`hamer_api.py`）

パイプラインを1回ロードしてインスタンスに保持し、単一画像 / 単一動画 / フォルダの3粒度で叩ける。

```python
from hamer_api import Hamer

h = Hamer(body_detector='regnety')

# 単一画像（ndarray BGR か path）
out = h.infer_image('frame.jpg')                       # 推論のみ（高速、wrist_only 相当）
out = h.infer_image(img_bgr, render=True)              # out['overlay'] に BGR ndarray
# out = {'hands': [...], 'width', 'height', 'overlay'?}

# 単一動画
h.infer_video('clip.mp4', wrist_json='clip.json')                       # JSON のみ
h.infer_video('clip.mp4', output_video='out.mp4', wrist_json='out.json') # mp4 + JSON

# フォルダ一括
h.infer_dir('/path/to/videos', output_dir='out', wrist_only=True)
```

既存の `hamer(input_dir, ...)` 関数と CLI (`python hamer_api.py ...`) はそのまま残してある（後方互換）。

### v2 再走（全動画）

旧 32本は schema v1 で手首しか入っていないので、全部 v2 で再推論。

**メインバッチ（実行中）** — `/misc/dl00/gayagaya/video/` の **31本（GX*.MP4）**
- 出力: `result/wrist/<stem>_wrist.json` を上書き
- コマンド: `python hamer_api.py /misc/dl00/gayagaya/video --output_dir result/wrist --wrist_only --no_skip_existing`
- ログ: `/tmp/hamer_v2_run.log`
- 見積もり: 約 2h20m（前回実績ベース）

**後続キュー（メインバッチ終了後に自動起動）** — `episode_000204.mp4`
- 入力: `/home/gayagaya/video/episode_000204.mp4`（メインバッチに含まれない別ディレクトリ）
- 出力: `result/wrist/episode_000204_wrist.json` を上書き
- ログ: `/tmp/hamer_episode_v2.log`
- 起動方法: バックグラウンドで `pgrep` ループしてメインバッチプロセス消失を検知 → `hamer_api.py /home/gayagaya/video --output_dir result/wrist --wrist_only --no_skip_existing` を実行

**メモ:** `/home/gayagaya/hamer` と `/misc/dl00/gayagaya/hamer` は **bind mount で同一物理ファイル**（inode 一致）。`result/wrist/` 内のファイルは1セットしかないので両側を別々に再走する必要はない。

### Git remote / ブランチ運用

`main` は upstream に追従、改造は `personal` ブランチに分離。

```
origin    → git@github.com:gayagayataiga/hamer.git  (fetch & push)   ← fork
upstream  → git@github.com:geopavlakos/hamer.git    (fetch only, push DISABLE) ← 本家
```

- `git config --local user.email ryotsu.kankiti.kotikame@gmail.com` / `user.name gayagayataiga` をローカル設定済み
- `main`: upstream/main と一致（現在 `3a01849`）
- `personal`: 本セッションの改造コミット（現在 `3b81f37 Add Hamer class API and expand wrist JSON to schema v2`）
- upstream の push URL は `DISABLE` で誤 push を防止

**本家更新を取り込む手順:**
```bash
git fetch upstream
git checkout main && git merge --ff-only upstream/main
git push origin main
git checkout personal && git merge main   # or rebase
```

### マルチGPU 並列 `hamer_parallel` 実装

複数 GPU に動画を振り分けて並列推論する関数 + CLI を追加。

**Python API:**
```python
from hamer_api import hamer_parallel
hamer_parallel('/path/to/videos', gpus=[0,1,2,3],
               output_dir='out', wrist_only=True, log_dir='logs/')
```

**CLI:**
```bash
python hamer_api.py /path/to/videos --output_dir out --wrist_only \
    --gpus 0,1,2,3 --log_dir logs/
```

**実装メモ:**
- 各 GPU に `subprocess.Popen` + `CUDA_VISIBLE_DEVICES=<idx>` で独立プロセスを起動（torch を import 前に env を切るので GPU 分離が確実）
- フレーム数を `cv2.VideoCapture` でプローブし、**LPT 貪欲法**（最長動画から大きい順に、その時点で最軽量のバケットに積む）で分配
- ワーカーは新規 CLI フラグ `--files_from <txt>` で自分の担当ファイル一覧を読む
- 既存フラグ（`skip_existing`, `wrist_only`, `async_io`, `body_detector`, `rescale_factor`, `extensions`, `recursive`）は全て伝搬
- `--log_dir` を指定すると `worker_gpu<i>.log` に個別ログ
- LPT の下限は最長動画 1 本の処理時間（GX010003: 2099 frames ≈ 35 分）。GPU を 6 台以上にしても 35 分未満には下がらない

**動作確認:** 9 frame × 2 クリップを GPU 1, 2 に 1 本ずつ並列割当 → 両ワーカー rc=0、v2 スキーマで正常生成

### 動作確認

- v2 スキーマ: 15 フレームのクリップで全フィールドが正しく出ることを確認（`joints_2d[0] == wrist_2d` 一致）
- `Hamer` クラス: GPU 1 で `infer_image(path)` / `infer_image(ndarray, render=True)` / `infer_video(wrist_json)` 全て成功
- `hamer_parallel`: GPU 1,2 の 2 ワーカー並列で 2 動画処理、両 worker rc=0

---

## 直近のセッションでやったこと（2026-05-12）

### `hamer_api.py` を新規追加 — 外部から関数として叩く API

`from hamer_api import hamer` でフォルダ内動画を一括推論できる関数を作成。
内部では `demo_video.py` の `build_pipeline` / `process_video` を再利用し、パイプラインのロードは初回 1 回のみ。

**シグネチャ:**
```python
hamer(input_dir, output_dir=None, extensions=('.mp4',),
      body_detector='regnety', batch_size=8, rescale_factor=2.0,
      async_io=True, recursive=False, skip_existing=True,
      bg_color=(1.0, 1.0, 1.0), pipe=None, files=None,
      wrist_only=False)
```

**出力:**
- `<stem>_full.mp4` — オーバーレイ動画
- `<stem>_handsonly.mp4` — メッシュのみ
- `<stem>_wrist.json` — 手首2Dピクセル座標（フレームごと）

CLI も同居：`python hamer_api.py <input_dir> --output_dir <out>`

### 手首ピクセル座標の JSON 出力（`demo_video.py`）

- `process_frame()` に `return_hands_info=True` / `wrist_only=True` を追加
- `process_video()` に `wrist_json_path=...` / `wrist_only=True` を追加
- ヘルパー `_write_wrist_json()` で `{frame, hands: [{is_right, wrist_2d}]}` 形式で書き出し
- カメラ座標の `wrist_3d` は当初出していたが**削除**（深度は UniDAC 側で行うため）
- 2D 投影は HaMeR の弱透視→透視変換 (`cam_crop_to_full`) と同じパイプライン、レンダリングのメッシュ位置と一致するので信頼できる

### `wrist_only=True` モード（軽量推論）

mp4 を一切書かず、検出+ViTPose+HaMeR の推論結果から手首2D座標のみ JSON で出力。
レンダラを呼ばないため通常モードより高速。

CLI: `python hamer_api.py <input_dir> --wrist_only`

### 動作確認済み

15フレームのテストクリップで以下を検証：
- 通常モード: `_full.mp4` + `_handsonly.mp4` + `_wrist.json` 生成
- `wrist_only=True`: `_wrist.json` のみ、座標値は通常モードと一致
- `skip_existing=True`: 既存出力スキップ

### 保留事項

マルチGPU並列実行 (`hamer_parallel`) は将来課題として下記に記載 → [将来できたらいいこと](#将来できたらいいこと)

---

## episode_000204.mp4 の推論

- 入力: `/home/gayagaya/video/episode_000204.mp4` (406 frames, 16MB)
- 出力先: `result/episode_000204_full.mp4` と `result/episode_000204_handsonly.mp4`
- 検出器: RegNetY（ViTDet と比較済み、差 < 2%）

## 実行コマンド

```bash
cd /misc/dl00/gayagaya/hamer
source .hamer/bin/activate

# full（オーバーレイ）と handsonly を別 GPU で並列
CUDA_VISIBLE_DEVICES=0 python demo_video.py \
  --input /home/gayagaya/video/episode_000204.mp4 \
  --output result/episode_000204_full.mp4 \
  --body_detector regnety &

CUDA_VISIBLE_DEVICES=1 python demo_video.py \
  --input /home/gayagaya/video/episode_000204.mp4 \
  --output result/episode_000204_handsonly.mp4 \
  --body_detector regnety --hands_only &

wait
```

406フレームなので各 ≈ 3〜4 分で終わる見込み。

## 検出器比較動画の再生成

以前 `result/compare_GX010001.mp4` (ViTDet vs RegNetY の横並び比較) を作ったが、cleanup で削除済み。必要なら再生成：

```bash
# RegNetY 版を一時的に生成
CUDA_VISIBLE_DEVICES=0 python demo_video.py \
  --input /misc/dl00/gayagaya/video/GX010001.MP4 \
  --output /tmp/GX010001_regnety.mp4 \
  --body_detector regnety

# 横並び比較動画（左: ViTDet, 右: RegNetY）
ffmpeg -y \
  -i result/GX010001_full.mp4 \
  -i /tmp/GX010001_regnety.mp4 \
  -filter_complex "[0:v]scale=1280:-1,drawtext=text='ViTDet':fontsize=40:fontcolor=white:box=1:boxcolor=black@0.6:x=20:y=20[a];[1:v]scale=1280:-1,drawtext=text='RegNetY':fontsize=40:fontcolor=white:box=1:boxcolor=black@0.6:x=20:y=20[b];[a][b]hstack=inputs=2" \
  -c:v libx264 -preset fast -crf 23 -an result/compare_GX010001.mp4

rm /tmp/GX010001_regnety.mp4
```

参考: 過去の定量比較（110フレーム）
- 平均ピクセル差: 1.30 / 255 (~0.5%)
- 検出ミスマッチレベル (diff > 10): 2/110 フレーム (1.8%)
- 推論時間: RegNetY が ViTDet に対し 40% 短縮

## 将来できたらいいこと

### マルチGPU 並列化 — **実装完了（2026-05-12 後半）**

`hamer_parallel()` として実装済み。詳細は上の節を参照。

短縮見積もり（52動画 / 13,619 frames ベース）:

| GPU 数 | 想定時間 |
|---|---|
| 1 | ~3.8 時間 |
| 2 | ~1.9 時間 |
| 4 | ~57 分 |
| 6 | ~38 分（最長動画 GX010003 ≈ 35 分が下限）|

### 次の候補（未着手）

- **左右ラベルの temporal consistency** — フレーム間で L/R が flip する問題、IoU トラッキングで吸収
- **時系列スムージング** — 1€ filter / Savitzky-Golay
- **逐次 yield / コールバック API** — リアルタイム用途
- **検出フィルタ** — スコア閾値、ROI、最大手数
- **`return_in_memory` モード** — ファイル書かず Python dict 返却
- **MANO の左手 x-mirror 自動適用オプション**
- **デバッグ可視化** — 特定フレームだけ画像出力

