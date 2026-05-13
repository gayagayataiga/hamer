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

**再確認（2026-05-12 夜、`docs/runbooks/parallel.md` の指示書ベース）:** GPU 2,3 で `episode_000204.mp4` (406f) と `GX010052.MP4` (79f) を並列実行。LPT 分配メッセージ表示、両ワーカー rc=0、v2 スキーマ JSON 正常生成（`frames`/`schema_version`/`detector` 全て揃う）。指示書 L66-68 の動作確認ポイントを全てパス。

### 単パス dual-output 化（2026-05-13）

旧 `hamer()` は `_full.mp4` と `_handsonly.mp4` を別々に作るため `process_video` を 2 回呼んでいた。**推論+レンダ結果が同一なのに 2 回回す設計ミス**だった。1 回の推論でメッシュ RGBA を作り、それを「入力画像へ合成 → full」と「単色背景へ合成 → handsonly」の 2 種類に composite すれば 1 パスで両方作れる。

**変更内容:**
- `process_frame` に `dual_output=True` フラグ追加。RGBA メッシュを1回だけ render、`(overlay_bgr, handsonly_bgr, side_bgr, hands_info)` の **4-tuple** を返す
- `process_video` に `handsonly_output_path=None` 引数。両指定なら 2 つの `VideoWriter` を駆動して 1 パスで両動画書き出し（async path / sync path の両方に対応）
- `hamer_api.hamer()` を `process_video` 1 回呼びにリファクタ（`skip_existing` は3出力揃って初めてスキップする all-or-nothing 判定に整理）
- `Hamer.infer_video` に `handsonly_video=` 引数追加。両指定で dual モード、handsonly のみ指定なら旧 `hands_only=True` 経路へフォールバック

**効果:** 推論コスト約半減、レンダ呼び出しも半減。1 動画あたりの実時間は ~2倍速

**動作確認:** 15 frame クリップで dual モードと wrist_only モード両方で正常動作確認

### GX010085/86 並列再走（完了、2026-05-13）

`/home/gayagaya/video/new/` の **2 本（GX010085: 1397f, GX010086: 1630f）**を単パス dual-output 化したコードで再走。
最初は GPU 3 単独で開始 → 途中で GPU 1 が空いたので 2 GPU 並列に切替。
- 旧コード（2 パス）だと推定 4〜5 時間級
- 新コード（1 パス）+ 2 GPU 並列で **wall clock 約 60 分**で完了
  - GPU 1: GX010086 → 約 60 分（0.46 fps）
  - GPU 3: GX010085 → 約 53 分（0.44 fps）

**生成物（全 6 ファイル、frames 数も期待通り）:**

| ファイル | frames |
|---|---|
| `result/GX010085_full.mp4` | 1397 |
| `result/GX010085_handsonly.mp4` | 1397 |
| `result/GX010085_wrist.json` | schema=2, 1397 |
| `result/GX010086_full.mp4` | 1630 |
| `result/GX010086_handsonly.mp4` | 1630 |
| `result/GX010086_wrist.json` | schema=2, 1630 |

**運用メモ（次回以降の参考）:**
- GPU 3 を最初フォルダ全体（85+86）対象で起動してたので、途中で kill + `--files_from` で 1 本に絞って再起動した。フォルダ自動ループだと早く終わった側を後発が上書きしてしまうので、並列時は `--files_from` 推奨
- 走行中ログ: `/tmp/hamer_gpu1.log`（GX010086）/ `/tmp/hamer_gpu3.log`（GX010085）

別セッションから走らせる手順は `docs/runbooks/video_new.md` に集約。

### 動作確認

- v2 スキーマ: 15 フレームのクリップで全フィールドが正しく出ることを確認（`joints_2d[0] == wrist_2d` 一致）
- `Hamer` クラス: GPU 1 で `infer_image(path)` / `infer_image(ndarray, render=True)` / `infer_video(wrist_json)` 全て成功
- `hamer_parallel`: GPU 1,2 の 2 ワーカー並列で 2 動画処理、両 worker rc=0

### 手の重複検出を解消（dedup A+C / 2026-05-13）

GX010086 の手数分布を見たら **2 手のフレームが 15% しかない**（残り 80%+ が 3〜6 手の過検出）ことが判明。`docs/HAND_DEDUP_PLAN.md` に診断と対策候補をまとめた上で **A（L/R 排他選択）+ C（最小サイズフィルタ）**を実装。

**変更（`demo_video.py::detect_hand_bboxes`）:**
- A: ViTPose の L 21pts / R 21pts のうち平均 confidence が高い側のみ採用（1 ROI から 1 bbox に絞る）
- C: bbox 短辺 < 64 px なら捨てる（モニター反射などの偽検出を除外）

**測定結果（dedup+F-1 込み）:**

| 動画 | 修正前 2-hand% | 修正後 2-hand% |
|---|---|---|
| GX010086 | 15.2% | **93.9%** |
| GX010013 | 26.1% | **90.7%** |

### L/R 交差フィルタ F-1 追加（2026-05-13）

エゴセントリック視点では装着者の左手が画像左、右手が画像右にあり、x 座標で交差することは通常起きない。`docs/HAND_LR_CROSSING_PLAN.md` の F-1 を実装：1 フレーム内の `is_right=0` と `is_right=1` のペアで x_center が逆転していたら低 confidence 側を棄却。

- `detect_hand_bboxes(reject_lr_crossing=True)` がデフォルト ON
- 三人称視点や腕組みを想定する場合は False に

GX010013 で測定すると dedup-only 90.2% → dedup+F-1 で **90.7%**。改善幅は小（+0.5pt）だが、副作用がほぼ無く、コミット `e632b99` で投入。

### 整理（cleanup phases、2026-05-13）

`docs/CLEANUP_PLAN.md` に沿って実施：

- **.gitignore に `/result/` 追加** — 単数形 `/result/` が漏れていて常に untracked 状態だったのを修正（commit `fb020c8`）
- **D 不要ファイル削除** — `hamer_evaluation_data.tar.gz`, 空の `out_videos/`, `__pycache__/` を削除
- **A `docs/` 整理** — ルートの `.md` を `docs/` と `docs/runbooks/` に移動。`NEXT.md` だけルート維持（commit `008fd4c`）
- **B `result/<stem>/` サブディレクトリ化** — 旧 `result/GX010001_full.mp4` → `result/GX010001/full.mp4` 等にフラットレイアウトから 1 動画 1 ディレクトリへ。`hamer_api.hamer()` の出力命名も合わせて変更（commit `ca27660`）
- **C wrist JSON 配置一本化** — B に吸収（不要）
- **E `result/` をリポジトリ外に逃がす** — 未着手、ジョブ完了後

下流参照は `/misc/dl00/gayagaya/` で grep して 0 ヒットなので配置自由に決められた。

### 全動画 dedup+F-1 dual-output 再走（実行中、2026-05-13 深夜）

`docs/CLEANUP_PLAN.md` 採用後の最終状態を全動画で揃えるため、`/misc/dl00/gayagaya/video/` の **31 本**を **dedup + F-1 + dual-output モード**で 4 GPU 並列再走。

- 出力: `result/<stem>/{full,handsonly}.mp4` + `result/<stem>/wrist.json`
- コマンド: `python hamer_api.py /misc/dl00/gayagaya/video --output_dir result --no_skip_existing --gpus 0,1,2,3 --log_dir /tmp/hamer_dual_logs`
- 見込み: ~3 時間で完走
- 経緯: 一度 wrist_only モードで 4 並列を組み始めたが、GX010013 の品質チェック後に「mp4 も新コードで揃えたい」となり、wrist_only ジョブを kill して dual-output に切替

### サブモジュール化向け再構築スクリプト（2026-05-13）

このリポジトリを別プロジェクトに submodule で取り込んだ際、`.hamer/` venv と `_DATA/` ペイロードが消えて即座に壊れる。一発復旧するために `scripts/rebuild_env.sh` を実装し、fresh clone から smoke test 通過まで自動で進むようにした。

- ランブック: `docs/SUBMODULE_SETUP.md`
- 設計と検証戦略: `docs/REBUILD_SCRIPT_PLAN.md`
- 実装: `scripts/rebuild_env.sh`（12 ステップ、冪等、`uv venv --seed` 優先、各環境変数で override 可）

`/tmp/hamer_rebuild_test/` に fresh clone して 10 回試行し、smoke test (`OK: pipeline loaded (HAMER on cuda)`) まで通過することを確認。安定化過程で踏み抜いた地雷 6 つ：

1. `uv venv` だけだと pip が入らず anaconda の pip を呼ぶ → `--seed`
2. torch wheel index は **cu124**（既存環境が `torch.version.cuda == 12.4`、cu117 は detectron2 で CUDA mismatch）
3. detectron2 の `setup.py` が `import torch` するので **`--no-build-isolation`**
4. setuptools 81 で `pkg_resources` 廃止 → torch.utils.cpp_extension が壊れるので **setuptools<70**
5. xtcocotools の C 拡張が numpy 1.x ABI 固定 → **numpy<2** を step 4 で pin、かつ `.[all]` install にも inline で同 constraint
6. `fetch_demo_data.sh` は tarball を cwd に置く / `download_models()` は `_DATA/` 直下を見る → 後段で `mv hamer_demo_data.tar.gz _DATA/` しないと 6 GB 再 DL

commit `831a897` で本体・ドキュメント追加。

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

