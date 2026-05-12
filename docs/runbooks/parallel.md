# マルチGPU 並列実行 指示書

`hamer_parallel`（`personal` ブランチ）で複数 GPU に動画を振り分けて並列推論する手順。

## 前提

- リポジトリ: `git@github.com:gayagayataiga/hamer.git`
- ブランチ: **`personal`**（`hamer_parallel` はここに入っている）
- 走らせるサーバーに `.hamer/` venv が同じ構成で組まれている、または再構築可能

## セットアップ

```bash
git clone git@github.com:gayagayataiga/hamer.git
cd hamer
git checkout personal
# venv セットアップ（既存サーバーなら不要）
# python -m venv .hamer
# source .hamer/bin/activate
# pip install -r requirements.txt  等
source .hamer/bin/activate
```

## 実行コマンド

### 4 GPU 並列でフォルダ一括（JSON のみ、上書き）

```bash
python hamer_api.py /path/to/videos \
    --output_dir /path/to/out \
    --wrist_only \
    --no_skip_existing \
    --gpus 0,1,2,3 \
    --log_dir logs/
```

### 主要オプション

| フラグ | 説明 |
|---|---|
| `--gpus 0,1,2,3` | 使う GPU index（カンマ区切り）。これを指定すると `hamer_parallel` 経路に入る |
| `--log_dir logs/` | 各ワーカーの stdout/stderr が `logs/worker_gpu<i>.log` に分かれる（無いと親プロセスに混ざる） |
| `--wrist_only` | JSON のみ（mp4 を書かない、高速） |
| `--no_skip_existing` | 既存出力を上書き |
| `--body_detector regnety` / `vitdet` | 検出器（既定 `regnety`） |
| `--rescale_factor 2.0` | 既定値で OK |
| `--batch_size 8` | OOM が出たら 4 に下げる |
| `--recursive` | サブディレクトリも対象 |
| `--extensions .mp4 .MP4 .mov` | 対象拡張子 |

## ベンチマーク（タイミング計測）

開始前と完了後にタイムスタンプ：

```bash
date +%s > /tmp/hamer_t0
python hamer_api.py /path/to/videos --output_dir out --wrist_only --no_skip_existing \
    --gpus 0,1,2,3 --log_dir logs/
echo "elapsed = $(($(date +%s) - $(cat /tmp/hamer_t0))) sec"
```

シリアル比較するなら同じデータで `--gpus 0` のみ指定（worker 1個）。

## 動作確認ポイント

- 起動時に `[hamer-parallel] N videos, M frames, split across K GPUs:` と LPT 分配が表示される
- 各ワーカーは `[hamer-parallel] spawned GPU X pid=...` で確認、`exited rc=0` で正常終了
- 各 `logs/worker_gpu<i>.log` の末尾に `[hamer] done. processed N videos`

## 内部仕様メモ

- 各 GPU に `subprocess.Popen` + `CUDA_VISIBLE_DEVICES=<idx>` で独立プロセス起動（torch を import 前に env を切るので GPU 分離が確実）
- フレーム数を `cv2.VideoCapture` でプローブし、**LPT 貪欲法**（最長動画から大きい順に、その時点で最軽量のバケットに積む）で動画を分配
- ワーカーは CLI フラグ `--files_from <txt>` で自分の担当ファイルだけ処理
- 既存出力スキップ（`skip_existing`）、`wrist_only`、`async_io` などのフラグは全て伝搬

## 短縮見積もり（参考）

52動画 / 13,619 frames ベース：

| GPU 数 | 想定時間 |
|---|---|
| 1 | ~3.8 時間 |
| 2 | ~1.9 時間 |
| 4 | ~57 分 |
| 6 | ~38 分（最長動画 GX010003 ≈ 35 分が下限） |

## トラブル時

- **`CUDA out of memory`** → `--batch_size 4` に下げる
- **ワーカーがすぐ落ちる** → `logs/worker_gpu<i>.log` の `Traceback` を確認。pyrender/EGL 系の問題なら `DISPLAY=` を空にして再実行
- **既存出力を再走させたい** → `--no_skip_existing` を付ける
- **特定動画だけ処理したい** → `--files_from <list.txt>`（改行区切りのパスリスト）
