# `/home/gayagaya/video/new/` を別セッションで走らせる手順

GX010085.MP4 (1397f) + GX010086.MP4 (1630f) を `result/` に
`_full.mp4` / `_handsonly.mp4` / `_wrist.json` の 3 出力で生成する。

`personal` ブランチの **単パス dual-output 化**（commit 予定、本セッションで実装済み）が必要。
旧コードだと推論 2 回回って倍時間かかるので、必ず最新の `personal` を使うこと。

## 前提

- リポジトリ: `git@github.com:gayagayataiga/hamer.git` の **`personal` ブランチ**
- `.hamer/` venv セットアップ済み
- 入力動画: `/home/gayagaya/video/new/GX010085.MP4`, `GX010086.MP4`（合計 3027 frames）
- 出力先: リポジトリ直下の `result/`

## 起動前に状態確認

```bash
cd /misc/dl00/gayagaya/hamer   # bind mount で /home/gayagaya/hamer と同じ

# 1) 走行中の hamer プロセスがあるか
ps -ef | grep "hamer_api.py" | grep -v grep
# あれば、状況次第で kill (PID は ps の出力から)

# 2) 部分出力が残っていないか
ls -la result/GX010085*.mp4 result/GX010085*.json \
       result/GX010086*.mp4 result/GX010086*.json 2>/dev/null

# 残骸があり --no_skip_existing で再走するなら削除しておく
# rm result/GX010085_*.mp4 result/GX010085_wrist.json \
#    result/GX010086_*.mp4 result/GX010086_wrist.json

# 3) 空いてる GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
```

## 実行コマンド

### A. 単一 GPU（一番安全、~1.5〜2 時間）

GPU 番号は空いてるやつに置換。

```bash
source .hamer/bin/activate
CUDA_VISIBLE_DEVICES=3 python hamer_api.py /home/gayagaya/video/new \
    --output_dir result \
    --no_skip_existing > /tmp/hamer_new_run.log 2>&1 &
```

進捗:
```bash
tail -f /tmp/hamer_new_run.log
```

### B. 2 GPU 並列（推奨、~45〜60 分、`hamer_parallel` 使用）

`--gpus` を空いてる GPU 2 つに合わせる。各 GPU に動画 1 本ずつ振られる（LPT 分配）。

```bash
source .hamer/bin/activate
python hamer_api.py /home/gayagaya/video/new \
    --output_dir result \
    --no_skip_existing \
    --gpus 0,3 \
    --log_dir /tmp/hamer_new_logs > /tmp/hamer_new_run.log 2>&1 &
```

進捗:
```bash
tail -f /tmp/hamer_new_run.log         # 親プロセス（LPT 分配と spawn/exit のみ）
tail -f /tmp/hamer_new_logs/worker_gpu0.log
tail -f /tmp/hamer_new_logs/worker_gpu3.log
```

## 期待する出力

完了時に以下 6 ファイル：

```
result/GX010085_full.mp4
result/GX010085_handsonly.mp4
result/GX010085_wrist.json   ← schema_version: 2
result/GX010086_full.mp4
result/GX010086_handsonly.mp4
result/GX010086_wrist.json   ← schema_version: 2
```

検証:
```bash
python - <<'PY'
import json, glob
for f in sorted(glob.glob('result/GX01008[56]_wrist.json')):
    d = json.load(open(f))
    print(f, 'schema=', d['schema_version'], 'frames=', d['n_frames_processed'])
PY

for f in result/GX01008[56]_*.mp4; do
    echo "$f: $(ffprobe -v error -count_packets -select_streams v:0 \
                  -show_entries stream=nb_read_packets -of csv=p=0 "$f") frames"
done
```

各 `_wrist.json` の `n_frames_processed` と各 `.mp4` のフレーム数が一致していれば OK。

## 重要メモ（単パス dual-output 化）

本セッションで `process_frame` / `process_video` / `hamer()` を改造し、`_full.mp4` と `_handsonly.mp4` を **1 回の推論で同時生成**するようにした。以前は 2 回 process_video を呼んでいて推論コストが倍だった。

- `process_frame(..., dual_output=True)` → 4-tuple `(overlay_bgr, handsonly_bgr, side_bgr, hands_info)`
- `process_video(..., handsonly_output_path=...)` → `output_path` と同時に handsonly mp4 も書き出す
- `hamer_api.hamer()` → 1 video あたり `process_video` 1 回呼び（旧 2 回）

これにより推論時間が **約半分**、レンダリングコストも 1 回分節約。

## トラブル時

- `CUDA out of memory` → `--batch_size 4` を追加
- ワーカー暴走 → `kill <pid>` で止めて再走（`--no_skip_existing` を付けないと既存ファイルでスキップされる）
- pyrender/EGL エラー → `DISPLAY=` を空にして再実行
