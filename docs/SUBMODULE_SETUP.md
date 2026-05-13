# 親プロジェクトに hamer をサブモジュール化して使う手順

将来 `hamer_api.Hamer` クラスを別プロジェクト（UniDAC / MimicAnno 等）から呼ぶ場合の構成と、その際に壊れる `.hamer/` venv の再構築手順をまとめる。

## ゴール

```
親プロジェクト/
├── .venv/                       ← 親側の venv（hamer の依存も入れる）
├── ...
└── third-party/
    └── hamer/                   ← この repo を git submodule で追加
        ├── hamer_api.py
        ├── demo_video.py
        ├── _DATA/               ← MANO + checkpoints（gitignored、別配置）
        └── third-party/
            └── ViTPose/         ← hamer の入れ子 submodule
```

親プロジェクトから:
```python
from hamer_api import Hamer   # third-party/hamer が sys.path に入っている前提
h = Hamer(body_detector='regnety')
out = h.infer_image('frame.jpg')
```

## 何が「壊れる」のか

git submodule で追加した時点では：
- ✅ Python ソースは入る
- ❌ **`.hamer/` venv は無い**（gitignored）
- ❌ **`_DATA/data/mano/MANO_RIGHT.pkl` 無い**（`/_DATA/` が gitignored）
- ❌ **`_DATA/hamer_ckpts/` のモデル重み無い**（同上）
- ❌ **`third-party/ViTPose/` は空ディレクトリ**（入れ子 submodule、未初期化）
- ❌ **`result/` シンボリックリンクは保存されるが、リンク先 `/misc/dl00/gayagaya/hamer_results` が新環境に無いと dead link**

→ ソースだけ来て **依存物 + venv + 入れ子 submodule + データ** は手動再構築が必要。

## セットアップ手順

### 1. 親プロジェクトに submodule として追加

```bash
cd 親プロジェクト
git submodule add git@github.com:gayagayataiga/hamer.git third-party/hamer
cd third-party/hamer
git checkout personal           # 改造ブランチ
git submodule update --init --recursive   # 入れ子 ViTPose も取得
cd ../..
```

### 2. 依存をインストール（2 通り）

#### 案 A: 親 venv に直接入れる（API 利用のみ、推奨）

```bash
cd 親プロジェクト
source .venv/bin/activate

# PyTorch（CUDA バージョンに合わせる、例 cu117）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# hamer 本体 + extras（hydra など）
pip install -e third-party/hamer[all]

# ViTPose（入れ子 submodule）
pip install -v -e third-party/hamer/third-party/ViTPose
```

#### 案 B: hamer 内に専用 venv を作る（独立性が欲しい場合）

```bash
cd third-party/hamer
python3.10 -m venv .hamer
source .hamer/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -e .[all]
pip install -v -e third-party/ViTPose
```

親から呼ぶときは `subprocess` 経由か、`.hamer/bin/python` を直接指定する形になる。

### 3. モデルチェックポイント取得（`fetch_demo_data.sh`）

```bash
cd third-party/hamer
bash fetch_demo_data.sh
# → _DATA/hamer_demo_data.tar.gz をダウンロード → 展開
# → _DATA/hamer_ckpts/, _DATA/vitpose_ckpts/ などが生成される
```

**約 6 GB のダウンロード**なので一度だけ。サーバー間でコピーして共有しても OK。

### 4. MANO モデル配置（公式登録が必要）

`MANO_RIGHT.pkl` は MANO サイト（https://mano.is.tue.mpg.de）に登録してダウンロードする必要がある。

既に持っているサーバーなら **シンボリックリンクで OK**：
```bash
mkdir -p _DATA/data/mano
ln -s /misc/dl00/gayagaya/ft-change/mano_v1_2/models/MANO_RIGHT.pkl _DATA/data/mano/MANO_RIGHT.pkl
```

（`docs/setup_notes.md` に既存 MANO ファイル位置の一覧あり）

### 5. `result/` のリンク先処理（任意）

submodule 内の `result` は元サーバー (`/misc/dl00/gayagaya/hamer_results`) を指したシンボリックリンク。新環境では：

- **過去 result を見ない場合**: そのまま放置（dead link、害なし）または `rm result` で削除
- **過去 result を使う場合**: シンボリックリンクを新環境の保存先に張り替え
  ```bash
  rm result
  ln -s /your/new/path/hamer_results result
  ```
- **新規 result を作る場合**: 出力先を呼び出し時に `output_dir=...` で指定すれば `result/` を使わない

### 6. 動作確認

```bash
# venv 起動 (案 A なら親 venv、案 B なら .hamer/)
python -c "
from hamer_api import Hamer
h = Hamer(body_detector='regnety')
print('pipeline built OK')
"
```

短いテスト動画があれば：
```bash
python -c "
from hamer_api import Hamer
h = Hamer(body_detector='regnety')
h.infer_video('test.mp4', wrist_json='test.json')
"
```

## 一発再構築スクリプト（実装済 + 検証済、2026-05-13）

**`scripts/rebuild_env.sh` に実装済み**。fresh clone + 既存ストレージ依存ゼロでテストし、smoke test (`OK: pipeline loaded (HAMER on cuda)`) まで通過することを確認。

```bash
cd third-party/hamer
bash scripts/rebuild_env.sh
```

主なオーバーライド可能環境変数：

| 変数 | 既定値 | 用途 |
|---|---|---|
| `PYTHON_BIN` | `python3.10` | 使う Python |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu124` | torch のホイール index |
| `MANO_SOURCE` | （自動検出） | MANO_RIGHT.pkl のソースパス |

### 検証で得た 6 つの落とし穴（重要、削るな）

10 回試行して安定化させた結果、以下が地雷だった。スクリプト内コメントにも同じ理由が書いてあるが、変更する人のためにここにも残す：

1. **`uv venv` だけだと pip が入らない** → `uv venv --seed` で pip/setuptools/wheel を同梱しないと、後段の `pip install` が anaconda などシステム側 pip を呼んでしまう
2. **torch のホイール index は cu124**（既存環境の `torch.version.cuda == 12.4` に合わせる。`cu117` だと detectron2 ビルド時に CUDA バージョン不一致でコケる）
3. **detectron2 は `--no-build-isolation` 必須** — その `setup.py` が `import torch` するが、PEP 517 isolated build 環境からは torch が見えないので普通に install すると失敗
4. **setuptools は <70 に pin** — setuptools 81 で `pkg_resources` が削除されたが、`torch.utils.cpp_extension` がまだ `from pkg_resources import packaging` をやる
5. **numpy は <2 に pin、しかも `.[all]` install 時にも inline で渡す** — xtcocotools の precompiled C 拡張は numpy 1.x ABI で固定されてる。step 4 で pin しても `.[all]` で transitive dep が numpy 2.x を引いてくる
6. **`fetch_demo_data.sh` は cwd にタールボールを落とす**が、`hamer.models.download_models()` は `_DATA/hamer_demo_data.tar.gz` を探す。fetch 後に `mv hamer_demo_data.tar.gz _DATA/` しないと、smoke test 時に hamer が再 DL（6 GB）を始める

設計と検証手順の詳細は `docs/REBUILD_SCRIPT_PLAN.md`。

## 親プロジェクトでの利用例（最小）

```python
# 親プロジェクト内、import path に hamer サブモジュールを通す
import sys, pathlib
HAMER = pathlib.Path(__file__).parent / "third-party/hamer"
sys.path.insert(0, str(HAMER))

# hamer は `_DATA/` 内のファイルを相対パスで参照するので chdir する
import os
os.chdir(HAMER)

from hamer_api import Hamer
h = Hamer(body_detector='regnety')
result = h.infer_image('/path/to/frame.jpg')
print(result['hands'])
```

`build_pipeline()` 内で既に親 cwd → hamer root への `chdir + 戻し` 処理を入れているので、上の `os.chdir` は不要かもしれない。要動作確認。

## 注意点

- **Detectron2 のインストールが遅い/重い**（git からビルドする）。初回 10〜20 分かかることがある
- **mmcv バージョン**: `setup.py` は `mmcv==1.3.9` を要求するが、ViTPose 側と衝突する場合は手動で調整
- **CUDA/Torch バージョンが親プロジェクトと違う**と OOM や ABI 不一致が出るので、案 A を採るなら親側に合わせる
- **`build_pipeline` は cwd を hamer root に一時切替**するので、マルチスレッドで複数インスタンス並列起動は競合する。`Hamer()` を共有して使う想定

## 関連ファイル

- `docs/setup_notes.md` — MANO ファイルの探し方
- `setup.py` — Python 依存リスト
- `fetch_demo_data.sh` — 公式チェックポイントの取得
- `.gitmodules` — ViTPose 入れ子 submodule の URL
- `docs/runbooks/parallel.md` — マルチ GPU で使う場合
