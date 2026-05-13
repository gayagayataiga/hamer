# `scripts/rebuild_env.sh` 計画と検証戦略

サブモジュール化された hamer 環境を **壊れた状態から一発で再構築**するためのシェルスクリプトと、それが本当に動くことを確認する検証手順。

## スクリプトが満たすべき要件

1. **冪等性 (idempotent)**: 何度走らせても安全。既存ステップはスキップ
2. **明示的なログ**: 各ステップ「何をしようとしてるか」を `echo` で見せる
3. **早期失敗 (`set -e`)**: 途中で失敗したら止まる
4. **環境前提を明示**: Python バージョン / CUDA / インターネット接続
5. **MANO 不在のエラーは親切**: 公式登録の URL を出す
6. **依存物の場所を可変に**: `MANO_SOURCE` を環境変数で差し替え可能に

## ステップ詳細

| # | ステップ | 何をする | 既にあれば |
|---|---|---|---|
| 1 | リポジトリ位置の決定 | `cd` to script's parent dir | — |
| 2 | Python 確認 | `python3.10 --version` でバージョンチェック | — |
| 3 | 入れ子 submodule | `git submodule update --init --recursive` | 既にチェックアウト済ならスキップ |
| 4 | venv 作成 | `python3.10 -m venv .hamer` | `.hamer/` あれば再利用 |
| 5 | pip upgrade | `pip install --upgrade pip` | — |
| 6 | torch インストール | `pip install torch torchvision --index-url ...` | `torch` import できればスキップ |
| 7 | hamer 本体 | `pip install -e .[all]` | `hamer` import できればスキップ |
| 8 | ViTPose | `pip install -v -e third-party/ViTPose` | `vitpose_model` import できればスキップ |
| 9 | チェックポイント | `bash fetch_demo_data.sh` | `_DATA/hamer_ckpts/` あればスキップ |
| 10 | MANO 配置 | 既知パスから symlink | 既に `_DATA/data/mano/MANO_RIGHT.pkl` あればスキップ |
| 11 | 動作確認 | `python -c "from hamer_api import Hamer; Hamer(...)"` | 必須（必ず走らせる） |

### 各ステップの判定コマンド

```bash
# 3) submodule
[ -f third-party/ViTPose/setup.py ] || git submodule update --init --recursive

# 4) venv
[ -x .hamer/bin/python ] || python3.10 -m venv .hamer

# 6) torch
source .hamer/bin/activate
python -c "import torch" 2>/dev/null || \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# 7) hamer
python -c "import hamer" 2>/dev/null || pip install -e .[all]

# 8) ViTPose
python -c "import vitpose_model" 2>/dev/null || pip install -v -e third-party/ViTPose

# 9) checkpoints
[ -d _DATA/hamer_ckpts ] || bash fetch_demo_data.sh

# 10) MANO
MANO_DST=_DATA/data/mano/MANO_RIGHT.pkl
if [ ! -e "$MANO_DST" ]; then
    MANO_SOURCE_DEFAULT=/misc/dl00/gayagaya/ft-change/mano_v1_2/models/MANO_RIGHT.pkl
    MANO_SOURCE=${MANO_SOURCE:-$MANO_SOURCE_DEFAULT}
    if [ -f "$MANO_SOURCE" ]; then
        mkdir -p "$(dirname $MANO_DST)"
        ln -s "$MANO_SOURCE" "$MANO_DST"
    else
        echo "ERROR: $MANO_SOURCE not found. Set MANO_SOURCE env var or download from https://mano.is.tue.mpg.de"
        exit 1
    fi
fi

# 11) smoke test
python -c "
from hamer_api import Hamer
h = Hamer(body_detector='regnety')
print('OK: pipeline loads')
"
```

## ファイル配置

```
hamer/
└── scripts/
    └── rebuild_env.sh       ← 新規（chmod +x で実行可）
```

`scripts/` ディレクトリは既存に無いので新規作成。

## 検証戦略

**目的:** 「壊れた状態から自動復旧できる」ことを実機で確かめる。

### 方針

既存の `/misc/dl00/gayagaya/hamer/` を壊すのは論外なので、**別の場所にクローン**して走らせる。
ストレージは共通なのでクローンは高速、シンボリックリンクで MANO は共有可。

### 検証手順

```bash
TEST_ROOT=/tmp/hamer_rebuild_test
rm -rf $TEST_ROOT
git clone -b personal git@github.com:gayagayataiga/hamer.git $TEST_ROOT
cd $TEST_ROOT
git submodule update --init --recursive   # ViTPose 取得（時間かかる）

# rebuild スクリプト実行（前提: scripts/rebuild_env.sh が repo にコミット済 ※今回はローカルからコピーで代用）
cp /misc/dl00/gayagaya/hamer/scripts/rebuild_env.sh scripts/
bash scripts/rebuild_env.sh
```

**判定基準:**
1. すべての pip install が完走
2. `_DATA/hamer_ckpts/` に重みが入っている（or fetch が完了）
3. `_DATA/data/mano/MANO_RIGHT.pkl` が存在（symlink）
4. ステップ 11 の smoke test が `OK: pipeline loads` を出す

### 想定所要時間

| ステップ | 時間 |
|---|---|
| git clone + submodule init | 1〜2 分 |
| torch install | 3〜5 分 |
| pip install -e .[all] (`detectron2` ビルド込み) | 10〜20 分 |
| ViTPose install | 2〜3 分 |
| fetch_demo_data.sh（6 GB ダウンロード） | 5〜15 分（回線次第） |
| smoke test (pipeline ロード) | 1 分 |
| **合計** | **約 25〜45 分** |

### ショートカット（時短）

既存 `.hamer/` と `_DATA/` を流用すれば検証を高速化できる：

```bash
# venv を流用
ln -s /misc/dl00/gayagaya/hamer/.hamer $TEST_ROOT/.hamer
# _DATA を流用
ln -s /misc/dl00/gayagaya/hamer/_DATA $TEST_ROOT/_DATA
```

ただしこれだと「venv も DATA も無いゼロ状態」を再現できないので、**本来の検証としては流用無しが望ましい**。
時短したい場合のみ採用、その旨ドキュメントに記載。

### リスク・注意

- **pip install 中に既存 venv との競合**: テスト venv は別パスなので無関係
- **fetch_demo_data.sh は gdown を使う**: gdown が動作不安定なら wget 版に切替（スクリプト内コメント参照）
- **MANO のシンボリックリンク先（`/misc/dl00/...`）が新環境で見えない場合**: その時は MANO 登録 → 手動配置の手順を案内
- **検証中に既存 hamer 環境を変更しない**: テストは完全に隔離した `/tmp/hamer_rebuild_test/` 内で

## 検証成功時の最終ステップ

1. `scripts/rebuild_env.sh` を本リポジトリに追加（chmod +x）
2. `docs/SUBMODULE_SETUP.md` の「一発再構築スクリプト案」セクションを「→ `scripts/rebuild_env.sh` 参照」に短縮
3. `NEXT.md` に「再構築テスト済み」と記載
4. commit + push (`personal` ブランチ)

## 検証失敗時のフォールバック

`scripts/rebuild_env.sh` のどのステップで止まったかを記録し、計画と差分を `docs/REBUILD_SCRIPT_PLAN.md` に追記。複数回試行して安定化させる。
