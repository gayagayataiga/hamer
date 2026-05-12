# ディレクトリ整理 計画

`/misc/dl00/gayagaya/hamer/`（personal ブランチ）の現状を見て、整理候補と方針をまとめる。
**実行はしない**。判断後に進める。

## 現状サマリ

### 直近のセッションで増えたもの

| ファイル / ディレクトリ | サイズ | 種類 |
|---|---|---|
| `NEXT.md` | 15 KB | セッション横断のメモ・運用ノート |
| `HAND_DEDUP_PLAN.md` | 7 KB | 設計ドキュメント |
| `RUN_PARALLEL.md` | 3.7 KB | 別サーバー実行手順 |
| `RUN_VIDEO_NEW.md` | 4.3 KB | video/new 実行手順 |
| `SETUP_NOTES.md` | 1.3 KB | venv セットアップ覚書 |
| `demo_video.py` | 28 KB | 改造済み（dual_output / dedup） |
| `hamer_api.py` | 17 KB | クラス API + parallel + dual |
| `result/` | **7.3 GB** | 全動画の `_full.mp4` / `_handsonly.mp4` / `_wrist.json` |
| `result/wrist/` | （内）| v2 JSON 32 本（v1 から再生成済み） |
| `__pycache__/` | — | Python キャッシュ（gitignored 想定） |

### 元々あるもの（upstream 由来、触らない）

- `LICENSE.md`, `README.md`
- `demo.py`, `eval.py`, `train.py`, `vitpose_model.py`, `setup.py`
- `hamer/`, `hamer.egg-info/`, `assets/`, `example_data/`, `third-party/`, `docker/`, `_DATA/`
- `fetch_demo_data.sh`, `fetch_training_data.sh`

### 不要そう・気になるもの

| 項目 | コメント |
|---|---|
| `hamer_evaluation_data.tar.gz` (3.9 MB) | 一度展開した tarball、もう要らないなら削除 |
| `out_videos/` | upstream 由来の空ディレクトリ、無害だが放置でも可 |
| `/tmp/hamer_*` ログ・テストファイル | セッション後始末。削除可（ジョブ完了後）|

## 整理候補

### A. ルートの Markdown 群を `docs/` にまとめる

**Before:**
```
hamer/
├── NEXT.md
├── HAND_DEDUP_PLAN.md
├── RUN_PARALLEL.md
├── RUN_VIDEO_NEW.md
├── SETUP_NOTES.md
├── README.md          ← upstream
├── LICENSE.md         ← upstream
└── ...
```

**After:**
```
hamer/
├── docs/
│   ├── NEXT.md
│   ├── HAND_DEDUP_PLAN.md
│   ├── runbooks/
│   │   ├── parallel.md           ← RUN_PARALLEL.md
│   │   └── video_new.md          ← RUN_VIDEO_NEW.md
│   └── setup_notes.md            ← SETUP_NOTES.md
├── README.md
├── LICENSE.md
└── ...
```

| メリット | デメリット |
|---|---|
| upstream README が目立つ | パス変更で git mv 必要、NEXT.md がワンクリックでなくなる |
| 種別ごと（plan / runbook / note）に整理 | NEXT.md は頻繁にアクセスするので逆に不便 |

**判断ポイント:** NEXT.md だけルート維持、他を `docs/` に移すハイブリッド案もあり。

### B. `result/` 構造化

**Before（flat）:**
```
result/
├── GX010001_full.mp4
├── GX010001_handsonly.mp4
├── GX010003_full.mp4
├── ...（×80+ ファイル）
└── wrist/
    └── *_wrist.json (32)
```

**After（動画ごとサブディレクトリ）:**
```
result/
├── GX010001/
│   ├── full.mp4
│   ├── handsonly.mp4
│   └── wrist.json
├── GX010003/
│   ├── full.mp4
│   ├── handsonly.mp4
│   └── wrist.json
└── ...
```

| メリット | デメリット |
|---|---|
| 1 動画あたりの成果物が一望できる | hamer_api 側の出力命名規約を変える必要 |
| 下流が「`<video>/wrist.json`」で取りに来やすい | 既存出力 80+ ファイルを移動する手間 |

**判断ポイント:** 下流（UniDAC など）がどう取りに来るかで決まる。今のままで困ってなければ後回し。

### C. wrist JSON 配置を一本化

現状の混在状態：

| パス | スキーマ | 由来 |
|---|---|---|
| `result/wrist/*.json` | v2 (32 本) | v2 再走後 |
| `result/GX010085_wrist.json` | v2 | 直近の dual-output 実行が `result/` 直下に書いた |
| `result/GX010086_wrist.json` | v2 | 同上 |

`result/wrist/` に揃えるか、`result/` 直下に揃えるか、はたまた B のサブディレクトリ案か。
**B を採用するなら自動で解決**。B しないなら：

- `wrist/` サブディレクトリに集約 → 既存 GX01008[56]_wrist.json を mv するだけ。`hamer_api.hamer()` の `output_dir` を呼び出し側で `result/wrist/` に分ける運用にする
- `result/` 直下にフラットに置く → `result/wrist/` から 32 本を `result/` に mv

**判断ポイント:** 整理メリットは小だが、下流コードが「どこに wrist.json があるか」決め打ちで書かれているなら影響あり。

### D. 不要ファイル削除

| 対象 | サイズ | 影響 |
|---|---|---|
| `hamer_evaluation_data.tar.gz` | 3.9 MB | 既に `_DATA/` に展開済みなら不要 |
| `out_videos/` | 0 | upstream 由来空ディレクトリ、消しても害ない |
| `__pycache__/` | — | 走らせれば再生成。gitignored なら現状で良い |
| `/tmp/hamer_*.log` | 数 MB | セッション後始末。**ジョブ完了前は消さない** |
| `/tmp/gpu{1,3}_files.txt` | 数 byte | テンポラリ、消しても可 |

### E. result/ そのものを別の場所に逃がす

`result/` 7.3 GB がリポジトリ直下にあると `find` や `git status` でうるさい。

選択肢:
- **シンボリックリンク化**: `result/` を `/misc/dl00/gayagaya/hamer_results/` 等に移してリンクを張る → ストレージ的には変わらないが、リポジトリは軽くなる
- **`.gitignore` に追加**: 既に追加されている？要確認

`git status` でちらつくのを抑えるだけなら .gitignore で十分。物理的に動かすのは下流が読みに来てる経路次第。

## 推奨優先順位

1. **D の不要ファイル削除**（リスク 0、即実施可）
   - `hamer_evaluation_data.tar.gz` 削除確認
   - `/tmp/hamer_*` ログのうち、ジョブ終了済みのものをクリーンアップ
2. **C の wrist JSON 配置一本化**（B 採用なら不要）
   - 下流の読みに来ている経路を確認 → 揃える方向を決める
3. **B の `result/` サブディレクトリ化**（やる場合は hamer_api 改修付き）
   - 下流の API 変更があるなら今のうちに
4. **A の docs/ 整理**（任意、好み）
   - 散らかった感はあるが致命的ではない
5. **E の result/ 場所移動**（下流影響大きいので慎重に）

## やらない方が良いこと

- `_DATA/`, `hamer/`, `hamer.egg-info/`, `third-party/` を触る（upstream / ビルド産物）
- `result/wrist/` の既存 v2 JSON を消す（再生成にコストがかかる）

## 補足

`.gitignore` の中身を確認すれば、何が tracked / untracked / ignored か明確になる。
特に `result/`, `__pycache__/`, `*.tar.gz` の扱いを再確認するとよい。

---

# 実行計画（D → C → B → A → E）

各フェーズで **(1) 事前確認 (2) 操作 (3) 検証 (4) ロールバック手段** を揃える。
各フェーズはコミット境界で区切り、問題が起きたら `git revert` で戻せるようにする。

## 🛡️ 絶対に守る方針：実行結果は削除しない

`result/` 配下の `_full.mp4` / `_handsonly.mp4` / `_wrist.json`（および `result/wrist/*.json`）は **後で JSON データとして使う前提**。
**`rm` 対象には絶対に入れない。**

- **D（削除）** — `result/` 配下には触らない。削除対象は `hamer_evaluation_data.tar.gz`、`/tmp/` のログ、空の `out_videos/`、Python キャッシュのみ
- **C / B（再配置）** — `mv` で **移動**のみ。中身を消さない
- **E（場所移し）** — `mv` + シンボリックリンクで **アクセス経路を維持**。物理的に消さない
- **A（docs/ 整理）** — Markdown のみで、`result/` には無関係

各フェーズの操作前に `ls result/ | wc -l` でファイル数を控え、操作後に同数であることを確認する。

## 共通の前提

- **走行中ジョブを止めない** — `tail -F /tmp/hamer_dedup_*` 確認、ジョブ完了後に作業
- **作業ブランチ:** `personal`（小さい変更ごとにコミット）
- **ハードリンクではなく `git mv`** を使う（履歴が追える）
- **下流（UniDAC など）が読みに来るパス** が変わる場合は事前に経路把握
- **`result/` 配下は不可触領域** — 削除コマンドは打たない、移動のみ

## .gitignore 整備（先にやっとく）

現状 `result/` は ignored になっておらず（`/results/` 複数形のみ）、`git status` で `?? result/` が常に出る。整理前に：

```bash
# .gitignore に追加
echo "" >> .gitignore
echo "# Personal: large output mp4/json from hamer runs" >> .gitignore
echo "/result/" >> .gitignore

git add .gitignore && git commit -m "Ignore result/ output directory"
git status   # result/ が消えていることを確認
```

ロールバック: `git revert HEAD` または `.gitignore` から該当行削除。

---

## D. 不要ファイル削除

### D.1 `hamer_evaluation_data.tar.gz` (3.9 MB)

事前確認:
```bash
ls -la hamer_evaluation_data.tar.gz
file hamer_evaluation_data.tar.gz
tar -tzf hamer_evaluation_data.tar.gz | head -10
# _DATA/ に同等の中身が展開されているか確認
ls _DATA/data/  # eval 用データの形跡があるか
```

操作:
```bash
rm hamer_evaluation_data.tar.gz
```

検証: 再 `python eval.py` を回すケースが無いなら影響なし。
ロールバック: `fetch_demo_data.sh` を見て再ダウンロード可能なら問題なし。

### D.2 `/tmp/hamer_*.log` クリーンアップ

事前確認:
```bash
# 走行中ジョブが無いことを確認
ps -ef | grep "hamer_api" | grep -v grep
ls -la /tmp/hamer_*.log /tmp/hamer_*_logs /tmp/gpu*_files.txt
```

操作（ジョブ完全終了後のみ）:
```bash
rm -f /tmp/hamer_v2_run.log /tmp/hamer_episode_v2.log \
      /tmp/hamer_new_run.log /tmp/hamer_gpu1.log /tmp/hamer_gpu3.log \
      /tmp/hamer_dedup_run.log
rm -rf /tmp/hamer_dedup_logs /tmp/hamer_new_logs
rm -f /tmp/gpu1_files.txt /tmp/gpu3_files.txt /tmp/hamer_t0
```

ロールバック: `/tmp` 配下なので再生成可、ロールバック不要。

### D.3 `out_videos/` 削除

事前確認:
```bash
ls -la out_videos/   # 中身があれば残す
```

操作（空であれば）:
```bash
rmdir out_videos
git add -A && git commit -m "Remove empty out_videos/ (upstream stub)"
```

ロールバック: `mkdir out_videos` で復元、コミット revert。

### D.4 `__pycache__/`

`.gitignore` で既に除外済み。物理的に消したければ：
```bash
rm -rf __pycache__ hamer/__pycache__ hamer/*/__pycache__
```
git 影響なし。

**フェーズ D の完了基準:** ルート直下のゴミファイルが消え、`/tmp` のテスト残骸ゼロ。コミット 1 本（または無し）。

---

## C. wrist JSON 配置一本化

### C.1 現状再確認

```bash
ls result/wrist/*_wrist.json | wc -l   # 期待 32 (またはそれ以上)
ls result/*_wrist.json                  # 期待 GX010085/86 の 2 本
```

### C.2 下流の参照経路を確認

`/misc/dl00/gayagaya/` 配下で `_wrist.json` を読んでるコードを探す：
```bash
grep -rn "_wrist.json\|/wrist/" /misc/dl00/gayagaya/MimicAnno \
                                  /misc/dl00/gayagaya/UniDAC \
                                  2>/dev/null | head -20
# 該当パスがあれば、それに合わせる
```

下流が `result/wrist/<stem>_wrist.json` を読んでいるなら → C.3a に進む。
下流が `result/<stem>_wrist.json` を読んでいるなら → C.3b に進む。
どちらも参照していないなら → C.3a を採用（整理度が高い）。

### C.3a 全て `result/wrist/` に集約

```bash
mv result/GX010085_wrist.json result/wrist/
mv result/GX010086_wrist.json result/wrist/
ls result/wrist/ | wc -l   # 34 になっているはず
```

`hamer_api.hamer()` を「wrist JSON だけ別の出力先に書く」ように改修:
- 新引数 `wrist_output_dir=None` を追加（None なら `output_dir` と同じ）
- ループ内 `wrist_out = (wrist_output_dir or output_dir) / f"{stem}_wrist.json"`
- CLI `--wrist_output_dir` を追加
- Hamer クラス `infer_dir(wrist_output_dir=...)` に対応

検証:
```bash
ls result/wrist/GX010085_wrist.json result/wrist/GX010086_wrist.json
python -c "import json; [print(f, json.load(open(f))['schema_version']) for f in __import__('glob').glob('result/wrist/*.json')]" | sort | uniq -c
```

コミット 2 本: 「Move stray wrist JSON into result/wrist/」 + 「Add wrist_output_dir option to hamer()」。

### C.3b 全て `result/` 直下にフラット化

```bash
mv result/wrist/*_wrist.json result/
rmdir result/wrist
```

`hamer_api.hamer()` は既にこの動作（output_dir 直下に書く）なので追加コード変更なし。

検証:
```bash
ls result/*_wrist.json | wc -l   # 34
test ! -d result/wrist && echo OK
```

コミット 1 本: 「Flatten wrist JSON layout into result/」。

**フェーズ C の完了基準:** wrist json が単一ディレクトリに揃う、`hamer_api` の出力先制御が下流の参照経路と一致、**JSON ファイル数の合計が操作前と一致**（移動のみで欠損なし）。

データ保全チェック:
```bash
# 操作前
BEFORE=$(find result -name "*_wrist.json" | wc -l)
echo "before: $BEFORE"
# 操作後
AFTER=$(find result -name "*_wrist.json" | wc -l)
echo "after: $AFTER"
test "$BEFORE" = "$AFTER" && echo "OK: no data lost" || echo "FAIL: count mismatch"
```

---

## B. `result/` をビデオごとサブディレクトリ化

C で「wrist/ に集約」した場合この B は意味が薄れる。**C.3b（フラット）を採用したケースで効果大**。

### B.1 ファイル命名規約変更

| 旧 | 新 |
|---|---|
| `result/GX010001_full.mp4` | `result/GX010001/full.mp4` |
| `result/GX010001_handsonly.mp4` | `result/GX010001/handsonly.mp4` |
| `result/GX010001_wrist.json` | `result/GX010001/wrist.json` |

### B.2 既存ファイル移行スクリプト

```bash
# dry-run
for f in result/*_full.mp4 result/*_handsonly.mp4 result/*_wrist.json; do
    base=$(basename "$f")
    stem=$(echo "$base" | sed -E 's/_(full|handsonly|wrist)\.(mp4|json)$//')
    suffix=$(echo "$base" | sed -E 's/.*_(full|handsonly|wrist)\.(mp4|json)$/\1.\2/')
    echo "mv $f result/$stem/$suffix"
done | head -10   # まず動作確認

# 実行
for f in result/*_full.mp4 result/*_handsonly.mp4 result/*_wrist.json; do
    base=$(basename "$f")
    stem=$(echo "$base" | sed -E 's/_(full|handsonly|wrist)\.(mp4|json)$//')
    suffix=$(echo "$base" | sed -E 's/.*_(full|handsonly|wrist)\.(mp4|json)$/\1.\2/')
    mkdir -p "result/$stem"
    mv "$f" "result/$stem/$suffix"
done
```

### B.3 `hamer_api` の出力命名変更

`hamer_api.hamer()` の以下を変更:
```python
full_out  = output_dir / stem / "full.mp4"
hands_out = output_dir / stem / "handsonly.mp4"
wrist_out = output_dir / stem / "wrist.json"
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / stem).mkdir(parents=True, exist_ok=True)
```

CLI 互換性のため `--flat_output` フラグで旧命名に戻せると親切。

### B.4 検証

```bash
ls result/ | head
# 期待: GX010001/  GX010003/  ...  GX010085/  GX010086/  episode_000204/

ls result/GX010001/
# 期待: full.mp4  handsonly.mp4  wrist.json  (3 ファイル)

# 短いクリップで新コード走らせて結果確認
python hamer_api.py /tmp/test_dir --output_dir /tmp/test_out
ls /tmp/test_out/clip/
```

**フェーズ B の完了基準:** 全動画が `result/<stem>/{full,handsonly,wrist}.{mp4,json}` になり、新コードも同じ構造で出力、**ファイル総数が操作前後で一致**。

データ保全チェック:
```bash
BEFORE_MP4=$(find result -maxdepth 1 -name "*.mp4" | wc -l)
BEFORE_JSON=$(find result -name "*_wrist.json" | wc -l)
echo "before: mp4=$BEFORE_MP4 json=$BEFORE_JSON"
# ... 移動操作 ...
AFTER_MP4=$(find result -name "*.mp4" | wc -l)
AFTER_JSON=$(find result -name "wrist.json" -o -name "*_wrist.json" | wc -l)
echo "after: mp4=$AFTER_MP4 json=$AFTER_JSON"
test "$BEFORE_MP4" = "$AFTER_MP4" -a "$BEFORE_JSON" = "$AFTER_JSON" && echo OK || echo FAIL
```

---

## A. `docs/` ディレクトリへの整理（ハイブリッド案）

NEXT.md は頻繁にアクセスするので **ルート維持**。それ以外を `docs/` に集約。

### A.1 ファイル配置

```bash
mkdir -p docs/runbooks

git mv HAND_DEDUP_PLAN.md docs/
git mv CLEANUP_PLAN.md docs/
git mv RUN_PARALLEL.md docs/runbooks/parallel.md
git mv RUN_VIDEO_NEW.md docs/runbooks/video_new.md
git mv SETUP_NOTES.md docs/setup_notes.md
```

### A.2 内部参照の修正

`NEXT.md` 内に `RUN_PARALLEL.md`, `RUN_VIDEO_NEW.md`, `HAND_DEDUP_PLAN.md` への言及があるので、新パスに置換:

```bash
sed -i 's|RUN_PARALLEL.md|docs/runbooks/parallel.md|g' NEXT.md
sed -i 's|RUN_VIDEO_NEW.md|docs/runbooks/video_new.md|g' NEXT.md
sed -i 's|HAND_DEDUP_PLAN.md|docs/HAND_DEDUP_PLAN.md|g' NEXT.md
# 必要なら他ファイルも
grep -rn "RUN_PARALLEL\|RUN_VIDEO_NEW\|HAND_DEDUP\|SETUP_NOTES\|CLEANUP_PLAN" \
    *.py *.md docs/ 2>/dev/null
```

### A.3 検証

```bash
ls docs/
# 期待: HAND_DEDUP_PLAN.md  CLEANUP_PLAN.md  runbooks/  setup_notes.md
ls docs/runbooks/
# 期待: parallel.md  video_new.md
ls *.md
# 期待: NEXT.md  README.md  LICENSE.md  (3 つだけ)
```

**フェーズ A の完了基準:** ルートの `.md` がアップストリーム由来 + `NEXT.md` のみ、それ以外は `docs/` 配下。

---

## E. `result/` をリポジトリ外に逃がす

最も影響が大きい。下流の参照経路を完全に把握してから実施。

### E.1 移動先決定

候補:
- `/misc/dl00/gayagaya/hamer_results/` （ストレージは同じ）
- `/misc/dl00/gayagaya/data/hamer_results/`

### E.2 移動 + シンボリックリンク

```bash
DEST=/misc/dl00/gayagaya/hamer_results
mkdir -p "$(dirname $DEST)"
mv result "$DEST"
ln -s "$DEST" result

ls -la result   # シンボリックリンクであることを確認
ls result/      # 中身がアクセスできることを確認
```

### E.3 下流の参照確認

```bash
# シンボリックリンク経由で透過アクセスできるはずだが、
# 絶対パスで読みに来ているコードがあれば書き換え
grep -rn "result/" /misc/dl00/gayagaya/UniDAC /misc/dl00/gayagaya/MimicAnno 2>/dev/null | head
```

### E.4 `.gitignore` 更新

`result` をシンボリックリンクで残しているので `.gitignore` の `/result/` は維持。
追加で `result` （末尾スラッシュ無し）も足しておくと安心。

**フェーズ E の完了基準:** `du -sh result/` でリンク先がアクセス可能、`git status` で `result` に関する出力なし、下流コードが影響なく動く、**移動先にファイルが残っている**。

データ保全チェック:
```bash
# 操作前
BEFORE=$(find result -type f | wc -l)
SIZE_BEFORE=$(du -sb result | cut -f1)
echo "before: files=$BEFORE bytes=$SIZE_BEFORE"
# ... mv + ln -s 操作 ...
# 操作後（シンボリックリンク経由でカウント）
AFTER=$(find -L result -type f | wc -l)
SIZE_AFTER=$(du -sbL result | cut -f1)
echo "after: files=$AFTER bytes=$SIZE_AFTER"
test "$BEFORE" = "$AFTER" -a "$SIZE_BEFORE" = "$SIZE_AFTER" && echo OK || echo FAIL
```

---

# 実行スケジュール（目安）

| フェーズ | 所要時間 | リスク | コミット数 |
|---|---|---|---|
| .gitignore 整備 | 2 分 | 0 | 1 |
| D 不要ファイル削除 | 5 分 | 0 | 0〜1 |
| C wrist JSON 統一 | 10〜20 分 | 低（下流確認次第）| 1〜2 |
| B サブディレクトリ化 | 30 分 | 中（hamer_api 改修 + 移動 + テスト）| 2 |
| A docs/ 整理 | 10 分 | 0 | 1 |
| E result/ 逃がし | 20 分 | 高（下流参照次第）| 1 |
| **合計** | **〜90 分** | — | **6〜8** |

各フェーズ完了時に push して、別セッションからも見えるようにする。
