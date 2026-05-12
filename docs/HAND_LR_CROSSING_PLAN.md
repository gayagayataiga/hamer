# 左右手の x 座標交差フィルタ 計画

`docs/HAND_DEDUP_PLAN.md` の A（L/R 排他）+ C（最小サイズ）の後に **追加で投入する後処理フィルタ**。

## 仮定

**入力動画はエゴセントリック（GoPro 頭部 or 胸部）視点**で、人物自身の腕の交差は通常起きない。
- 装着者の左手 → 画像の **左側（小さい x）** に映る
- 装着者の右手 → 画像の **右側（大きい x）** に映る
- HaMeR の `is_right = 0` は装着者の左手、`is_right = 1` は装着者の右手

したがって、**`is_right=0`（左手）の x_center > `is_right=1`（右手）の x_center は矛盾**。

第三者視点（鏡像）や手を交差する動作（ヨガなど）には適用できない仮定なので、**オプション化**する。

## 現状の残課題（GX010085 で 25% に 3-hand）

dedup（A+C）投入後の GX010086 は 2-hand=93.9% まで改善したが、**GX010085 は 63.9% で 3-hand が 425 frames (25%)残っている**。3 個目の手の正体は以下が候補：

1. ROI が複数（鏡像反射が body detector に拾われ別人物として ROI 化）
2. 大きめの反射 bbox（短辺 > 64 px をすり抜ける）
3. 装着者以外の手（共同作業者など）

→ L/R が x 座標で **矛盾する pair** を弾けば 1 と 2 のうち少なくとも片方が消える見込み。
人物の手（is_right=0, 1）が双方確かに装着者の手なら crossing は起きないので、crossing している pair の方が偽。

## フィルタ設計

### F-1. 純 crossing 検出（最低限）

1 フレーム内で `is_right=0` (左手) の集合 L と `is_right=1` (右手) の集合 R を取り出す。
- すべての (l, r) ペアについて `x_center(l) > x_center(r)` なら crossing 候補
- crossing している pair のうち **mean confidence が低い方を棄却**

### F-2. 重心ベース cluster 化（拡張）

frame 内の全 hand を bbox x_center でソートし：
- 中央値を境にして「左クラスタ」「右クラスタ」に分ける
- 左クラスタ内で `is_right=1` がいれば棄却（**ラベル不一致**）
- 右クラスタ内で `is_right=0` がいれば棄却
- さらに「**1 クラスタに 2 個以上**」あれば low-confidence 側を棄却

F-2 は 3+ 手のケースも整理できるので強い。F-1 はシンプル。

### 推奨

**F-1 から始める**（実装小、効果見極めやすい）。GX010013/GX010085 で残骸が消えなければ F-2 に拡張。

## 実装場所

`demo_video.py::detect_hand_bboxes` の return 直前か、別関数に切り出して呼ぶ：

```python
def _filter_crossing(bboxes, is_right, confidences, enable=True):
    if not enable or len(bboxes) < 2:
        return bboxes, is_right
    # x_center for each
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    keep = np.ones(len(bboxes), dtype=bool)
    L_idx = np.where(is_right == 0)[0]
    R_idx = np.where(is_right == 1)[0]
    for li in L_idx:
        for ri in R_idx:
            if cx[li] > cx[ri] and keep[li] and keep[ri]:
                # crossing — drop lower-confidence side
                if confidences[li] < confidences[ri]:
                    keep[li] = False
                else:
                    keep[ri] = False
    return bboxes[keep], is_right[keep]
```

そのために `detect_hand_bboxes` を **confidence も返す**ように拡張する必要がある（既に L/R 排他選択で計算済なので保持して返すだけ）。

## 期待効果見積もり

GX010085 で現在 3-hand が 425/1397 frames。crossing 起因（reflection 系含む）の比率を仮に 60% とすれば：

| 状態 | 2-hand% | 3-hand 残数 |
|---|---|---|
| 現状 (A+C) | 63.9% | 425 |
| **F-1 投入後** | **~80%** | ~170 |
| F-2 投入後 | ~88% | ~50 |

3 個目が「実際に第 3 者が映り込んでる」「腕の交差」など仮定外のケースは残る。

## 副作用 / リスク

- **エゴセントリック仮定**: 第三者視点では誤って正しい手を棄却する → **CLI / API フラグでデフォルト ON / OFF 切替可能に**
- **複数人**: 2 人並んで作業しているシーンでは、人物 A の R 手と人物 B の L 手が crossing 判定される可能性。ROI（person bbox）でグルーピングしてから crossing 判定する案もあるが、まずは frame 全体で評価し、必要なら ROI 化
- **動作中の交差**: 拍手・腕組みなどで一時的に crossing する → frame 単発なので大勢に影響しない見込み。気になれば temporal smoothing と組合せ

## 実装ステップ

1. `detect_hand_bboxes` の return に `confidences` を含める（内部 3 つ目戻り値）
2. `_filter_crossing()` ヘルパーを追加
3. `process_frame` の bbox 受け取り側で `confidences` も受ける
4. `detect_hand_bboxes(..., reject_lr_crossing=True)` フラグ追加
5. `process_video(..., reject_lr_crossing=True)` 引数追加
6. `hamer_api.hamer(..., reject_lr_crossing=True)` 引数追加
7. CLI `--no_reject_lr_crossing` を追加（デフォルト ON）
8. `Hamer.infer_*` のオプションにも追加

## 検証計画（GX010013 で）

1. **再走前の baseline** を取る（現状 GX010013 は dedup なし v2 で生成済み）
   - 現在の `result/GX010013/wrist.json` の手数分布を計測
2. **dedup のみ (A+C)** で再走、手数分布計測
3. **dedup + F-1** で再走、手数分布計測
4. 3 つを比較し、F-1 の追加効果を実測

GPU 1 つで GX010013（1f 数取得して）数分で済む。

## ファイル変更まとめ

- `demo_video.py`: `detect_hand_bboxes` 拡張、`_filter_crossing` 追加、`process_frame` シグネチャ拡張
- `hamer_api.py`: `hamer()` / `Hamer.infer_*` / CLI に `reject_lr_crossing` 渡し
- `docs/HAND_LR_CROSSING_PLAN.md`: 本ファイル
- `NEXT.md`: 投入後に効果を追記
