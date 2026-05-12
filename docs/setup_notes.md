# HaMeR セットアップメモ

## MANO モデルのありか（サーバー内）

`find /misc/dl00/gayagaya /home/gayagaya -iname "MANO*.pkl"` で見つかったもの：

| パス | 備考 |
|---|---|
| `/misc/dl00/gayagaya/ft-change/mano_v1_2/models/MANO_RIGHT.pkl` | 公式配布 (`mano_v1_2`) のディレクトリ構造そのまま。**推奨** |
| `/misc/dl00/gayagaya/ft-change/mano_v1_2/models/MANO_LEFT.pkl` | 同上（左手） |
| `/home/gayagaya/ft-change/mano_v1_2/models/MANO_RIGHT.pkl` | 上の home 側コピー（同じもの？） |
| `/home/gayagaya/ft-change/mano_v1_2/models/MANO_LEFT.pkl` | 同上 |
| `/misc/dl00/gayagaya/CrossDex/mano-models/MANO_RIGHT.pkl` | CrossDex 用 |
| `/home/gayagaya/CrossDex/mano-models/MANO_RIGHT.pkl` | 同上 |
| `/misc/dl00/gayagaya/ft-change/retarget_dex3/data/mano_model/MANO_RIGHT.pkl` | retarget_dex3 用 |
| `/home/gayagaya/ft-change/retarget_dex3/data/mano_model/MANO_RIGHT.pkl` | 同上 |

## HaMeR への配置

HaMeR は `_DATA/data/mano/MANO_RIGHT.pkl` を参照する（`hamer/configs_hydra/experiment/default.yaml` の `MANO.MODEL_PATH`）。
シンボリックリンクで通せばOK：

```bash
cd /misc/dl00/gayagaya/hamer
mkdir -p _DATA/data/mano
ln -s /misc/dl00/gayagaya/ft-change/mano_v1_2/models/MANO_RIGHT.pkl _DATA/data/mano/MANO_RIGHT.pkl
```
