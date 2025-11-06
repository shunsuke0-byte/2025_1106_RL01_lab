# マップ切り替え学習ガイド

## 簡単な使用方法

### 1. マップの変更方法

`room_lighting_optimization.py`の上部にある`MAP_PATH`を変更するだけです：

```python
# ============================================================
# マップ設定 - ここを変更するだけで学習できます
# ============================================================

# 使用するマップのパスを指定（Noneの場合はデフォルトマップ）
MAP_PATH = "custom_map.CustomMap500x1000"  # カスタムマップを使用

# 他のマップを使用する場合の例:
# MAP_PATH = None  # デフォルトマップ（四隅に照明）
# MAP_PATH = "room_shape_examples.SixLightsRoom"  # 6つの照明がある部屋
# MAP_PATH = "room_shape_examples.CenterLightRoom"  # 中央に1つ照明がある部屋
```

### 2. 実行方法

#### 方法1: シェルスクリプトを使用（推奨）
```bash
cd /Users/tanakashunsuke/Desktop/universitiy/KubotaLab/python/20251028/project_02
./run_learning.sh
```

#### 方法2: 手動で実行
```bash
cd /Users/tanakashunsuke/Desktop/universitiy/KubotaLab/python/20251028/project_02
source /Users/tanakashunsuke/Desktop/python/python01/bin/activate
python room_lighting_optimization.py
```

## 利用可能なマップ

### デフォルトマップ
```python
MAP_PATH = None  # 四隅に照明がある正方形の部屋
```

### カスタムマップ
```python
MAP_PATH = "custom_map.CustomMap500x1000"  # 1000×500の横長部屋に5つの照明
```


## 新しいマップの追加

### 1. マップファイルを作成
```python
# my_custom_map.py
from room_shape import RoomShapeBase

class MyCustomMap(RoomShapeBase):
    def _define_light_zones(self):
        return {
            'zone1': {
                'x_range': (0, 100),
                'y_range': (0, 100),
                'center': (50, 50)
            },
            # 他のゾーンを追加...
        }
    
    def _define_boundaries(self):
        return {
            'x_min': 0.0,
            'x_max': 500.0,
            'y_min': 0.0,
            'y_max': 500.0
        }
```

### 2. MAP_PATHを設定
```python
MAP_PATH = "my_custom_map.MyCustomMap"
```

## 学習結果

学習が完了すると以下のファイルが生成されます：

- `room_lighting_q_table.npy`: 学習済みQ-table
- `room_lighting_config.json`: 環境設定
- `room_lighting_training.gif`: 学習過程の動画
- `room_lighting_optimal.gif`: 最適照明パターンの動画

## トラブルシューティング

### マップが読み込めない場合
- ファイルパスが正しいか確認
- クラス名が正しいか確認
- インポートエラーがないか確認

### 学習が収束しない場合
- `hyperparameters.py`でパラメータを調整
- エピソード数を増やす
- 学習率を調整

## 設定ファイル

- `hyperparameters.py`: 学習パラメータの設定
- `room_shape.py`: 部屋形状の基底クラス
- `custom_map.py`: カスタムマップの定義

## 可視化ツール

学習結果を可視化する場合：
```bash
python visualize_q_learning.py
```
