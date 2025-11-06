# ハイパーパラメータ管理ガイド

## 概要

`hyperparameters.py`ファイルで、project_02のすべてのハイパーパラメータを一箇所で管理できます。

## ファイル構成

```
project_02/
├── hyperparameters.py          # ハイパーパラメータ設定
├── room_lighting_optimization.py  # メインプログラム（ハイパーパラメータを使用）
├── visualize_q_learning.py     # 可視化プログラム
└── ...
```

## パラメータカテゴリ

### 1. 環境設定パラメータ

```python
MAP_SIZE = 500              # マップサイズ（ピクセル）
AGENT_SPEED = 5             # エージェントの移動速度（ピクセル/ステップ）
ZONE_SIZE = 100             # 照明ゾーンのサイズ（ピクセル）
MAX_STEPS_PER_EPISODE = 200 # 1エピソードの最大ステップ数
```

### 2. Q-Learning パラメータ

```python
N_STATES = 100              # 状態数（10×10グリッドに離散化）
N_ACTIONS = 16              # 行動数（4つの照明の組み合わせ）
LEARNING_RATE = 0.1         # 学習率
DISCOUNT_FACTOR = 0.95      # 割引率（γ）
EPSILON_INITIAL = 1.0       # 初期ε値（完全ランダム）
EPSILON_DECAY = 0.995       # ε減衰率
EPSILON_MIN = 0.01          # 最小ε値
N_EPISODES = 1000           # 総エピソード数
```

### 3. 報酬システムパラメータ

```python
NEAR_THRESHOLD = 200.0      # 近距離判定の閾値（ピクセル）
NEAR_LIGHT_REWARD = 10.0    # 近くて点灯している照明の報酬
NEAR_DARK_PENALTY = -15.0   # 近いのに消灯している照明のペナルティ
FAR_LIGHT_PENALTY = -5.0    # 遠いのに点灯している照明のペナルティ
ENERGY_COST_PER_LIGHT = 1.0 # 1つの照明あたりの電力コスト
```

### 4. 可視化パラメータ

```python
AGENT_RADIUS = 10           # エージェントの表示半径（ピクセル）
ARROW_LENGTH = 20           # 方向矢印の長さ（ピクセル）
ARROW_WIDTH = 3             # 方向矢印の太さ（ピクセル）
TRAINING_GIF_DURATION = 200 # 学習過程GIFのフレーム間隔（ミリ秒）
OPTIMAL_GIF_DURATION = 100  # 最適パターンGIFのフレーム間隔（ミリ秒）
```

## 使用方法

### 1. パラメータの確認

```bash
cd project_02
python hyperparameters.py
```

### 2. パラメータの変更

`hyperparameters.py`ファイルを編集してパラメータを変更：

```python
# 例：学習率を変更
LEARNING_RATE = 0.05  # 0.1 → 0.05に変更

# 例：エピソード数を変更
N_EPISODES = 2000  # 1000 → 2000に変更

# 例：エージェント速度を変更
AGENT_SPEED = 10  # 5 → 10に変更
```

### 3. 動的なパラメータ変更

プログラム内で動的にパラメータを変更：

```python
from hyperparameters import set_learning_parameters, set_environment_parameters

# 学習パラメータを変更
set_learning_parameters(lr=0.05, episodes=2000)

# 環境パラメータを変更
set_environment_parameters(map_size=600, speed=10)
```

## パラメータの影響

### 学習速度への影響

| パラメータ | 値を上げると | 値を下げると |
|-----------|-------------|-------------|
| `LEARNING_RATE` | 学習が速い（不安定） | 学習が遅い（安定） |
| `EPSILON_DECAY` | 探索期間が短い | 探索期間が長い |
| `N_EPISODES` | 学習時間が長い | 学習時間が短い |

### エージェントの動きへの影響

| パラメータ | 値を上げると | 値を下げると |
|-----------|-------------|-------------|
| `AGENT_SPEED` | 動きが速い | 動きが遅い |
| `MAX_STEPS_PER_EPISODE` | 1エピソードが長い | 1エピソードが短い |
| `MAP_SIZE` | マップが大きい | マップが小さい |

### 報酬システムへの影響

| パラメータ | 値を上げると | 値を下げると |
|-----------|-------------|-------------|
| `NEAR_LIGHT_REWARD` | 照明を点灯しやすくなる | 照明を点灯しにくくなる |
| `NEAR_DARK_PENALTY` | 暗い状態を避けやすくなる | 暗い状態を避けにくくなる |
| `ENERGY_COST_PER_LIGHT` | 省エネを重視する | 省エネを軽視する |

## 推奨設定

### 高速学習（テスト用）

```python
N_EPISODES = 100
LEARNING_RATE = 0.2
EPSILON_DECAY = 0.99
MAX_STEPS_PER_EPISODE = 100
```

### 安定学習（本格運用）

```python
N_EPISODES = 2000
LEARNING_RATE = 0.05
EPSILON_DECAY = 0.999
MAX_STEPS_PER_EPISODE = 500
```

### 高速移動（動き重視）

```python
AGENT_SPEED = 10
MAP_SIZE = 800
ZONE_SIZE = 150
```

### 精密移動（精度重視）

```python
AGENT_SPEED = 2
MAP_SIZE = 300
ZONE_SIZE = 50
```

## パラメータ検証

`hyperparameters.py`を実行すると、パラメータの妥当性が自動検証されます：

```bash
python hyperparameters.py
```

エラーがある場合は、修正が必要なパラメータが表示されます。

## トラブルシューティング

### Q: 学習が収束しない

- `LEARNING_RATE`を下げる（0.1 → 0.05）
- `EPSILON_DECAY`を上げる（0.995 → 0.999）
- `N_EPISODES`を増やす（1000 → 2000）

### Q: 学習が遅い

- `LEARNING_RATE`を上げる（0.1 → 0.2）
- `EPSILON_DECAY`を下げる（0.995 → 0.99）
- `MAX_STEPS_PER_EPISODE`を減らす（200 → 100）

### Q: エージェントの動きが不自然

- `AGENT_SPEED`を調整（5 → 3 または 10）
- `MAP_SIZE`と`ZONE_SIZE`の比率を調整

### Q: メモリ不足

- `MAX_STEPS_PER_EPISODE`を減らす
- `SAVE_TRAINING_FRAMES = False`に設定
- `SAVE_OPTIMAL_FRAMES = False`に設定

## カスタマイズ例

### 例1: より大きなマップで学習

```python
MAP_SIZE = 800
ZONE_SIZE = 150
AGENT_SPEED = 8
MAX_STEPS_PER_EPISODE = 300
```

### 例2: より長い学習

```python
N_EPISODES = 5000
LEARNING_RATE = 0.05
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.005
```

### 例3: 省エネ重視

```python
ENERGY_COST_PER_LIGHT = 2.0
NEAR_LIGHT_REWARD = 15.0
FAR_LIGHT_PENALTY = -10.0
```

## 注意事項

1. **パラメータ変更後は必ず検証を実行**
2. **大幅な変更は段階的に行う**
3. **学習結果はパラメータに大きく依存する**
4. **メモリ使用量に注意**

## まとめ

`hyperparameters.py`により、すべてのパラメータを一箇所で管理でき、実験や調整が格段に楽になります。パラメータの影響を理解して、目的に応じた最適な設定を見つけてください！
