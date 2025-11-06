# project_04 使用ガイド

### 主な変更点
- ✅ **10×10グリッド** → **500×500ピクセル**の連続空間
- ✅ **ランダム移動** → **まっすぐ進んで壁で跳ね返る**物理的な動き
- ✅ **離散的な移動** → **5ピクセル/ステップ**の滑らかな移動
- ✅ **3×3照明ゾーン** → **100×100ピクセル照明ゾーン**

## セットアップ

### 1. 必要なパッケージのインストール

```bash
cd project_02
pip install -r requirements.txt
```

または個別にインストール：

```bash
pip install numpy Pillow matplotlib
```

### 2. 動作確認（パッケージなし）

ロジックが正しく動作することは既にテスト済みです：
- ✅ 壁での跳ね返りロジック
- ✅ 状態の離散化
- ✅ 照明ゾーン検出
- ✅ 行動→照明変換

## 実行方法

### 学習の実行

```bash
cd project_02
python room_lighting_optimization.py
```

**学習時間**: 約5〜10分（エピソード数: 1000）

**生成されるファイル**:
- `room_lighting_q_table.npy` - 学習済みQ-table
- `room_lighting_config.json` - 環境設定
- `room_lighting_training.gif` - 学習過程の動画
- `room_lighting_optimal.gif` - 最適パターンの動画

### 可視化の実行

```bash
cd project_02
python visualize_q_learning.py
```

**インタラクティブ機能**:
- マウスを動かすと各位置での最適照明パターンを表示
- クリックすると詳細情報をコンソールに出力

## プログラムの特徴

### 1. 物理的な移動システム

```python
# エージェントはまっすぐ進む
self.agent_x += self.velocity_x
self.agent_y += self.velocity_y

# 壁で物理的に跳ね返る
if next_x < 0:
    next_x = -next_x
    self.velocity_x = -self.velocity_x
    self.agent_angle = math.pi - self.agent_angle
```

### 2. 連続空間での動作

- **移動**: ピクセル単位の滑らかな動き
- **状態**: 10×10グリッドに離散化（Q-tableのサイズを抑制）
- **軌跡**: ランダムウォークではなく、物理法則に従った直線運動

### 3. ビジュアル表現

**学習過程の動画（training.gif）**:
- 緑の円: エージェント
- 赤い矢印: 移動方向
- 黄色: 点灯中の照明
- グレー: 消灯中の照明

**最適パターンの動画（optimal.gif）**:
- 学習後の最適な照明制御パターン
- 200ステップの完全な軌跡

## パラメータ調整

`room_lighting_optimization.py`の`main()`関数で以下を変更可能：

```python
# 環境パラメータ
env = RoomLightingEnvironment(
    map_size=500,      # マップサイズ（ピクセル）
    agent_speed=5,     # 移動速度（ピクセル/ステップ）
    zone_size=100      # 照明ゾーンサイズ（ピクセル）
)

# Q-Learningパラメータ
agent = QLearningAgent(
    n_states=100,            # 状態数
    n_actions=16,            # 行動数
    learning_rate=0.1,       # 学習率
    discount_factor=0.95,    # 割引率
    epsilon=1.0,             # 初期ε値
    epsilon_decay=0.995,     # ε減衰率
    epsilon_min=0.01         # 最小ε値
)

# 学習設定
train_agent(env, agent, n_episodes=1000)  # エピソード数
```

## トラブルシューティング

### Q: パッケージがインストールできない

```bash
# Python 3のpipを使用
pip3 install -r requirements.txt

# または、ユーザー環境にインストール
pip install --user -r requirements.txt
```

### Q: 学習が遅い

- `n_episodes`を減らす（1000 → 500）
- `display_interval`を増やす（50 → 100）

### Q: 動画が生成されない

- `Pillow`パッケージが正しくインストールされているか確認
- ディスク容量を確認

### Q: 可視化が表示されない

- `matplotlib`が正しくインストールされているか確認
- バックエンドの設定を確認

## 出力ファイルの説明

### room_lighting_q_table.npy
- 形状: (100, 16)
- 100状態 × 16行動のQ値テーブル
- numpy形式で保存

### room_lighting_config.json
```json
{
  "map_size": 500,
  "agent_speed": 5,
  "zone_size": 100,
  "max_steps": 200,
  "num_light_zones": 4
}
```

### room_lighting_training.gif
- 最初の3エピソードの学習過程
- 各エピソードの最初の10ステップ
- フレーム間隔: 200ms

### room_lighting_optimal.gif
- 学習後の最適パターン
- 200ステップの完全な軌跡
- フレーム間隔: 100ms

## project_01との比較実行

両方のプロジェクトを実行して違いを確認できます：

```bash
# project_01（グリッド版）
cd project_01
python room_lighting_optimization.py

# project_02（ピクセル版）
cd project_02
python room_lighting_optimization.py
```

**動画を比較**:
- `project_01/room_lighting_optimal.gif` - 離散的な動き
- `project_02/room_lighting_optimal.gif` - 連続的な動き

## カスタマイズ例

### 1. より速い移動速度

```python
env = RoomLightingEnvironment(
    map_size=500,
    agent_speed=10,  # 5 → 10に変更
    zone_size=100
)
```

### 2. より大きな照明ゾーン

```python
env = RoomLightingEnvironment(
    map_size=500,
    agent_speed=5,
    zone_size=150  # 100 → 150に変更
)
```

### 3. より長いエピソード

```python
# RoomLightingEnvironment クラス内
self.max_steps = 500  # 200 → 500に変更
```

## 今後の拡張アイデア

1. **障害物の追加**: 部屋の中央に机などの障害物
2. **複数エージェント**: 2人以上が同時に移動
3. **加速度の導入**: より物理的な動き
4. **異なる部屋形状**: 長方形、L字型など
5. **時間帯の考慮**: 昼夜で照明の必要性が変わる

## サポート

問題が発生した場合は、以下を確認してください：
- Python 3.7以上がインストールされている
- 必要なパッケージがすべてインストールされている
- ディスク容量が十分にある（動画生成のため）

## ライセンス

このプロジェクトは教育目的で作成されています。

