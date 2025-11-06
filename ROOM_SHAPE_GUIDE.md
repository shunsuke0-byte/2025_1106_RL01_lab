# 部屋形状の定義ガイド

このガイドでは、新しい部屋形状を簡単に定義する方法を説明します。

## 基本的な使い方

### 1. 新しい部屋形状クラスを作成

`RoomShapeBase`を継承して、`_define_light_zones()`と`_define_boundaries()`メソッドを実装するだけです。

```python
from room_shape import RoomShapeBase

class MyCustomRoom(RoomShapeBase):
    """
    カスタム部屋の説明
    """
    
    def _define_light_zones(self):
        """照明ゾーンの配置を定義"""
        return {
            'zone1': {
                'x_range': (0, 100),      # X座標の範囲
                'y_range': (0, 100),      # Y座標の範囲
                'center': (50, 50)        # ゾーンの中心座標
            },
            'zone2': {
                'x_range': (400, 500),
                'y_range': (0, 100),
                'center': (450, 50)
            },
            # 必要なだけ照明ゾーンを追加できます
        }
    
    def _define_boundaries(self):
        """部屋の境界を定義"""
        return {
            'x_min': 0.0,
            'x_max': 500.0,
            'y_min': 0.0,
            'y_max': 500.0
        }
```

### 2. 作成した部屋形状を使用

```python
from room_lighting_optimization import RoomLightingEnvironment
from my_custom_room import MyCustomRoom

# カスタム部屋で環境を作成
env = RoomLightingEnvironment(room_shape_class=MyCustomRoom)

# 学習を実行
# ... (以降は通常通り)
```

## 実装例

### 例1: 長方形の部屋

```python
class RectangularRoom(RoomShapeBase):
    """横長の長方形の部屋"""
    
    def __init__(self, map_size=500, zone_size=100):
        self.map_width = int(map_size * 1.6)  # 幅を1.6倍に
        self.map_height = map_size
        super().__init__(map_size, zone_size)
    
    def _define_light_zones(self):
        return {
            'top_left': {
                'x_range': (0, self.zone_size),
                'y_range': (0, self.zone_size),
                'center': (self.zone_size // 2, self.zone_size // 2)
            },
            'top_right': {
                'x_range': (self.map_width - self.zone_size, self.map_width),
                'y_range': (0, self.zone_size),
                'center': (self.map_width - self.zone_size // 2, self.zone_size // 2)
            },
            # ... 他の照明ゾーン
        }
    
    def _define_boundaries(self):
        return {
            'x_min': 0.0,
            'x_max': float(self.map_width),
            'y_min': 0.0,
            'y_max': float(self.map_height)
        }
    
    # 長方形の場合、これらのメソッドもオーバーライド
    def get_center_position(self):
        return (self.map_width // 2, self.map_height // 2)
    
    def get_random_position(self):
        import random
        x = random.uniform(0, self.map_width)
        y = random.uniform(0, self.map_height)
        return (x, y)
```

### 例2: 中央に1つだけ照明がある部屋

```python
class CenterLightRoom(RoomShapeBase):
    """中央に1つだけ照明がある部屋"""
    
    def _define_light_zones(self):
        center = self.map_size // 2
        half_zone = self.zone_size // 2
        
        return {
            'center': {
                'x_range': (center - half_zone, center + half_zone),
                'y_range': (center - half_zone, center + half_zone),
                'center': (center, center)
            },
        }
    
    def _define_boundaries(self):
        return {
            'x_min': 0.0,
            'x_max': float(self.map_size),
            'y_min': 0.0,
            'y_max': float(self.map_size)
        }
```

### 例3: グリッド状に照明がある部屋

```python
class GridLightsRoom(RoomShapeBase):
    """3×3のグリッド状に9つの照明がある部屋"""
    
    def _define_light_zones(self):
        zones = {}
        grid_size = 3
        cell_size = self.map_size / grid_size
        
        for row in range(grid_size):
            for col in range(grid_size):
                zone_name = f'zone_{row}_{col}'
                x_start = int(col * cell_size + (cell_size - self.zone_size) / 2)
                y_start = int(row * cell_size + (cell_size - self.zone_size) / 2)
                
                zones[zone_name] = {
                    'x_range': (x_start, x_start + self.zone_size),
                    'y_range': (y_start, y_start + self.zone_size),
                    'center': (x_start + self.zone_size // 2, y_start + self.zone_size // 2)
                }
        
        return zones
    
    def _define_boundaries(self):
        return {
            'x_min': 0.0,
            'x_max': float(self.map_size),
            'y_min': 0.0,
            'y_max': float(self.map_size)
        }
```

## 高度な使い方

### 壁の跳ね返り処理をカスタマイズ

複雑な形状の部屋の場合、`apply_wall_bounce()`メソッドをオーバーライドできます。

```python
class ComplexRoom(RoomShapeBase):
    """複雑な形状の部屋"""
    
    def apply_wall_bounce(self, x, y, velocity_x, velocity_y):
        """
        カスタムの壁跳ね返り処理
        
        Returns:
            Tuple: (新しいx, 新しいy, 新しいvelocity_x, 新しいvelocity_y, 新しい角度)
        """
        # ここにカスタムロジックを実装
        # ...
        
        return next_x, next_y, new_velocity_x, new_velocity_y, angle
```

### 他のメソッドもオーバーライド可能

- `get_center_position()`: 部屋の中心位置を返す
- `get_random_position()`: ランダムな位置を返す
- `is_position_in_bounds()`: 位置が境界内かチェック
- `validate_position()`: 位置を境界内に制限

## 利用可能な例

`room_shape_examples.py`に以下の実装例があります：

1. **RectangularRoom**: 横長の長方形の部屋
2. **SixLightsRoom**: 6つの照明がある部屋（四隅 + 上下中央）
3. **CenterLightRoom**: 中央に1つだけ照明がある部屋
4. **WallLightsRoom**: 壁沿いに照明がある部屋
5. **GridLightsRoom**: グリッド状に9つの照明がある部屋

### 例の実行方法

```python
from room_shape_examples import SixLightsRoom
from room_lighting_optimization import RoomLightingEnvironment

# 6つの照明がある部屋で環境を作成
env = RoomLightingEnvironment(room_shape_class=SixLightsRoom)
```

## ファイル構成

```
project_02/
├── room_shape.py              # 基底クラス（RoomShapeBase）とデフォルト実装
├── room_shape_examples.py     # 実装例集
├── room_lighting_optimization.py  # メインプログラム
└── ROOM_SHAPE_GUIDE.md       # このガイド
```

## 注意事項

1. **照明ゾーンの数**: Q-Learningの行動数は照明の組み合わせで決まります（2^照明数）。照明が多すぎると学習が困難になる可能性があります。

2. **座標系**: 左上が(0, 0)で、右下が(map_size, map_size)です。

3. **ゾーンの範囲**: `x_range`と`y_range`は`(min, max)`の形式で、`min`は含み、`max`は含みません。

4. **中心座標**: `center`は照明ゾーンの中心座標で、距離計算に使用されます。

## トラブルシューティング

### Q: 照明が正しく表示されない

A: `_define_light_zones()`の戻り値の形式を確認してください。各ゾーンには`x_range`, `y_range`, `center`が必要です。

### Q: エージェントが壁を突き抜ける

A: `_define_boundaries()`が正しく設定されているか確認してください。また、複雑な形状の場合は`apply_wall_bounce()`のオーバーライドが必要かもしれません。

### Q: 学習が収束しない

A: 照明の数が多すぎる可能性があります。照明数を減らすか、ハイパーパラメータを調整してください。

## さらなる情報

- `room_shape.py`のソースコードを参照
- `room_shape_examples.py`で実装例を確認
- `hyperparameters.py`で学習パラメータを調整

