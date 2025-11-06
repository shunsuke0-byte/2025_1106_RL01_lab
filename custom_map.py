"""
カスタムマップ: 500×1000の縦長の部屋に8つの照明
"""

from room_shape import RoomShapeBase
from typing import Dict


class CustomMap500x1000(RoomShapeBase):
    """
    1000×500の横長の部屋
    
    照明配置:
    - light_1: (600, 0)-(800, 200)
    - light_2: (800, 0)-(1000, 200)
    - light_3: (900, 100)-(1000, 400)
    - light_4: (600, 400)-(800, 500)
    - light_5: (800, 400)-(1000, 500)
    - light_6: (0, 0)-(600, 500)  # 大きな長方形の照明ゾーン
    - light_7: (600, 200)-(800, 400)  # 追加
    - light_8: (800, 200)-(900, 400)  # 追加
    """
    
    def __init__(self, map_size: int = 500, zone_size: int = 100):
        # 横長の部屋: 幅1000 × 高さ500
        self.map_width = 1000
        self.map_height = 500
        super().__init__(map_size, zone_size)
    
    def _define_light_zones(self) -> Dict[str, Dict]:
        """
        8つの照明ゾーンを指定位置に配置
        """
        return {
            'light_1': {
                'x_range': (600, 800),
                'y_range': (0, 200),
                'center': (700, 100)
            },
            'light_2': {
                'x_range': (800, 1000),
                'y_range': (0, 200),
                'center': (900, 100)
            },
            'light_3': {
                'x_range': (900, 1000),
                'y_range': (200, 400),
                'center': (950, 250)
            },
            'light_4': {
                'x_range': (600, 800),
                'y_range': (400, 500),
                'center': (700, 450)
            },
            'light_5': {
                'x_range': (800, 1000),
                'y_range': (400, 500),
                'center': (900, 450)
            },
            'light_6': {  # 大きな長方形の照明ゾーン
                'x_range': (0, 600),
                'y_range': (0, 500),
                'center': (300, 250)
            },
            'light_7': {  # 追加: (600, 200)-(800, 400)
                'x_range': (600, 800),
                'y_range': (200, 400),
                'center': (700, 300)
            },
            'light_8': {  # 追加: (800, 200)-(900, 400)
                'x_range': (800, 900),
                'y_range': (200, 400),
                'center': (850, 300)
            },
        }
    
    def _define_boundaries(self) -> Dict[str, float]:
        """
        部屋の境界を定義（幅1000 × 高さ500）
        """
        return {
            'x_min': 0.0,
            'x_max': 1000.0,
            'y_min': 0.0,
            'y_max': 500.0
        }
    
    def get_center_position(self):
        """部屋の中心位置を返す"""
        return (self.map_width // 2, self.map_height // 2)
    
    def get_random_position(self):
        """部屋内のランダムな位置を返す"""
        import random
        x = random.uniform(0, self.map_width)
        y = random.uniform(0, self.map_height)
        return (x, y)
    
    def _define_walls(self):
        """
        内部壁を定義
        - 壁1: x=600, y=0-200 (縦の壁)
        - 壁2: x=800, y=0-200 (縦の壁)
        - 壁3: x=800, y=400-500 (縦の壁)
        """
        wall_width = 10  # 壁の幅（ピクセル）
        return [
            # 壁1: x=600, y=0-200
            {'x_range': (600 - wall_width // 2, 600 + wall_width // 2), 'y_range': (0, 200)},
            # 壁2: x=800, y=0-200
            {'x_range': (800 - wall_width // 2, 800 + wall_width // 2), 'y_range': (0, 200)},
            # 壁3: x=800, y=400-500
            {'x_range': (800 - wall_width // 2, 800 + wall_width // 2), 'y_range': (400, 500)},
        ]

