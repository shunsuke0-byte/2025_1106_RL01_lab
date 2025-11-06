import math
import random
from typing import Dict, Tuple, Set, List
from abc import ABC, abstractmethod


class RoomShapeBase(ABC):
    """
    部屋の形状を定義する基底クラス
    
    このクラスを継承して、_define_light_zones()と_define_boundaries()を
    実装することで、新しい部屋の形状を簡単に定義できます。
    
    使用例:
        class MyCustomRoom(RoomShapeBase):
            def _define_light_zones(self):
                return {
                    'zone1': {'x_range': (0, 100), 'y_range': (0, 100), 'center': (50, 50)},
                    'zone2': {'x_range': (400, 500), 'y_range': (0, 100), 'center': (450, 50)},
                }
            
            def _define_boundaries(self):
                return {'x_min': 0.0, 'x_max': 500.0, 'y_min': 0.0, 'y_max': 500.0}
    """
    
    def __init__(self, map_size: int = 500, zone_size: int = 100):
        """
        部屋の形状を初期化
        
        Args:
            map_size: マップのサイズ（ピクセル）
            zone_size: 照明ゾーンのサイズ（ピクセル）
        """
        self.map_size = map_size
        self.zone_size = zone_size
        
        # 照明エリア定義（サブクラスで実装）
        self.light_zones = self._define_light_zones()
        
        # 壁の境界定義（サブクラスで実装）
        self.boundaries = self._define_boundaries()
        
        # 内部壁の定義（オプション、サブクラスで実装可能）
        self.walls = self._define_walls() if hasattr(self, '_define_walls') else []
    
    @abstractmethod
    def _define_light_zones(self) -> Dict[str, Dict]:
        """
        照明ゾーンの配置を定義（サブクラスで実装必須）
        
        Returns:
            Dict: 照明ゾーンの情報辞書
            例: {
                'zone_name': {
                    'x_range': (x_min, x_max),
                    'y_range': (y_min, y_max),
                    'center': (center_x, center_y)
                }
            }
        """
        pass
    
    @abstractmethod
    def _define_boundaries(self) -> Dict[str, float]:
        """
        部屋の境界を定義（サブクラスで実装必須）
        
        Returns:
            Dict: 境界の情報辞書
            例: {'x_min': 0.0, 'x_max': 500.0, 'y_min': 0.0, 'y_max': 500.0}
        """
        pass
    
    def get_light_zone(self, zone_name: str) -> Dict:
        """
        指定された照明ゾーンの情報を取得
        
        Args:
            zone_name: ゾーン名
            
        Returns:
            Dict: ゾーン情報
        """
        return self.light_zones.get(zone_name, {})
    
    def get_all_light_zones(self) -> Dict[str, Dict]:
        """
        全ての照明ゾーンの情報を取得
        
        Returns:
            Dict: 全照明ゾーンの情報辞書
        """
        return self.light_zones.copy()
    
    def get_zone_names(self) -> List[str]:
        """
        照明ゾーン名のリストを取得
        
        Returns:
            List[str]: ゾーン名のリスト
        """
        return list(self.light_zones.keys())
    
    def is_position_in_zone(self, x: float, y: float, zone_name: str) -> bool:
        """
        指定位置が照明ゾーン内にあるかチェック
        
        Args:
            x: X座標
            y: Y座標
            zone_name: ゾーン名
            
        Returns:
            bool: ゾーン内にいればTrue
        """
        if zone_name not in self.light_zones:
            return False
            
        zone = self.light_zones[zone_name]
        x_in = zone['x_range'][0] <= x < zone['x_range'][1]
        y_in = zone['y_range'][0] <= y < zone['y_range'][1]
        return x_in and y_in
    
    def get_distance_to_zone(self, x: float, y: float, zone_name: str) -> float:
        """
        指定位置から照明ゾーンの中心までの距離を計算
        
        Args:
            x: X座標
            y: Y座標
            zone_name: ゾーン名
            
        Returns:
            float: ユークリッド距離
        """
        if zone_name not in self.light_zones:
            return float('inf')
            
        center_x, center_y = self.light_zones[zone_name]['center']
        return math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    def _define_walls(self) -> List[Dict]:
        """
        壁を定義（オプション、サブクラスで実装可能）
        
        Returns:
            List[Dict]: 壁のリスト
            例: [
                {'x_range': (200, 300), 'y_range': (100, 150)},  # 横長の壁
                {'x_range': (400, 450), 'y_range': (200, 400)},  # 縦長の壁
            ]
        """
        return []  # デフォルトでは壁なし
    
    def is_position_in_wall(self, x: float, y: float) -> bool:
        """
        指定位置が壁の中にあるかチェック
        
        Args:
            x: X座標
            y: Y座標
            
        Returns:
            bool: 壁の中にいればTrue
        """
        for wall in self.walls:
            x_in = wall['x_range'][0] <= x < wall['x_range'][1]
            y_in = wall['y_range'][0] <= y < wall['y_range'][1]
            if x_in and y_in:
                return True
        return False
    
    def check_wall_collision(self, x: float, y: float, 
                            next_x: float, next_y: float) -> Tuple[bool, str]:
        """
        壁との衝突をチェック
        
        Args:
            x: 現在のX座標
            y: 現在のY座標
            next_x: 次のX座標
            next_y: 次のY座標
        
        Returns:
            Tuple[bool, str]: (衝突したか, 衝突した壁の方向)
        """
        for wall in self.walls:
            x_min, x_max = wall['x_range']
            y_min, y_max = wall['y_range']
            
            # 壁の境界内にいるかチェック
            if x_min <= next_x < x_max and y_min <= next_y < y_max:
                # どの方向から衝突したか判定
                if x < x_min:  # 左から
                    return True, 'left'
                elif x >= x_max:  # 右から
                    return True, 'right'
                elif y < y_min:  # 上から
                    return True, 'top'
                elif y >= y_max:  # 下から
                    return True, 'bottom'
                else:  # 既に壁の中にいる場合、最も近い方向を判定
                    dist_left = abs(x - x_min)
                    dist_right = abs(x - x_max)
                    dist_top = abs(y - y_min)
                    dist_bottom = abs(y - y_max)
                    min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
                    if min_dist == dist_left:
                        return True, 'left'
                    elif min_dist == dist_right:
                        return True, 'right'
                    elif min_dist == dist_top:
                        return True, 'top'
                    else:
                        return True, 'bottom'
        
        return False, ''
    
    def apply_wall_bounce(self, x: float, y: float, velocity_x: float, velocity_y: float) -> Tuple[float, float, float, float, float]:
        """
        壁での跳ね返り処理を適用（外部境界と内部壁の両方に対応）
        
        Args:
            x: 現在のX座標
            y: 現在のY座標
            velocity_x: X方向の速度
            velocity_y: Y方向の速度
            
        Returns:
            Tuple: (新しいx, 新しいy, 新しいvelocity_x, 新しいvelocity_y, 新しい角度)
        """
        next_x = x + velocity_x
        next_y = y + velocity_y
        new_velocity_x = velocity_x
        new_velocity_y = velocity_y
        angle = math.atan2(velocity_y, velocity_x)
        
        # 外部境界のチェックと跳ね返り（既存の処理）
        if next_x < self.boundaries['x_min']:
            next_x = 2 * self.boundaries['x_min'] - next_x
            new_velocity_x = -new_velocity_x
            angle = math.pi - angle
        elif next_x >= self.boundaries['x_max']:
            next_x = 2 * self.boundaries['x_max'] - next_x
            new_velocity_x = -new_velocity_x
            angle = math.pi - angle
        
        if next_y < self.boundaries['y_min']:
            next_y = 2 * self.boundaries['y_min'] - next_y
            new_velocity_y = -new_velocity_y
            angle = -angle
        elif next_y >= self.boundaries['y_max']:
            next_y = 2 * self.boundaries['y_max'] - next_y
            new_velocity_y = -new_velocity_y
            angle = -angle
        
        # 内部壁のチェックと跳ね返り（追加）
        collided, direction = self.check_wall_collision(x, y, next_x, next_y)
        if collided:
            # 壁の範囲を取得
            for wall in self.walls:
                x_min, x_max = wall['x_range']
                y_min, y_max = wall['y_range']
                
                if x_min <= next_x < x_max and y_min <= next_y < y_max:
                    if direction in ['left', 'right']:
                        new_velocity_x = -new_velocity_x
                        angle = math.pi - angle
                        # 壁の外側に配置
                        if direction == 'left':
                            next_x = x_min - 0.1
                        else:
                            next_x = x_max + 0.1
                    elif direction in ['top', 'bottom']:
                        new_velocity_y = -new_velocity_y
                        angle = -angle
                        # 壁の外側に配置
                        if direction == 'top':
                            next_y = y_min - 0.1
                        else:
                            next_y = y_max + 0.1
                    break
        
        # 角度を正規化（0〜2π）
        angle = angle % (2 * math.pi)
        
        return next_x, next_y, new_velocity_x, new_velocity_y, angle
    
    def is_position_in_bounds(self, x: float, y: float) -> bool:
        """
        指定位置が部屋の境界内にあるかチェック
        
        Args:
            x: X座標
            y: Y座標
            
        Returns:
            bool: 境界内にいればTrue
        """
        return (self.boundaries['x_min'] <= x < self.boundaries['x_max'] and
                self.boundaries['y_min'] <= y < self.boundaries['y_max'])
    
    def get_center_position(self) -> Tuple[float, float]:
        """
        部屋の中心位置を取得
        
        Returns:
            Tuple[float, float]: (x, y) 座標
        """
        return (self.map_size // 2, self.map_size // 2)
    
    def get_random_position(self) -> Tuple[float, float]:
        """
        部屋内のランダムな位置を取得
        
        Returns:
            Tuple[float, float]: (x, y) 座標
        """
        x = random.uniform(0, self.map_size)
        y = random.uniform(0, self.map_size)
        return (x, y)
    
    def get_random_angle(self) -> float:
        """
        ランダムな角度を取得
        
        Returns:
            float: ラジアン単位の角度
        """
        return random.uniform(0, 2 * math.pi)
    
    def get_room_info(self) -> Dict:
        """
        部屋の情報を取得
        
        Returns:
            Dict: 部屋の情報辞書
        """
        return {
            'map_size': self.map_size,
            'zone_size': self.zone_size,
            'num_light_zones': len(self.light_zones),
            'boundaries': self.boundaries,
            'light_zones': self.light_zones
        }
    
    def validate_position(self, x: float, y: float) -> Tuple[float, float]:
        """
        位置を部屋の境界内に制限
        
        Args:
            x: X座標
            y: Y座標
            
        Returns:
            Tuple[float, float]: 制限された(x, y)座標
        """
        x = max(self.boundaries['x_min'], min(x, self.boundaries['x_max'] - 0.1))
        y = max(self.boundaries['y_min'], min(y, self.boundaries['y_max'] - 0.1))
        return x, y


# ============================================================
# 具体的な部屋形状の実装例
# ============================================================

class SquareCornersRoom(RoomShapeBase):
    """
    四隅に照明がある正方形の部屋（デフォルト）
    
    500×500ピクセルの正方形の部屋で、四隅に照明ゾーンを配置します。
    """
    
    def _define_light_zones(self) -> Dict[str, Dict]:
        """
        四隅に照明ゾーンを配置
        """
        return {
            'top_left': {
                'x_range': (0, self.zone_size),
                'y_range': (0, self.zone_size),
                'center': (self.zone_size // 2, self.zone_size // 2)
            },
            'top_right': {
                'x_range': (self.map_size - self.zone_size, self.map_size),
                'y_range': (0, self.zone_size),
                'center': (self.map_size - self.zone_size // 2, self.zone_size // 2)
            },
            'bottom_left': {
                'x_range': (0, self.zone_size),
                'y_range': (self.map_size - self.zone_size, self.map_size),
                'center': (self.zone_size // 2, self.map_size - self.zone_size // 2)
            },
            'bottom_right': {
                'x_range': (self.map_size - self.zone_size, self.map_size),
                'y_range': (self.map_size - self.zone_size, self.map_size),
                'center': (self.map_size - self.zone_size // 2, self.map_size - self.zone_size // 2)
            },
        }
    
    def _define_boundaries(self) -> Dict[str, float]:
        """
        正方形の境界を定義
        """
        return {
            'x_min': 0.0,
            'x_max': float(self.map_size),
            'y_min': 0.0,
            'y_max': float(self.map_size)
        }


# デフォルトのRoomShapeクラス（後方互換性のため）
RoomShape = SquareCornersRoom
