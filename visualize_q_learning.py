import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import json
import math

# 日本語フォント設定（追加）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策

class QLearningVisualizer:
    """
    Q-Learning学習結果を視覚化するクラス（ピクセル単位版）
    
    マウスカーソルでマップ上を動かして、各位置でのQ値や最適行動を確認できる
    """
    
    def __init__(self, q_table_path='room_lighting_q_table.npy', 
                 config_path='room_lighting_config.json'):
        """
        初期化
        
        Args:
            q_table_path: Q-tableのファイルパス
            config_path: 設定ファイルのパス
        """
        # Load Q-table
        self.q_table = np.load(q_table_path)
        print(f"Q-table loaded: {self.q_table.shape}")
        
        # 設定を読み込み
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.map_width = config.get('map_width', config.get('map_size', 500))
            self.map_height = config.get('map_height', config.get('map_size', 500))
            self.map_size = max(self.map_width, self.map_height)  # 互換性のため
            self.zone_size = config.get('zone_size', 100)
            self.agent_speed = config.get('agent_speed', 5)
            self.grid_x_size = config.get('grid_x_size', 10)
            self.grid_y_size = config.get('grid_y_size', 10)
            self.n_actions = config.get('n_actions', 16)
            self.num_lights = config.get('num_light_zones', 4)
        except:
            self.map_width = 500
            self.map_height = 500
            self.map_size = 500
            self.zone_size = 100
            self.agent_speed = 5
            self.grid_x_size = 10
            self.grid_y_size = 10
            self.n_actions = 16
            self.num_lights = 4
        
        # グリッドサイズ（状態の離散化用）
        self.cell_x_size = self.map_width / self.grid_x_size
        self.cell_y_size = self.map_height / self.grid_y_size
        
        # 照明エリア定義（カスタムマップ用）
        self.light_zones = self._get_light_zones()
        
        # 行動の説明（照明数に応じて動的に生成）
        self.action_names = self._generate_action_names()
        
        # 現在のカーソル位置
        self.current_pos = None
        
        # 図の設定
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def _generate_action_names(self):
        """Generate action descriptions (dynamic based on number of lights)"""
        names = []
        # 照明ラベル（最大10個まで対応）
        zone_labels = [f'L{i+1}' for i in range(self.num_lights)]
        
        for action in range(self.n_actions):
            # Represent ON/OFF of each light in binary
            lights_on = []
            for i in range(self.num_lights):
                if action & (1 << i):
                    lights_on.append(zone_labels[i])
            
            if lights_on:
                names.append(', '.join(lights_on))
            else:
                names.append('All OFF')
        
        return names
    
    def _get_light_zones(self):
        """
        照明エリアの定義を取得（カスタムマップ対応）
        """
        if self.num_lights == 4 and self.map_width == 500 and self.map_height == 500:
            # デフォルトマップ（四隅）
            return {
                'top_left': {
                    'x_range': (0, self.zone_size),
                    'y_range': (0, self.zone_size),
                },
                'top_right': {
                    'x_range': (self.map_width - self.zone_size, self.map_width),
                    'y_range': (0, self.zone_size),
                },
                'bottom_left': {
                    'x_range': (0, self.zone_size),
                    'y_range': (self.map_height - self.zone_size, self.map_height),
                },
                'bottom_right': {
                    'x_range': (self.map_width - self.zone_size, self.map_width),
                    'y_range': (self.map_height - self.zone_size, self.map_height),
                },
            }
        elif self.num_lights == 5 and self.map_width == 1000 and self.map_height == 500:
            # カスタムマップ（1000×500、5つの照明）
            return {
                'light_1': {
                    'x_range': (600, 800),
                    'y_range': (0, 200),
                },
                'light_2': {
                    'x_range': (800, 1000),
                    'y_range': (0, 200),
                },
                'light_3': {
                    'x_range': (900, 1000),
                    'y_range': (200, 400),
                },
                'light_4': {
                    'x_range': (600, 800),
                    'y_range': (400, 500),
                },
                'light_5': {
                    'x_range': (800, 1000),
                    'y_range': (400, 500),
                },
            }
        elif self.num_lights == 6 and self.map_width == 1000 and self.map_height == 500:
            # カスタムマップ（1000×500、6つの照明）- light_6を追加
            return {
                'light_1': {
                    'x_range': (600, 800),
                    'y_range': (0, 200),
                },
                'light_2': {
                    'x_range': (800, 1000),
                    'y_range': (0, 200),
                },
                'light_3': {
                    'x_range': (900, 1000),
                    'y_range': (200, 400),
                },
                'light_4': {
                    'x_range': (600, 800),
                    'y_range': (400, 500),
                },
                'light_5': {
                    'x_range': (800, 1000),
                    'y_range': (400, 500),
                },
                'light_6': {  # 大きな長方形の照明ゾーン
                    'x_range': (0, 600),
                    'y_range': (0, 500),
                },
            }
        elif self.num_lights == 8 and self.map_width == 1000 and self.map_height == 500:
            # カスタムマップ（1000×500、8つの照明）- light_7, light_8を追加
            return {
                'light_1': {
                    'x_range': (600, 800),
                    'y_range': (0, 200),
                },
                'light_2': {
                    'x_range': (800, 1000),
                    'y_range': (0, 200),
                },
                'light_3': {
                    'x_range': (900, 1000),
                    'y_range': (200, 400),
                },
                'light_4': {
                    'x_range': (600, 800),
                    'y_range': (400, 500),
                },
                'light_5': {
                    'x_range': (800, 1000),
                    'y_range': (400, 500),
                },
                'light_6': {  # 大きな長方形の照明ゾーン
                    'x_range': (0, 600),
                    'y_range': (0, 500),
                },
                'light_7': {  # 追加: (600, 200)-(800, 400)
                    'x_range': (600, 800),
                    'y_range': (200, 400),
                },
                'light_8': {  # 追加: (800, 200)-(900, 400)
                    'x_range': (800, 900),
                    'y_range': (200, 400),
                },
            }
        else:
            # その他のマップ（空の辞書を返す）
            return {}
    
    def _pixel_to_state(self, pixel_x, pixel_y):
        """ピクセル座標を状態番号に変換"""
        grid_x = min(int(pixel_x / self.cell_x_size), self.grid_x_size - 1)
        grid_y = min(int(pixel_y / self.cell_y_size), self.grid_y_size - 1)
        return grid_y * self.grid_x_size + grid_x
    
    def visualize(self):
        """
        Q-Learning学習結果を視覚化
        
        3つのサブプロット:
        1. マップ全体のQ値ヒートマップ（最大Q値）
        2. 各位置での最適行動（照明マップ）
        3. カーソル位置の詳細情報
        """
        # 1. Q値ヒートマップ（最大Q値）
        self._plot_q_value_heatmap()
        
        # 2. 最適行動マップ
        self._plot_optimal_action_map()
        
        # 3. 詳細情報（初期は空）
        self._plot_detail_info(None)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_q_value_heatmap(self):
        """Draw Q-value heatmap"""
        ax = self.axes[0]
        ax.clear()
        ax.set_title('Q-Value Heatmap (Max Q-Value)', fontsize=14, fontweight='bold')
        
        # Calculate max Q-value for each grid position
        max_q_values = np.zeros((self.grid_y_size, self.grid_x_size))
        for i in range(self.grid_y_size):
            for j in range(self.grid_x_size):
                state = i * self.grid_x_size + j
                if state < self.q_table.shape[0]:
                    max_q_values[i, j] = np.max(self.q_table[state])
        
        # Draw heatmap
        extent = [0, self.map_width, self.map_height, 0]
        im = ax.imshow(max_q_values, cmap='YlOrRd', interpolation='nearest', extent=extent, aspect='auto')
        
        # Display light zones with frames
        if self.light_zones:
            colors = ['red', 'green', 'blue', 'magenta', 'orange', 'cyan', 'yellow', 'purple', 'brown', 'pink']
            for i, (zone_name, zone_info) in enumerate(self.light_zones.items()):
                x1, x2 = zone_info['x_range']
                y1, y2 = zone_info['y_range']
                color = colors[i % len(colors)]
                rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
        
        # Grid lines
        for i in range(self.grid_x_size + 1):
            pos = i * self.cell_x_size
            ax.axvline(x=pos, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        for i in range(self.grid_y_size + 1):
            pos = i * self.cell_y_size
            ax.axhline(y=pos, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Colorbar
        plt.colorbar(im, ax=ax, label='Max Q-Value')
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_xlim(0, self.map_width)
        ax.set_ylim(self.map_height, 0)
    
    def _plot_optimal_action_map(self):
        """Draw optimal action map with light zones highlighted"""
        ax = self.axes[1]
        ax.clear()
        ax.set_title('Light Status Map (Optimal Action)', fontsize=14, fontweight='bold')
        
        # Create base map (all black/off)
        light_map = np.zeros((int(self.map_height), int(self.map_width)))
        
        # For visualization, show lights for current cursor position
        if self.current_pos is not None:
            pixel_x, pixel_y = self.current_pos
            state = self._pixel_to_state(pixel_x, pixel_y)
            if state < self.q_table.shape[0]:
                optimal_action = np.argmax(self.q_table[state])
                
                # Decode which lights are ON (照明数に応じて動的)
                zone_names = list(self.light_zones.keys())
                for i in range(min(self.num_lights, len(zone_names))):
                    if optimal_action & (1 << i):  # Check if bit is set
                        zone_name = zone_names[i]
                        if zone_name in self.light_zones:
                            zone = self.light_zones[zone_name]
                            x1, x2 = zone['x_range']
                            y1, y2 = zone['y_range']
                            # Mark this zone as lit
                            light_map[int(y1):int(y2), int(x1):int(x2)] = 1
        
        # Draw light map (0=off/black, 1=on/yellow)
        im = ax.imshow(light_map, cmap='binary_r', interpolation='nearest', vmin=0, vmax=1, 
                      extent=[0, self.map_width, self.map_height, 0])
        
        # Display light zones with colored frames
        colors = ['red', 'green', 'blue', 'magenta', 'orange', 'cyan', 'yellow', 'purple', 'brown', 'pink']
        for i, (zone_name, zone_info) in enumerate(self.light_zones.items()):
            x1, x2 = zone_info['x_range']
            y1, y2 = zone_info['y_range']
            color = colors[i % len(colors)]
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        
        # Highlight current agent position
        if self.current_pos is not None:
            pixel_x, pixel_y = self.current_pos
            circle = Circle((pixel_x, pixel_y), 15, 
                          linewidth=3, edgecolor='cyan', facecolor='cyan', alpha=0.7)
            ax.add_patch(circle)
        
        # Grid lines
        for i in range(self.grid_x_size + 1):
            pos = i * self.cell_x_size
            ax.axvline(x=pos, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        for i in range(self.grid_y_size + 1):
            pos = i * self.cell_y_size
            ax.axhline(y=pos, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Legend
        legend_text = f'{self.num_lights} lights: L1-L{self.num_lights}\nYellow: Light ON | Black: Light OFF'
        ax.text(0.5, -0.15, legend_text, ha='center', va='top', 
               fontsize=9, transform=ax.transAxes)
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_xlim(0, self.map_width)
        ax.set_ylim(self.map_height, 0)
    
    def _plot_detail_info(self, pos):
        """Draw detailed information"""
        ax = self.axes[2]
        ax.clear()
        ax.axis('off')
        
        if pos is None:
            ax.text(0.5, 0.5, 'Move cursor over\nthe map', 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            return
        
        pixel_x, pixel_y = pos
        state = self._pixel_to_state(pixel_x, pixel_y)
        q_values = self.q_table[state]
        
        # Title
        title_text = f'Position: ({pixel_x:.0f}, {pixel_y:.0f}) px | State: {state}\n'
        ax.text(0.5, 0.95, title_text, ha='center', va='top', fontsize=14, 
               fontweight='bold', transform=ax.transAxes)
        
        # Q-value details (top 5)
        sorted_indices = np.argsort(q_values)[::-1]
        
        detail_text = 'Q-Value Ranking (Top 5):\n\n'
        for rank, idx in enumerate(sorted_indices[:5], 1):
            action_name = self.action_names[idx]
            q_value = q_values[idx]
            marker = '*' if rank == 1 else f'{rank}.'
            detail_text += f'{marker} Action {idx}: {action_name}\n'
            detail_text += f'   Q-Value: {q_value:.2f}\n\n'
        
        # Statistics
        detail_text += f'\nStatistics:\n'
        detail_text += f'Max Q-Value: {np.max(q_values):.2f}\n'
        detail_text += f'Min Q-Value: {np.min(q_values):.2f}\n'
        detail_text += f'Avg Q-Value: {np.mean(q_values):.2f}\n'
        
        ax.text(0.05, 0.85, detail_text, ha='left', va='top', fontsize=11, 
               family='monospace', transform=ax.transAxes)
    
    def on_mouse_move(self, event):
        """マウス移動イベント"""
        if event.inaxes in [self.axes[0], self.axes[1]]:
            if event.xdata is not None and event.ydata is not None:
                pixel_x = event.xdata
                pixel_y = event.ydata
                
                if 0 <= pixel_x < self.map_size and 0 <= pixel_y < self.map_size:
                    if self.current_pos != (pixel_x, pixel_y):
                        self.current_pos = (pixel_x, pixel_y)
                        
                        # Update all panels
                        self._plot_optimal_action_map()  # Update light status
                        self._plot_detail_info(self.current_pos)
                        
                        # Highlight cursor position on Q-value heatmap
                        ax = self.axes[0]
                        # Remove existing highlights
                        for patch in ax.patches[:]:
                            if hasattr(patch, 'is_highlight'):
                                patch.remove()
                        
                        # Add new highlight (grid cell)
                        grid_x = int(pixel_x / self.cell_x_size)
                        grid_y = int(pixel_y / self.cell_y_size)
                        rect = Rectangle((grid_x * self.cell_x_size, grid_y * self.cell_y_size), 
                                       self.cell_x_size, self.cell_y_size,
                                       linewidth=3, edgecolor='red', 
                                       facecolor='none', linestyle='--')
                        rect.is_highlight = True
                        ax.add_patch(rect)
                        
                        self.fig.canvas.draw_idle()
    
    def on_click(self, event):
        """Mouse click event"""
        if event.inaxes in [self.axes[0], self.axes[1]]:
            if event.xdata is not None and event.ydata is not None:
                pixel_x = event.xdata
                pixel_y = event.ydata
                
                if 0 <= pixel_x < self.map_size and 0 <= pixel_y < self.map_size:
                    state = self._pixel_to_state(pixel_x, pixel_y)
                    optimal_action = np.argmax(self.q_table[state])
                    action_name = self.action_names[optimal_action]
                    
                    print(f"\nPosition ({pixel_x:.0f}, {pixel_y:.0f}) Details:")
                    print(f"  State ID: {state}")
                    print(f"  Optimal Action: {optimal_action} ({action_name})")
                    print(f"  Max Q-Value: {np.max(self.q_table[state]):.2f}")


def main():
    """Main function"""
    print("=" * 60)
    print("Q-Learning Result Visualizer (ピクセル単位版)")
    print("=" * 60)
    print("\nHow to use:")
    print("  - Move mouse over map: Show detailed info for each position")
    print("  - Click mouse: Print detailed info to console")
    print("  - Close window: Exit program")
    print("\nColored frames: Light zones (100x100px)")
    print("Red dashed frame: Current cursor grid cell")
    print("=" * 60)
    
    try:
        visualizer = QLearningVisualizer()
        visualizer.visualize()
    except FileNotFoundError as e:
        print(f"\nError: File not found")
        print(f"  {e}")
        print("\nPlease run room_lighting_optimization.py first to complete training.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

