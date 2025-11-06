import numpy as np  # 数値計算ライブラリ
import time  # 時間制御
import os  # OS操作
import argparse  # コマンドライン引数解析
from PIL import Image, ImageDraw, ImageFont  # 動画保存用
import json  # 設定保存用
import random  # ランダム移動用
import math  # 数学関数
try:
    import matplotlib.pyplot as plt  # 学習曲線の可視化用
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlibがインストールされていません。学習曲線の可視化はスキップされます。")
from hyperparameters import *  # ハイパーパラメータをインポート
from room_shape import RoomShape, RoomShapeBase  # 部屋形状定義クラスをインポート

# ============================================================
# マップ設定 - ここを変更するだけで学習できます
# ============================================================

# 使用するマップのパスを指定（Noneの場合はデフォルトマップ）
MAP_PATH = "custom_map.CustomMap500x1000"  # カスタムマップを使用

# 他のマップを使用する場合の例:
# MAP_PATH = None  # デフォルトマップ（四隅に照明）

# ============================================================

def load_map_class(map_path):
    """
    マップパスからクラスを動的に読み込む
    """
    if map_path is None:
        return RoomShape  # デフォルトマップ
    
    try:
        # パスを分割（例: "custom_map.CustomMap500x1000" -> "custom_map", "CustomMap500x1000"）
        module_name, class_name = map_path.rsplit('.', 1)
        
        # モジュールを動的にインポート
        module = __import__(module_name, fromlist=[class_name])
        map_class = getattr(module, class_name)
        
        print(f"✓ マップを読み込みました: {map_path}")
        return map_class
        
    except Exception as e:
        print(f"✗ マップの読み込みに失敗しました: {map_path}")
        print(f"  エラー: {e}")
        print(f"  デフォルトマップを使用します")
        return RoomShape

# ============================================================

class RoomLightingEnvironment:
    """
    部屋照明最適化環境クラス（ピクセル単位版）
    
    500×500ピクセルの部屋の四隅に照明があり、エージェントが近づいたら照明を点灯する
    エージェントはまっすぐ進み、壁で綺麗に跳ね返る
    """
    
    def __init__(self, room_shape_class=None, map_size=MAP_SIZE, agent_speed=AGENT_SPEED, zone_size=ZONE_SIZE):
        """
        環境の初期化
        
        Args:
            room_shape_class: 部屋形状クラス（RoomShapeBaseを継承したクラス）
                            Noneの場合はデフォルトのRoomShapeを使用
            map_size: マップのサイズ（ピクセル）- カスタムマップでは無視される
            agent_speed: エージェントの移動速度（ピクセル/ステップ）
            zone_size: 照明ゾーンのサイズ（ピクセル）
        """
        self.agent_speed = agent_speed
        self.zone_size = zone_size
        
        # 部屋形状の定義（追加）
        if room_shape_class is None:
            self.room_shape = RoomShape(map_size, zone_size)
        else:
            self.room_shape = room_shape_class(map_size, zone_size)
        
        # 実際のマップサイズをroom_shapeから取得（カスタムマップ対応）
        boundaries = self.room_shape.boundaries
        self.map_width = boundaries['x_max'] - boundaries['x_min']
        self.map_height = boundaries['y_max'] - boundaries['y_min']
        # 互換性のため、最大サイズをmap_sizeとして保持
        self.map_size = max(self.map_width, self.map_height)
        
        # 照明数を取得
        self.num_lights = len(self.room_shape.get_zone_names())
        
        # 状態数を動的に計算（10×10グリッドを基準に、マップサイズに応じて調整）
        # ただし、状態数は100-400の範囲に制限（学習効率のため）
        base_grid_size = 10
        if self.map_size <= 500:
            self.grid_size = base_grid_size
        elif self.map_size <= 1000:
            self.grid_size = 15  # 1000×500の場合、15×7 = 105状態
        else:
            self.grid_size = 20
        
        # 実際のグリッドサイズを計算（横長・縦長に対応）
        if self.map_width > self.map_height:
            self.grid_x_size = self.grid_size
            self.grid_y_size = max(7, int(self.grid_size * self.map_height / self.map_width))
        elif self.map_height > self.map_width:
            self.grid_x_size = max(7, int(self.grid_size * self.map_width / self.map_height))
            self.grid_y_size = self.grid_size
        else:
            self.grid_x_size = self.grid_size
            self.grid_y_size = self.grid_size
        
        self.n_states = self.grid_x_size * self.grid_y_size
        
        # 行動数を動的に計算（2^照明数）
        self.n_actions = 2 ** self.num_lights
        
        # エージェントの初期位置（部屋の中央・ピクセル座標）
        center_x, center_y = self.room_shape.get_center_position()
        self.agent_x = float(center_x)
        self.agent_y = float(center_y)
        
        # エージェントの移動方向（ランダムな角度で初期化）
        self.agent_angle = self.room_shape.get_random_angle()  # ラジアン
        
        # 移動速度の成分
        self.velocity_x = agent_speed * math.cos(self.agent_angle)
        self.velocity_y = agent_speed * math.sin(self.agent_angle)
        
        # 現在点灯している照明（ゾーン名のセット）
        self.active_lights = set()
        
        # ステップカウンター
        self.step_count = 0
        self.max_steps = MAX_STEPS_PER_EPISODE  # 1エピソードの最大ステップ数
        
    def _get_distance_to_zone(self, zone_name):
        """
        エージェントから指定ゾーンの中心までの距離を計算
        
        Args:
            zone_name: ゾーン名
        
        Returns:
            float: ユークリッド距離
        """
        return self.room_shape.get_distance_to_zone(self.agent_x, self.agent_y, zone_name)
    
    def _is_in_zone(self, zone_name):
        """
        エージェントが指定ゾーン内にいるかチェック
        
        Args:
            zone_name: ゾーン名
        
        Returns:
            bool: ゾーン内にいればTrue
        """
        return self.room_shape.is_position_in_zone(self.agent_x, self.agent_y, zone_name)
    
    def reset(self):
        """
        環境をリセット
        
        Returns:
            int: 初期状態番号
        """
        center_x, center_y = self.room_shape.get_center_position()
        self.agent_x = float(center_x)
        self.agent_y = float(center_y)
        self.agent_angle = self.room_shape.get_random_angle()
        self.velocity_x = self.agent_speed * math.cos(self.agent_angle)
        self.velocity_y = self.agent_speed * math.sin(self.agent_angle)
        self.active_lights = set()
        self.step_count = 0
        return self._get_state()
    
    def _get_state(self):
        """
        現在の状態を返す
        
        状態 = エージェントの位置を離散化したグリッド座標
        （マップサイズに応じて動的にグリッドサイズを調整）
        
        Returns:
            int: 状態番号
        """
        cell_x_size = self.map_width / self.grid_x_size
        cell_y_size = self.map_height / self.grid_y_size
        
        grid_x = min(int(self.agent_x / cell_x_size), self.grid_x_size - 1)
        grid_y = min(int(self.agent_y / cell_y_size), self.grid_y_size - 1)
        
        return grid_y * self.grid_x_size + grid_x
    
    def step(self, action):
        """
        行動を実行して次の状態、報酬、終了フラグを返す
        
        Args:
            action: 照明配置パターンの番号（0から2^照明数-1まで）
                   NビットでN個の照明のON/OFFを表現
        
        Returns:
            next_state (int): 次の状態番号
            reward (float): 報酬
            done (bool): エピソード終了フラグ
        """
        # 行動から照明配置を決定
        self.active_lights = self._action_to_lights(action)
        
        # エージェントを移動（壁で跳ね返り）追加
        self._move_agent()
        
        # 報酬を計算
        reward = self._calculate_reward()
        
        # ステップカウント
        self.step_count += 1
        
        # エピソード終了判定
        done = self.step_count >= self.max_steps
        
        return self._get_state(), reward, done
    
    def _move_agent(self):
        """
        エージェントを移動させ、壁で跳ね返る処理を行う（追加）
        """
        # 部屋形状クラスを使用して壁での跳ね返り処理を適用
        next_x, next_y, new_velocity_x, new_velocity_y, new_angle = self.room_shape.apply_wall_bounce(
            self.agent_x, self.agent_y, self.velocity_x, self.velocity_y
        )
        
        # 位置と速度を更新
        self.agent_x = next_x
        self.agent_y = next_y
        self.velocity_x = new_velocity_x
        self.velocity_y = new_velocity_y
        self.agent_angle = new_angle
    
    def _action_to_lights(self, action):
        """
        行動番号から照明配置に変換
        
        Args:
            action: 行動番号（0-15）
                   ビット0: 左上照明
                   ビット1: 右上照明
                   ビット2: 左下照明
                   ビット3: 右下照明
        
        Returns:
            set: 点灯する照明ゾーン名のセット
        """
        active = set()
        zone_names = self.room_shape.get_zone_names()
        
        for i, zone_name in enumerate(zone_names):
            if action & (1 << i):  # i番目のビットが1なら
                active.add(zone_name)
        
        return active
    
    def _calculate_reward(self):
        """
        報酬を計算
        
        報酬設計:
            - エージェントに近い照明が点灯している: +10.0 / 照明
            - エージェントから遠い照明が点灯している: -5.0 / 照明（無駄な電力）
            - エージェントに近い照明が消灯している: -15.0 / 照明（暗い）
            - 省エネ: 使用照明数 * -1.0
        
        Returns:
            float: 報酬値
        """
        reward = 0.0
        
        # 距離の閾値（ピクセル単位）
        near_threshold = NEAR_THRESHOLD
        
        for zone_name in self.room_shape.get_zone_names():
            distance = self._get_distance_to_zone(zone_name)
            is_near = distance <= near_threshold
            is_on = zone_name in self.active_lights
            
            if is_near and is_on:
                # 近くて点灯している：良い
                reward += NEAR_LIGHT_REWARD
            elif is_near and not is_on:
                # 近いのに消灯している：悪い（暗い）
                reward += NEAR_DARK_PENALTY
            elif not is_near and is_on:
                # 遠いのに点灯している：無駄
                reward += FAR_LIGHT_PENALTY
            # 遠くて消灯している場合は何もしない（適切）
        
        # 省エネ報酬
        num_lights = len(self.active_lights)
        reward += num_lights * ENERGY_COST_PER_LIGHT
        
        return reward
    
    def get_map_display(self):
        """
        現在のマップ状態を文字列で返す
        
        Returns:
            str: マップの情報文字列
        """
        display_lines = []
        display_lines.append("=" * 50)
        display_lines.append(f"エージェント位置: ({self.agent_x:.1f}, {self.agent_y:.1f})")
        display_lines.append(f"移動角度: {math.degrees(self.agent_angle):.1f}度")
        display_lines.append(f"ステップ: {self.step_count}/{self.max_steps}")
        display_lines.append(f"点灯照明: {', '.join(self.active_lights) if self.active_lights else 'なし'}")
        display_lines.append("=" * 50)
        return '\n'.join(display_lines)
    
    def render_to_image(self, cell_size=1):
        """
        現在のマップ状態を画像として返す
        
        Args:
            cell_size: スケーリング係数（デフォルト1でピクセル単位）
        
        Returns:
            Image: PIL Imageオブジェクト
        
        色の定義:
            緑色: エージェント位置
            黄色: 照明エリア（点灯）
            グレー: 照明エリア（消灯）
            黒色: 通常の床
        """
        img_width = int(self.map_width * cell_size)
        img_height = int(self.map_height * cell_size)
        img = Image.new('RGB', (img_width, img_height), color='black')
        draw = ImageDraw.Draw(img)
        
        # 照明ゾーンを描画
        for zone_name, zone_info in self.room_shape.get_all_light_zones().items():
            x1 = int(zone_info['x_range'][0] * cell_size)
            y1 = int(zone_info['y_range'][0] * cell_size)
            x2 = int(zone_info['x_range'][1] * cell_size)
            y2 = int(zone_info['y_range'][1] * cell_size)
            
            if zone_name in self.active_lights:
                color = 'yellow'  # 点灯中
            else:
                color = 'gray'  # 消灯中
            
            draw.rectangle([x1, y1, x2, y2], fill=color, outline='white')
        
        # エージェントを描画（円形）
        agent_radius = int(AGENT_RADIUS * cell_size)
        agent_x = int(self.agent_x * cell_size)
        agent_y = int(self.agent_y * cell_size)
        draw.ellipse(
            [agent_x - agent_radius, agent_y - agent_radius,
             agent_x + agent_radius, agent_y + agent_radius],
            fill='green', outline='white'
        )
        
        # 移動方向を示す矢印（追加）
        arrow_length = int(ARROW_LENGTH * cell_size)
        arrow_end_x = agent_x + int(arrow_length * math.cos(self.agent_angle))
        arrow_end_y = agent_y + int(arrow_length * math.sin(self.agent_angle))
        draw.line([agent_x, agent_y, arrow_end_x, arrow_end_y], fill='red', width=int(ARROW_WIDTH * cell_size))
        
        return img
    
    def get_config(self):
        """
        環境の設定を辞書形式で返す
        
        Returns:
            dict: 設定情報
        """
        return {
            'map_width': self.map_width,
            'map_height': self.map_height,
            'map_size': self.map_size,  # 互換性のため保持
            'agent_speed': self.agent_speed,
            'zone_size': self.zone_size,
            'max_steps': self.max_steps,
            'num_light_zones': self.num_lights,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'grid_x_size': self.grid_x_size,
            'grid_y_size': self.grid_y_size,
        }


class QLearningAgent:
    """
    Q-Learning エージェントクラス
    """
    
    def __init__(self, n_states=N_STATES, n_actions=N_ACTIONS, learning_rate=LEARNING_RATE, 
                 discount_factor=DISCOUNT_FACTOR, epsilon=EPSILON_INITIAL, 
                 epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN):
        """
        エージェントの初期化
        
        Args:
            n_states: 状態数
            n_actions: 行動数
            learning_rate: 学習率
            discount_factor: 割引率
            epsilon: ε-greedy法の初期ε値
            epsilon_decay: εの減衰率
            epsilon_min: εの最小値
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-tableの初期化
        self.q_table = np.zeros((n_states, n_actions))
    
    def get_action(self, state, training=True):
        """
        状態に基づいて行動を選択
        
        Args:
            state: 現在の状態
            training: 学習中かどうか
        
        Returns:
            int: 選択された行動
        """
        if training and np.random.random() < self.epsilon:
            # ランダムな行動を選択（探索）
            return np.random.randint(0, self.n_actions)
        else:
            # Q値が最大の行動を選択（活用）
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Q値を更新
        
        Args:
            state: 現在の状態
            action: 実行した行動
            reward: 得られた報酬
            next_state: 次の状態
            done: エピソード終了フラグ
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Q値の更新
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
    
    def decay_epsilon(self):
        """
        εを減衰させる
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(env, agent, n_episodes=N_EPISODES, display_interval=DISPLAY_INTERVAL, verbose=VERBOSE, save_video=True):
    """
    エージェントを学習させる
    
    Args:
        env: 環境
        agent: エージェント
        n_episodes: エピソード数
        display_interval: 表示間隔
        verbose: 詳細表示フラグ
    
    Returns:
        list: 各エピソードの報酬履歴
    """
    rewards_history = []
    frames = []  # GIF用のフレーム
    
    print("=" * 50)
    print("部屋照明最適化Q-Learning 学習開始（ピクセル単位版）")
    print("=" * 50)
    
    for episode in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            step += 1
            
            # 行動選択
            action = agent.get_action(state, training=True)
            
            # 行動実行
            next_state, reward, done = env.step(action)
            
            # Q値更新
            agent.update(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            
            # 最初のエピソードと定期的に表示
            if verbose and (episode == 1 or episode % display_interval == 1) and step <= 10:
                print(f"\nエピソード: {episode}/{n_episodes}")
                print(env.get_map_display())
                
                # フレーム保存（最初の数エピソードのみ）
                if episode <= 3:
                    frames.append(env.render_to_image())
        
        rewards_history.append(total_reward)
        agent.decay_epsilon()
        
        # 定期的に進捗表示
        if episode % STATS_INTERVAL == 0:
            avg_reward = np.mean(rewards_history[-STATS_INTERVAL:])
            print(f"\nエピソード {episode}/{n_episodes} | 平均報酬: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f}")
    
    print("\n" + "=" * 50)
    print("学習完了！")
    print("=" * 50)
    
    # 学習進捗の評価
    if len(rewards_history) > 0:
        print("\n学習進捗を評価中...")
        evaluation_result = evaluate_learning_progress(rewards_history)
        print_learning_evaluation(rewards_history, evaluation_result)
        
        # 学習曲線の可視化
        if HAS_MATPLOTLIB:
            plot_learning_curve(rewards_history)
    
    # 学習過程のGIF保存（save_videoがTrueの場合のみ）
    if frames and SAVE_TRAINING_FRAMES and save_video:
        print("\n動画を保存中...")
        frames[0].save(
            'room_lighting_training.gif',
            save_all=True,
            append_images=frames[1:],
            duration=TRAINING_GIF_DURATION,
            loop=0
        )
        print(f"学習過程を保存しました: room_lighting_training.gif ({len(frames)}フレーム)")
    elif frames and not save_video:
        print(f"\n動画生成をスキップしました（--no-renderオプション）")
    
    return rewards_history


def plot_learning_curve(rewards_history, save_path='learning_curve.png'):
    """
    学習曲線を可視化して保存
    
    Args:
        rewards_history: 各エピソードの報酬履歴
        save_path: 保存先のパス
    """
    if not HAS_MATPLOTLIB:
        print("学習曲線の可視化をスキップします（matplotlibが利用できません）")
        return
    
    try:
        # 移動平均の計算
        window = min(100, len(rewards_history) // 10)  # ウィンドウサイズ（最小100エピソード）
        if window > 0:
            moving_avg = []
            for i in range(len(rewards_history)):
                start_idx = max(0, i - window + 1)
                moving_avg.append(np.mean(rewards_history[start_idx:i+1]))
        else:
            moving_avg = rewards_history
        
        # グラフの作成
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. Learning Curve: Reward Progress (Top graph)
        episodes = range(1, len(rewards_history) + 1)
        ax1.plot(episodes, rewards_history, alpha=0.3, color='blue', label='Episode Reward')
        ax1.plot(episodes, moving_avg, color='red', linewidth=2, label=f'Moving Average ({window} episodes)')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.set_title('Learning Curve: Reward Progress', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Reward Distribution Histogram (Bottom graph)
        # This histogram shows the frequency distribution of rewards across all episodes
        # It helps visualize:
        # - How rewards are distributed (normal distribution, skewed, etc.)
        # - The spread of rewards (variance)
        # - Whether rewards are concentrated in certain ranges
        ax2.hist(rewards_history, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(np.mean(rewards_history), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(rewards_history):.2f}')
        ax2.axvline(np.median(rewards_history), color='blue', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(rewards_history):.2f}')
        ax2.set_xlabel('Reward', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Reward Distribution (Histogram)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 学習曲線を保存しました: {save_path}")
    except Exception as e:
        print(f"学習曲線の保存中にエラーが発生しました: {e}")


def evaluate_learning_progress(rewards_history, convergence_window=200, convergence_threshold=0.1):
    """
    学習進捗を評価
    
    Args:
        rewards_history: 各エピソードの報酬履歴
        convergence_window: 収束判定のウィンドウサイズ
        convergence_threshold: 収束判定の閾値（標準偏差の比率）
    
    Returns:
        dict: 評価結果の辞書
    """
    if len(rewards_history) < convergence_window:
        return {
            'converged': False,
            'reason': f'エピソード数が不足しています（{len(rewards_history)} < {convergence_window}）'
        }
    
    # 統計情報
    all_avg = np.mean(rewards_history)
    recent_avg = np.mean(rewards_history[-convergence_window:])
    early_avg = np.mean(rewards_history[:convergence_window])
    
    # 標準偏差
    recent_std = np.std(rewards_history[-convergence_window:])
    recent_mean = recent_avg
    
    # 改善率
    improvement = recent_avg - early_avg
    improvement_rate = (improvement / abs(early_avg)) * 100 if early_avg != 0 else 0
    
    # 収束判定（最後のNエピソードの標準偏差が平均の一定割合以下）
    convergence_ratio = recent_std / abs(recent_mean) if recent_mean != 0 else float('inf')
    is_converged = convergence_ratio < convergence_threshold and improvement > 0
    
    # トレンド分析
    if len(rewards_history) >= 2 * convergence_window:
        mid_avg = np.mean(rewards_history[convergence_window:2*convergence_window])
        trend = '上昇' if recent_avg > mid_avg else '下降' if recent_avg < mid_avg else '横ばい'
    else:
        trend = '分析不可'
    
    return {
        'converged': is_converged,
        'all_avg': all_avg,
        'recent_avg': recent_avg,
        'early_avg': early_avg,
        'improvement': improvement,
        'improvement_rate': improvement_rate,
        'recent_std': recent_std,
        'convergence_ratio': convergence_ratio,
        'trend': trend,
        'max_reward': np.max(rewards_history),
        'min_reward': np.min(rewards_history),
        'last_100_avg': np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else None
    }


def print_learning_evaluation(rewards_history, evaluation_result):
    """
    学習評価結果を表示
    
    Args:
        rewards_history: 各エピソードの報酬履歴
        evaluation_result: evaluate_learning_progressの結果
    """
    print("\n" + "=" * 60)
    print("学習進捗評価")
    print("=" * 60)
    
    print(f"\n【基本統計】")
    print(f"  全エピソード平均報酬: {evaluation_result['all_avg']:.2f}")
    if evaluation_result['last_100_avg'] is not None:
        print(f"  最後の100エピソード平均: {evaluation_result['last_100_avg']:.2f}")
    print(f"  最初の{min(200, len(rewards_history))}エピソード平均: {evaluation_result['early_avg']:.2f}")
    print(f"  最後の200エピソード平均: {evaluation_result['recent_avg']:.2f}")
    print(f"  最大報酬: {evaluation_result['max_reward']:.2f}")
    print(f"  最小報酬: {evaluation_result['min_reward']:.2f}")
    
    print(f"\n【学習改善】")
    print(f"  改善量: {evaluation_result['improvement']:.2f}")
    print(f"  改善率: {evaluation_result['improvement_rate']:.1f}%")
    print(f"  トレンド: {evaluation_result['trend']}")
    
    print(f"\n【収束判定】")
    if evaluation_result['converged']:
        print(f"  ✓ 学習が収束している可能性があります")
        print(f"  最後の200エピソードの標準偏差/平均比: {evaluation_result['convergence_ratio']:.4f}")
    else:
        print(f"  ⚠ 学習がまだ収束していない可能性があります")
        if 'reason' in evaluation_result:
            print(f"  理由: {evaluation_result['reason']}")
        else:
            print(f"  最後の200エピソードの標準偏差/平均比: {evaluation_result['convergence_ratio']:.4f}")
            print(f"  （0.1以下で収束と判定されます）")
    
    # 学習の質の評価
    print(f"\n【学習の質】")
    if evaluation_result['improvement_rate'] > 50:
        quality = "優秀"
    elif evaluation_result['improvement_rate'] > 20:
        quality = "良好"
    elif evaluation_result['improvement_rate'] > 0:
        quality = "改善中"
    else:
        quality = "改善が必要"
    
    print(f"  評価: {quality}")
    if evaluation_result['recent_avg'] > evaluation_result['all_avg']:
        print(f"  ✓ 最近のエピソードが全平均より高い（学習が進んでいる）")
    else:
        print(f"  ⚠ 最近のエピソードが全平均より低い（学習に問題がある可能性）")
    
    print("=" * 60)


def test_agent(env, agent, verbose=True, save_video=True):
    """
    学習済みエージェントをテスト
    
    Args:
        env: 環境
        agent: 学習済みエージェント
        verbose: 詳細表示フラグ
        save_video: 動画を保存するかどうか
    
    Returns:
        float: 総報酬
    """
    frames = []
    total_reward_all = 0
    episode_count = 0
    # 保存するフレーム数（デフォルト10000）
    try:
        max_frames = OPTIMAL_FRAMES_TO_SAVE
    except NameError:
        max_frames = 10000
    
    if verbose:
        print("\n" + "=" * 50)
        print("最適照明パターンの実行")
        print(f"保存フレーム数: {max_frames}フレーム")
        print("=" * 50)
    
    # 10000フレーム分実行（複数エピソード）
    while len(frames) < max_frames:
        episode_count += 1
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        if verbose and episode_count == 1:
            print("\n初期状態:")
            print(env.get_map_display())
        
        # 最初のフレームを保存
        if len(frames) == 0:
            frames.append(env.render_to_image())
        
        while not done and len(frames) < max_frames:
            step += 1
            
            # 最適な行動を選択（探索なし）
            action = agent.get_action(state, training=False)
            
            # 行動実行
            next_state, reward, done = env.step(action)
            
            total_reward += reward
            state = next_state
            
            if verbose and step % 100 == 0:
                print(f"エピソード {episode_count}, ステップ {step}, 保存フレーム数: {len(frames)}/{max_frames}")
            
            # フレームを保存
            if len(frames) < max_frames:
                frames.append(env.render_to_image())
        
        total_reward_all += total_reward
        
        if verbose:
            print(f"エピソード {episode_count} 完了: {step}ステップ, 報酬: {total_reward:.2f}")
        
        # エピソードが終了したが、まだフレームが足りない場合は次のエピソードを開始
        if len(frames) < max_frames and done:
            if verbose:
                print(f"次のエピソードを開始します... (現在: {len(frames)}/{max_frames}フレーム)")
    
    if verbose:
        print("\n" + "=" * 50)
        print("経路完了！")
        print(f"実行エピソード数: {episode_count}")
        print(f"保存フレーム数: {len(frames)}フレーム")
        print(f"総報酬: {total_reward_all:.2f}")
        print("=" * 50)
    
    # 最適パターンのGIF保存（save_videoがTrueの場合のみ）
    if frames and SAVE_OPTIMAL_FRAMES and save_video:
        print("\n最適照明パターンの動画を保存中...")
        frames[0].save(
            'room_lighting_optimal.gif',
            save_all=True,
            append_images=frames[1:],
            duration=OPTIMAL_GIF_DURATION,
            loop=0
        )
        video_duration = len(frames) * OPTIMAL_GIF_DURATION / 1000
        print(f"最適照明パターンを保存しました: room_lighting_optimal.gif")
        print(f"  フレーム数: {len(frames)}フレーム")
        print(f"  動画の長さ: {video_duration:.1f}秒 ({video_duration/60:.2f}分)")
        print(f"  FPS: {1000/OPTIMAL_GIF_DURATION:.1f}fps")
    elif frames and not save_video:
        print(f"\n動画生成をスキップしました（--no-renderオプション）")
    
    return total_reward_all


def parse_arguments():
    """
    コマンドライン引数を解析
    
    Returns:
        argparse.Namespace: 解析された引数
    """
    parser = argparse.ArgumentParser(
        description='部屋照明最適化Q-Learningシステム',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用例:
  # 学習と動画生成を実行
  python room_lighting_optimization.py
  
  # 動画生成をスキップして学習のみ実行
  python room_lighting_optimization.py --no-render
  
  # 既存のQ-tableを使用して動画生成のみ実行
  python room_lighting_optimization.py --render-only
  
  # カスタムQ-tableファイルを指定して動画生成
  python room_lighting_optimization.py --render-only --q-table custom_q_table.npy
        '''
    )
    
    parser.add_argument(
        '--render-only',
        action='store_true',
        help='既存のQ-tableを読み込んで動画生成のみを実行（学習をスキップ）'
    )
    
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='動画生成をスキップして学習のみを実行'
    )
    
    parser.add_argument(
        '--q-table',
        type=str,
        default='room_lighting_q_table.npy',
        help='読み込むQ-tableファイルのパス（デフォルト: room_lighting_q_table.npy）'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='room_lighting_config.json',
        help='読み込む設定ファイルのパス（デフォルト: room_lighting_config.json）'
    )
    
    return parser.parse_args()


def load_existing_qtable(q_table_path, n_states, n_actions):
    """
    既存のQ-tableを読み込む
    
    Args:
        q_table_path: Q-tableファイルのパス
        n_states: 期待される状態数
        n_actions: 期待される行動数
    
    Returns:
        np.ndarray: 読み込んだQ-table
    """
    if not os.path.exists(q_table_path):
        raise FileNotFoundError(f"Q-tableファイルが見つかりません: {q_table_path}")
    
    q_table = np.load(q_table_path)
    
    # サイズチェック
    if q_table.shape != (n_states, n_actions):
        raise ValueError(
            f"Q-tableのサイズが一致しません。"
            f"期待: ({n_states}, {n_actions}), 実際: {q_table.shape}"
        )
    
    print(f"✓ Q-tableを読み込みました: {q_table_path}")
    print(f"  サイズ: {q_table.shape}")
    return q_table


def main():
    """
    メイン関数
    """
    # コマンドライン引数を解析
    args = parse_arguments()
    
    # パラメータの表示と検証
    print_parameters()
    validate_parameters()
    
    # マップクラスを読み込み
    map_class = load_map_class(MAP_PATH)
    
    # 環境を初期化
    env = RoomLightingEnvironment(room_shape_class=map_class)
    
    # 環境から動的に状態数と行動数を取得してエージェントを初期化
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
    
    print(f"\nマップサイズ: {env.map_width}×{env.map_height}ピクセル")
    print(f"エージェント速度: {env.agent_speed}ピクセル/ステップ")
    print(f"照明ゾーンサイズ: {env.zone_size}×{env.zone_size}ピクセル")
    print(f"状態数: {env.n_states} ({env.grid_x_size}×{env.grid_y_size}グリッド)")
    print(f"行動数: {env.n_actions} (2^{env.num_lights} = {2**env.num_lights})")
    print(f"照明ゾーン: {env.num_lights}つ")
    print(f"照明ゾーン名: {', '.join(env.room_shape.get_zone_names())}")
    
    # --render-onlyが指定された場合、既存のQ-tableを読み込んで動画生成のみ実行
    if args.render_only:
        print("\n" + "=" * 50)
        print("動画生成モード（学習をスキップ）")
        print("=" * 50)
        
        try:
            # 既存のQ-tableを読み込む
            agent.q_table = load_existing_qtable(
                args.q_table,
                env.n_states,
                env.n_actions
            )
            
            # 設定ファイルが存在する場合は読み込む（オプション）
            if os.path.exists(args.config):
                with open(args.config, 'r') as f:
                    config = json.load(f)
                print(f"✓ 設定ファイルを読み込みました: {args.config}")
            
            # 動画生成のみ実行
            print(f"\n最適照明パターンの動画を生成します...")
            test_reward = test_agent(env, agent, save_video=True)
            
            print("\n" + "=" * 50)
            print("動画生成完了！")
            print("=" * 50)
            
        except FileNotFoundError as e:
            print(f"\nエラー: {e}")
            print("学習を先に実行してください。")
            return
        except ValueError as e:
            print(f"\nエラー: {e}")
            print("Q-tableのサイズが環境と一致しません。")
            return
        
        return
    
    # 通常モード: 学習を実行
    print("\n" + "=" * 50)
    print("学習モード")
    if args.no_render:
        print("（動画生成をスキップ）")
    print("=" * 50)
    
    # 動画生成をスキップするかどうか
    save_video = not args.no_render
    
    # 学習
    rewards_history = train_agent(env, agent, save_video=save_video)
    
    # Q-table統計情報表示（学習進捗評価はtrain_agent内で既に表示済み）
    print(f"\nQ-table統計:")
    print(f"  平均Q値: {np.mean(agent.q_table):.4f}")
    print(f"  最大Q値: {np.max(agent.q_table):.4f}")
    print(f"  最小Q値: {np.min(agent.q_table):.4f}")
    print(f"  Q値の範囲: {np.max(agent.q_table) - np.min(agent.q_table):.4f}")
    
    # モデル保存
    print(f"\nモデルを保存中...")
    np.save('room_lighting_q_table.npy', agent.q_table)
    print(f"Q-tableを保存しました: room_lighting_q_table.npy")
    
    # 環境設定保存
    config = env.get_config()
    with open('room_lighting_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"環境設定を保存しました: room_lighting_config.json")
    
    # テスト実行（動画生成をスキップするかどうか）
    if not args.no_render:
        print(f"\n最適照明パターンを表示します...")
        test_reward = test_agent(env, agent, save_video=True)
    else:
        print(f"\n動画生成をスキップしました（--no-renderオプション）")


if __name__ == "__main__":
    main()

