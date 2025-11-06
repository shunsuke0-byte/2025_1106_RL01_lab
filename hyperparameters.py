"""
ハイパーパラメータ設定ファイル

project_02のすべてのハイパーパラメータを一箇所で管理
"""

# =============================================================================
# 環境設定パラメータ
# =============================================================================

# マップ設定
MAP_SIZE = 500              # マップサイズ（ピクセル）
AGENT_SPEED = 10             # エージェントの移動速度（ピクセル/ステップ）
ZONE_SIZE = 100             # 照明ゾーンのサイズ（ピクセル）
# MAX_STEPS_PER_EPISODE = 200 # 1エピソードの最大ステップ数
MAX_STEPS_PER_EPISODE = 1000 # 1エピソードの最大ステップ数

# 照明ゾーン設定
LIGHT_ZONES = {
    'top_left': {'x_range': (0, ZONE_SIZE), 'y_range': (0, ZONE_SIZE)},
    'top_right': {'x_range': (MAP_SIZE - ZONE_SIZE, MAP_SIZE), 'y_range': (0, ZONE_SIZE)},
    'bottom_left': {'x_range': (0, ZONE_SIZE), 'y_range': (MAP_SIZE - ZONE_SIZE, MAP_SIZE)},
    'bottom_right': {'x_range': (MAP_SIZE - ZONE_SIZE, MAP_SIZE), 'y_range': (MAP_SIZE - ZONE_SIZE, MAP_SIZE)},
}

# =============================================================================
# Q-Learning パラメータ
# =============================================================================

# 状態・行動空間
N_STATES = 100              # 状態数（10×10グリッドに離散化）
N_ACTIONS = 16              # 行動数（4つの照明の組み合わせ、2^4 = 16通り）

# 学習パラメータ
LEARNING_RATE = 0.1         # 学習率
DISCOUNT_FACTOR = 0.95      # 割引率（γ）
EPSILON_INITIAL = 1.0       # 初期ε値（完全ランダム）
EPSILON_DECAY = 0.995       # ε減衰率
EPSILON_MIN = 0.01          # 最小ε値

# 学習設定
# N_EPISODES = 1000           # 総エピソード数
N_EPISODES = 3000           # 総エピソード数
DISPLAY_INTERVAL = 50       # 表示間隔（エピソード）

# =============================================================================
# 報酬システムパラメータ
# =============================================================================

# 距離ベースの報酬
NEAR_THRESHOLD = 100.0      # 近距離判定の閾値（ピクセル）
NEAR_LIGHT_REWARD = 10.0    # 近くて点灯している照明の報酬
NEAR_DARK_PENALTY = -15.0   # 近いのに消灯している照明のペナルティ
FAR_LIGHT_PENALTY = -5.0    # 遠いのに点灯している照明のペナルティ

# 省エネ報酬
ENERGY_COST_PER_LIGHT = 1.0 # 1つの照明あたりの電力コスト

# =============================================================================
# 可視化パラメータ
# =============================================================================

# 画像生成設定
CELL_SIZE = 1               # スケーリング係数（デフォルト1でピクセル単位）
AGENT_RADIUS = 10           # エージェントの表示半径（ピクセル）
ARROW_LENGTH = 20           # 方向矢印の長さ（ピクセル）
ARROW_WIDTH = 3             # 方向矢印の太さ（ピクセル）

# 動画生成設定
TRAINING_GIF_DURATION = 200 # 学習過程GIFのフレーム間隔（ミリ秒）
OPTIMAL_GIF_DURATION = 50   # 最適パターンGIFのフレーム間隔（ミリ秒、20fps）
OPTIMAL_FRAMES_TO_SAVE = 10000  # 保存するフレーム数（10000フレーム分）

# =============================================================================
# デバッグ・ログ設定
# =============================================================================

VERBOSE = True              # 詳細表示フラグ
SAVE_TRAINING_FRAMES = True # 学習過程のフレーム保存フラグ
SAVE_OPTIMAL_FRAMES = True  # 最適パターンのフレーム保存フラグ

# 統計表示間隔
STATS_INTERVAL = 100        # 統計表示間隔（エピソード）

# =============================================================================
# パラメータ検証関数
# =============================================================================

def validate_parameters():
    """
    パラメータの妥当性を検証
    """
    errors = []
    
    # マップサイズの検証
    if MAP_SIZE <= 0:
        errors.append("MAP_SIZE must be positive")
    
    if AGENT_SPEED <= 0:
        errors.append("AGENT_SPEED must be positive")
    
    if ZONE_SIZE <= 0 or ZONE_SIZE >= MAP_SIZE:
        errors.append("ZONE_SIZE must be positive and less than MAP_SIZE")
    
    # 学習パラメータの検証
    if not 0 < LEARNING_RATE <= 1:
        errors.append("LEARNING_RATE must be between 0 and 1")
    
    if not 0 <= DISCOUNT_FACTOR <= 1:
        errors.append("DISCOUNT_FACTOR must be between 0 and 1")
    
    if not 0 <= EPSILON_INITIAL <= 1:
        errors.append("EPSILON_INITIAL must be between 0 and 1")
    
    if not 0 < EPSILON_DECAY < 1:
        errors.append("EPSILON_DECAY must be between 0 and 1")
    
    if not 0 <= EPSILON_MIN <= EPSILON_INITIAL:
        errors.append("EPSILON_MIN must be between 0 and EPSILON_INITIAL")
    
    # 状態・行動空間の検証
    if N_STATES <= 0:
        errors.append("N_STATES must be positive")
    
    if N_ACTIONS <= 0:
        errors.append("N_ACTIONS must be positive")
    
    if errors:
        raise ValueError("Parameter validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
    
    print("✓ All parameters are valid!")

# =============================================================================
# パラメータ表示関数
# =============================================================================

def print_parameters():
    """
    現在のパラメータを表示
    """
    print("=" * 60)
    print("ハイパーパラメータ設定")
    print("=" * 60)
    
    print("\n【環境設定】")
    print(f"  マップサイズ: {MAP_SIZE}×{MAP_SIZE}ピクセル")
    print(f"  エージェント速度: {AGENT_SPEED}ピクセル/ステップ")
    print(f"  照明ゾーンサイズ: {ZONE_SIZE}×{ZONE_SIZE}ピクセル")
    print(f"  最大ステップ数: {MAX_STEPS_PER_EPISODE}/エピソード")
    
    print("\n【Q-Learning設定】")
    print(f"  状態数: {N_STATES}")
    print(f"  行動数: {N_ACTIONS}")
    print(f"  学習率: {LEARNING_RATE}")
    print(f"  割引率: {DISCOUNT_FACTOR}")
    print(f"  初期ε値: {EPSILON_INITIAL}")
    print(f"  ε減衰率: {EPSILON_DECAY}")
    print(f"  最小ε値: {EPSILON_MIN}")
    print(f"  総エピソード数: {N_EPISODES}")
    
    print("\n【報酬システム】")
    print(f"  近距離閾値: {NEAR_THRESHOLD}ピクセル")
    print(f"  近くて点灯: +{NEAR_LIGHT_REWARD}")
    print(f"  近いのに消灯: {NEAR_DARK_PENALTY}")
    print(f"  遠いのに点灯: {FAR_LIGHT_PENALTY}")
    print(f"  電力コスト: {ENERGY_COST_PER_LIGHT}/照明")
    
    print("\n【可視化設定】")
    print(f"  エージェント半径: {AGENT_RADIUS}ピクセル")
    print(f"  矢印長さ: {ARROW_LENGTH}ピクセル")
    print(f"  学習GIF間隔: {TRAINING_GIF_DURATION}ms")
    print(f"  最適GIF間隔: {OPTIMAL_GIF_DURATION}ms")
    
    print("=" * 60)

# =============================================================================
# パラメータ変更関数
# =============================================================================

def set_learning_parameters(lr=None, gamma=None, epsilon_init=None, 
                          epsilon_decay=None, epsilon_min=None, episodes=None):
    """
    学習パラメータを動的に変更
    
    Args:
        lr: 学習率
        gamma: 割引率
        epsilon_init: 初期ε値
        epsilon_decay: ε減衰率
        epsilon_min: 最小ε値
        episodes: エピソード数
    """
    global LEARNING_RATE, DISCOUNT_FACTOR, EPSILON_INITIAL
    global EPSILON_DECAY, EPSILON_MIN, N_EPISODES
    
    if lr is not None:
        LEARNING_RATE = lr
    if gamma is not None:
        DISCOUNT_FACTOR = gamma
    if epsilon_init is not None:
        EPSILON_INITIAL = epsilon_init
    if epsilon_decay is not None:
        EPSILON_DECAY = epsilon_decay
    if epsilon_min is not None:
        EPSILON_MIN = epsilon_min
    if episodes is not None:
        N_EPISODES = episodes
    
    print("学習パラメータを更新しました")

def set_environment_parameters(map_size=None, speed=None, zone_size=None, max_steps=None):
    """
    環境パラメータを動的に変更
    
    Args:
        map_size: マップサイズ
        speed: エージェント速度
        zone_size: 照明ゾーンサイズ
        max_steps: 最大ステップ数
    """
    global MAP_SIZE, AGENT_SPEED, ZONE_SIZE, MAX_STEPS_PER_EPISODE
    
    if map_size is not None:
        MAP_SIZE = map_size
    if speed is not None:
        AGENT_SPEED = speed
    if zone_size is not None:
        ZONE_SIZE = zone_size
    if max_steps is not None:
        MAX_STEPS_PER_EPISODE = max_steps
    
    print("環境パラメータを更新しました")

# =============================================================================
# メイン実行
# =============================================================================

if __name__ == "__main__":
    print_parameters()
    validate_parameters()
