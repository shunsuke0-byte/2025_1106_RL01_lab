"""
モデル評価スクリプト

学習済みQ-tableを読み込んで、複数回のテストエピソードで性能を評価します。
以下の指標を計算します：
- 平均報酬、標準偏差、最大/最小報酬
- 省エネエラー数（遠いのに点灯している回数）
- 距離エラー数（200ピクセル以上離れているのに点灯している回数）
"""

import numpy as np
import argparse
import json
import os
from room_lighting_optimization import (
    RoomLightingEnvironment, 
    QLearningAgent,
    load_map_class,
    MAP_PATH
)
from hyperparameters import NEAR_THRESHOLD

def evaluate_agent_detailed(env, agent, n_test_episodes=100, distance_threshold=200.0, verbose=True):
    """
    学習済みエージェントを詳細に評価
    
    Args:
        env: 環境
        agent: 学習済みエージェント
        n_test_episodes: テストエピソード数
        distance_threshold: 距離エラーの閾値（ピクセル）
        verbose: 詳細表示フラグ
    
    Returns:
        dict: 評価結果の辞書
    """
    test_rewards = []
    energy_errors = []  # 省エネエラー数（遠いのに点灯）
    distance_errors = []  # 距離エラー数（閾値以上離れているのに点灯）
    near_dark_errors = []  # 近いのに消灯エラー数
    total_steps = 0  # 総ステップ数
    num_lights = env.num_lights  # 照明数
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"モデル評価開始（{n_test_episodes}エピソード）")
        print("=" * 60)
        print(f"近距離閾値: {NEAR_THRESHOLD}ピクセル")
        print(f"距離エラー閾値: {distance_threshold}ピクセル")
        print(f"照明数: {num_lights}")
        print("=" * 60)
    
    for episode in range(n_test_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        # エピソードごとのエラーカウント
        episode_energy_errors = 0
        episode_distance_errors = 0
        episode_near_dark_errors = 0
        
        while not done:
            step += 1
            
            # 探索なし（training=False）で行動選択
            action = agent.get_action(state, training=False)
            
            # 行動実行
            next_state, reward, done = env.step(action)
            
            # エラーカウント
            for zone_name in env.room_shape.get_zone_names():
                distance = env._get_distance_to_zone(zone_name)
                is_on = zone_name in env.active_lights
                
                # 省エネエラー（近距離閾値より遠いのに点灯）
                if distance > NEAR_THRESHOLD and is_on:
                    episode_energy_errors += 1
                
                # 距離エラー（指定閾値以上離れているのに点灯）
                if distance >= distance_threshold and is_on:
                    episode_distance_errors += 1
                
                # 近いのに消灯エラー
                if distance <= NEAR_THRESHOLD and not is_on:
                    episode_near_dark_errors += 1
            
            total_reward += reward
            state = next_state
        
        test_rewards.append(total_reward)
        energy_errors.append(episode_energy_errors)
        distance_errors.append(episode_distance_errors)
        near_dark_errors.append(episode_near_dark_errors)
        total_steps += step  # 総ステップ数を記録
        
        # 進捗表示
        if verbose and (episode + 1) % 10 == 0:
            print(f"エピソード {episode + 1}/{n_test_episodes} 完了")
    
    # 統計計算
    results = {
        # 報酬統計
        'mean_reward': np.mean(test_rewards),
        'std_reward': np.std(test_rewards),
        'min_reward': np.min(test_rewards),
        'max_reward': np.max(test_rewards),
        'median_reward': np.median(test_rewards),
        
        # エラー統計
        'total_energy_errors': np.sum(energy_errors),
        'mean_energy_errors_per_episode': np.mean(energy_errors),
        'std_energy_errors': np.std(energy_errors),
        
        'total_distance_errors': np.sum(distance_errors),
        'mean_distance_errors_per_episode': np.mean(distance_errors),
        'std_distance_errors': np.std(distance_errors),
        
        'total_near_dark_errors': np.sum(near_dark_errors),
        'mean_near_dark_errors_per_episode': np.mean(near_dark_errors),
        'std_near_dark_errors': np.std(near_dark_errors),
        
        # 詳細データ
        'all_rewards': test_rewards,
        'all_energy_errors': energy_errors,
        'all_distance_errors': distance_errors,
        'all_near_dark_errors': near_dark_errors,
        
        # 設定情報
        'n_test_episodes': n_test_episodes,
        'distance_threshold': distance_threshold,
        'near_threshold': NEAR_THRESHOLD,
        'total_steps': total_steps,
        'num_lights': num_lights,
    }
    
    return results


def print_evaluation_results(results):
    """
    評価結果を見やすく表示
    
    Args:
        results: evaluate_agent_detailedの結果
    """
    print("\n" + "=" * 60)
    print("評価結果サマリー")
    print("=" * 60)
    
    print(f"\n【報酬統計】（{results['n_test_episodes']}エピソード）")
    print(f"  平均報酬: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  中央値: {results['median_reward']:.2f}")
    print(f"  最大報酬: {results['max_reward']:.2f}")
    print(f"  最小報酬: {results['min_reward']:.2f}")
    
    print(f"\n【省エネエラー】（{results['near_threshold']:.0f}ピクセルより遠いのに点灯）")
    print(f"  総エラー数: {results['total_energy_errors']:.0f}回")
    print(f"  エピソードあたり平均: {results['mean_energy_errors_per_episode']:.2f} ± {results['std_energy_errors']:.2f}回")
    
    print(f"\n【距離エラー】（{results['distance_threshold']:.0f}ピクセル以上離れているのに点灯）")
    print(f"  総エラー数: {results['total_distance_errors']:.0f}回")
    print(f"  エピソードあたり平均: {results['mean_distance_errors_per_episode']:.2f} ± {results['std_distance_errors']:.2f}回")
    
    print(f"\n【暗闇エラー】（{results['near_threshold']:.0f}ピクセル以内なのに消灯）")
    print(f"  総エラー数: {results['total_near_dark_errors']:.0f}回")
    print(f"  エピソードあたり平均: {results['mean_near_dark_errors_per_episode']:.2f} ± {results['std_near_dark_errors']:.2f}回")
    
    # エラー率の計算（実際のステップ数と照明数を使用）
    total_steps = results.get('total_steps', results['n_test_episodes'] * 1000)
    num_lights = results.get('num_lights', 4)
    total_decisions = total_steps * num_lights  # 各ステップで各照明について判断
    
    print(f"\n【エラー率】")
    energy_error_rate = (results['total_energy_errors'] / total_decisions) * 100 if total_decisions > 0 else 0
    distance_error_rate = (results['total_distance_errors'] / total_decisions) * 100 if total_decisions > 0 else 0
    near_dark_error_rate = (results['total_near_dark_errors'] / total_decisions) * 100 if total_decisions > 0 else 0
    
    print(f"  省エネエラー率: {energy_error_rate:.4f}%")
    print(f"  距離エラー率: {distance_error_rate:.4f}%")
    print(f"  暗闇エラー率: {near_dark_error_rate:.4f}%")
    
    print("=" * 60)


def save_evaluation_results(results, output_path='evaluation_results.json'):
    """
    評価結果をJSONファイルに保存
    
    Args:
        results: 評価結果の辞書
        output_path: 保存先パス
    """
    # NumPy配列・型をPython標準型に変換
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        else:
            return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 評価結果を保存しました: {output_path}")


def save_evaluation_summary(results, output_path='evaluation_summary.txt'):
    """
    評価結果のサマリーを見やすいテキスト形式で保存
    
    Args:
        results: 評価結果の辞書
        output_path: 保存先パス
    """
    # エラー率の計算（実際のステップ数と照明数を使用）
    total_steps = results.get('total_steps', results['n_test_episodes'] * 1000)
    num_lights = results.get('num_lights', 4)
    total_decisions = total_steps * num_lights  # 各ステップで各照明について判断
    
    energy_error_rate = (results['total_energy_errors'] / total_decisions) * 100 if total_decisions > 0 else 0
    distance_error_rate = (results['total_distance_errors'] / total_decisions) * 100 if total_decisions > 0 else 0
    near_dark_error_rate = (results['total_near_dark_errors'] / total_decisions) * 100 if total_decisions > 0 else 0
    
    summary = f"""================================================================
評価結果サマリー
================================================================

【報酬統計】（{results['n_test_episodes']}エピソード、探索なし）
  平均報酬: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}
  中央値: {results['median_reward']:.2f}
  最大報酬: {results['max_reward']:.2f}
  最小報酬: {results['min_reward']:.2f}

【省エネエラー】（{results['near_threshold']:.0f}ピクセルより遠いのに点灯）
  総エラー数: {results['total_energy_errors']:.0f}回
  エピソードあたり平均: {results['mean_energy_errors_per_episode']:.2f} ± {results['std_energy_errors']:.2f}回
  エラー率: {energy_error_rate:.2f}%

【距離エラー】（{results['distance_threshold']:.0f}ピクセル以上離れているのに点灯）
  総エラー数: {results['total_distance_errors']:.0f}回
  エピソードあたり平均: {results['mean_distance_errors_per_episode']:.2f} ± {results['std_distance_errors']:.2f}回
  エラー率: {distance_error_rate:.2f}%

【暗闇エラー】（{results['near_threshold']:.0f}ピクセル以内なのに消灯）
  総エラー数: {results['total_near_dark_errors']:.0f}回
  エピソードあたり平均: {results['mean_near_dark_errors_per_episode']:.2f} ± {results['std_near_dark_errors']:.2f}回
  エラー率: {near_dark_error_rate:.2f}%

================================================================
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"✓ 評価サマリーを保存しました: {output_path}")


def parse_arguments():
    """
    コマンドライン引数を解析
    """
    parser = argparse.ArgumentParser(
        description='学習済みモデルの評価スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用例:
  # デフォルト設定で評価（100エピソード）
  python evaluate_model.py
  
  # 200エピソードで評価
  python evaluate_model.py --episodes 200
  
  # 距離エラー閾値を300ピクセルに設定
  python evaluate_model.py --distance-threshold 300
  
  # カスタムQ-tableを指定
  python evaluate_model.py --q-table custom_q_table.npy
        '''
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
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='テストエピソード数（デフォルト: 100）'
    )
    
    parser.add_argument(
        '--distance-threshold',
        type=float,
        default=200.0,
        help='距離エラーの閾値（ピクセル、デフォルト: 200.0）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='評価結果の保存先（デフォルト: evaluation_results.json）'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='詳細表示を抑制'
    )
    
    return parser.parse_args()


def main():
    """
    メイン関数
    """
    # コマンドライン引数を解析
    args = parse_arguments()
    
    verbose = not args.quiet
    
    if verbose:
        print("=" * 60)
        print("学習済みモデルの評価")
        print("=" * 60)
    
    # Q-tableの存在確認
    if not os.path.exists(args.q_table):
        print(f"\nエラー: Q-tableファイルが見つかりません: {args.q_table}")
        print("先に学習を実行してください:")
        print("  python room_lighting_optimization.py")
        return
    
    # マップクラスを読み込み
    map_class = load_map_class(MAP_PATH)
    
    # 環境を初期化
    env = RoomLightingEnvironment(room_shape_class=map_class)
    
    # エージェントを初期化
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
    
    # Q-tableを読み込み
    try:
        agent.q_table = np.load(args.q_table)
        if verbose:
            print(f"\n✓ Q-tableを読み込みました: {args.q_table}")
            print(f"  サイズ: {agent.q_table.shape}")
    except Exception as e:
        print(f"\nエラー: Q-tableの読み込みに失敗しました: {e}")
        return
    
    # 設定ファイルが存在する場合は読み込む
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        if verbose:
            print(f"✓ 設定ファイルを読み込みました: {args.config}")
    
    if verbose:
        print(f"\nマップサイズ: {env.map_width}×{env.map_height}ピクセル")
        print(f"照明数: {env.num_lights}")
        print(f"状態数: {env.n_states}")
        print(f"行動数: {env.n_actions}")
    
    # 評価実行
    results = evaluate_agent_detailed(
        env, 
        agent, 
        n_test_episodes=args.episodes,
        distance_threshold=args.distance_threshold,
        verbose=verbose
    )
    
    # 結果表示
    print_evaluation_results(results)
    
    # 結果保存
    save_evaluation_results(results, args.output)
    
    # サマリー保存
    summary_path = args.output.replace('.json', '_summary.txt') if args.output.endswith('.json') else 'evaluation_summary.txt'
    save_evaluation_summary(results, summary_path)
    
    if verbose:
        print("\n評価完了！")


if __name__ == "__main__":
    main()

