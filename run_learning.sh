#!/bin/bash

# 部屋照明最適化学習実行スクリプト

echo "============================================================"
echo "部屋照明最適化Q-Learning 学習開始"
echo "============================================================"

# Python環境をアクティベート
echo "Python環境をアクティベート中..."
source /Users/tanakashunsuke/Desktop/python/python01/bin/activate

# 現在のディレクトリに移動
cd /Users/tanakashunsuke/Desktop/universitiy/KubotaLab/python/20251028/project_02

# 学習を実行
echo "学習を開始します..."
python room_lighting_optimization.py

echo "============================================================"
echo "学習完了！"
echo "============================================================"
