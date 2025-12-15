#!/bin/bash
# 强化学习项目快速启动脚本

set -e

RL_HOME="/home/ryan/2repo/my_infra/RL"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                   强化学习快速启动脚本                              ║"
echo "║             Reinforcement Learning Quick Start                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# 检查Python版本
echo "📋 检查环境..."
python_version=$(python3 --version 2>&1)
echo "✓ Python: $python_version"

# 检查必要的包
echo ""
echo "📦 检查依赖包..."
packages=("numpy" "matplotlib")
missing_packages=()

for package in "${packages[@]}"; do
    python3 -c "import $package" 2>/dev/null && echo "✓ $package" || missing_packages+=("$package")
done

# 尝试导入torch（可选）
python3 -c "import torch" 2>/dev/null && echo "✓ torch" || echo "⚠ torch (可选，Policy Gradient需要)"

if [ ${#missing_packages[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  需要安装以下包:"
    for package in "${missing_packages[@]}"; do
        echo "  - $package"
    done
    echo ""
    echo "请运行: pip install numpy matplotlib torch"
    echo ""
fi

# 显示项目结构
echo ""
echo "📁 项目结构:"
echo ""
ls -lh "$RL_HOME" | tail -n +2 | awk '{print "  " $9}'
echo ""

# 提供选项菜单
while true; do
    echo "═══════════════════════════════════════════════════════════════"
    echo "请选择要运行的模块:"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "1) 模块1: MDP基础概念"
    echo "2) 模块2: 动态规划"
    echo "3) 模块3: 蒙特卡洛方法"
    echo "4) 模块4: 时序差分学习"
    echo "5) 模块5: 策略梯度方法"
    echo ""
    echo "6) 查看学习指南"
    echo "7) 查看完整README"
    echo "0) 退出"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    read -p "请输入选项 [0-7]: " choice
    echo ""
    
    case $choice in
        1)
            echo "运行 MDP基础 演示..."
            echo "═══════════════════════════════════════════════════════════════"
            cd "$RL_HOME"
            python3 01_MDP/mdp_basics.py
            ;;
        2)
            echo "运行 动态规划 演示..."
            echo "═══════════════════════════════════════════════════════════════"
            cd "$RL_HOME"
            python3 02_dynamic_programming/dp_solver.py 2>&1 | head -50
            ;;
        3)
            echo "运行 蒙特卡洛方法 演示..."
            echo "═══════════════════════════════════════════════════════════════"
            cd "$RL_HOME"
            python3 03_monte_carlo/mc_learning.py 2>&1 | head -50
            ;;
        4)
            echo "运行 时序差分学习 演示..."
            echo "═══════════════════════════════════════════════════════════════"
            cd "$RL_HOME"
            python3 04_temporal_difference/td_learning.py 2>&1 | head -50
            ;;
        5)
            echo "检查 torch 依赖..."
            if python3 -c "import torch" 2>/dev/null; then
                echo "运行 策略梯度方法 演示..."
                echo "═══════════════════════════════════════════════════════════════"
                cd "$RL_HOME"
                python3 05_policy_gradient/pg_learning.py 2>&1 | head -50
            else
                echo "⚠️  需要安装 PyTorch: pip install torch"
            fi
            ;;
        6)
            echo "查看学习指南..."
            echo "═══════════════════════════════════════════════════════════════"
            python3 "$RL_HOME/LEARNING_GUIDE.py" | head -100
            read -p "按Enter继续..."
            ;;
        7)
            echo "查看README..."
            echo "═══════════════════════════════════════════════════════════════"
            head -100 "$RL_HOME/README.md"
            read -p "按Enter继续..."
            ;;
        0)
            echo "👋 再见！祝你学习愉快！"
            exit 0
            ;;
        *)
            echo "❌ 无效选项，请重试"
            ;;
    esac
    
    echo ""
    read -p "按Enter返回菜单..."
    clear
done
