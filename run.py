#!/usr/bin/env python3
"""
简单的运行脚本
在运行前需要激活gfootball虚拟环境
"""

import argparse
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import main


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Google Research Football 决策树智能体')

    parser.add_argument('--render', action='store_true', default=True,
                       help='启用渲染 (默认: True)')
    parser.add_argument('--write_video', action='store_true', default=False,
                       help='录制视频 (默认: False)')
    parser.add_argument('--logdir', type=str, default='',
                       help='日志目录 (默认: 空)')
    parser.add_argument('--num_episodes', type=int, default=1,
                       help='比赛局数 (默认: 1)')
    parser.add_argument('--max_steps', type=int, default=3000,
                       help='每局最大步数 (默认: 3000)')
    
    return parser.parse_args()


if __name__ == "__main__":
    print("=" * 60)
    print("Google Research Football 决策树智能体")
    print("=" * 60)
    print("请确保已经:")
    print("1. 安装了Google Research Football环境")
    print("2. 激活了相应的虚拟环境")
    print("3. 安装了必要的依赖包 (numpy等)")
    print("=" * 60)
    
    # 解析命令行参数
    args = parse_args()
    
    # 打印配置信息
    print("环境配置:")
    print(f"  渲染模式: {'启用' if args.render else '禁用'}")
    print(f"  视频录制: {'启用' if args.write_video else '禁用'}")
    print(f"  日志目录: {args.logdir if args.logdir else '无'}")
    print(f"运行配置:")
    print(f"  比赛局数: {args.num_episodes}")
    print(f"  最大步数: {args.max_steps}")
    print("=" * 60)
    
    # 运行主程序
    main(args)