#!/usr/bin/env python3
"""
简单的运行脚本
在运行前需要激活gfootball虚拟环境
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入主函数
from src.main import main

if __name__ == "__main__":
    print("=" * 60)
    print("Google Research Football 决策树智能体")
    print("=" * 60)
    print("请确保已经:")
    print("1. 安装了Google Research Football环境")
    print("2. 激活了相应的虚拟环境")
    print("3. 安装了必要的依赖包 (numpy等)")
    print("=" * 60)
    
    main()