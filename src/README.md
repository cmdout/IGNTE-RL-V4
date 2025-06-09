# Google Research Football 决策树智能体

基于战术分析和第一性原理设计的足球智能体，实现4-5-1阵型的Mid-Block防守和快速反击战术。

## 项目结构

```
src/
├── main.py                     # 项目主入口
├── run.py                      # 运行脚本
├── gfootball_agent/            # 核心智能体模块
│   ├── __init__.py
│   ├── agent.py                # 主Agent类
│   ├── config.py               # 配置文件
│   ├── decision_logic/         # 决策逻辑
│   │   ├── __init__.py
│   │   ├── top_level_logic.py  # 顶层决策分发
│   │   ├── normal_mode.py      # 常规模式逻辑
│   │   └── set_pieces.py       # 定位球逻辑
│   └── roles/                  # 球员角色决策
│       ├── __init__.py
│       ├── goalkeeper.py       # 守门员
│       ├── defender.py         # 后卫
│       ├── midfielder.py       # 中场
│       └── forward.py          # 前锋
└── utils/                      # 工具模块
    ├── __init__.py
    ├── features.py             # 特征工程
    └── actions.py              # 动作管理
```

## 核心特性

### 战术系统
- **4-5-1阵型**: 1名守门员、4名后卫、5名中场、1名前锋
- **Mid-Block防守**: 中场区域紧凑防守，封锁传球线路
- **快速反击**: 断球后迅速向前推进，利用对方失位

### 决策架构
- **分层决策**: 根据game_mode分发到常规/定位球模式
- **角色化智能**: 每个位置有专门的决策逻辑
- **状态驱动**: 基于控球权(我方/对方/无人)进行决策

### 技术实现
- **粘性动作管理**: 智能处理连续性动作
- **特征工程**: 距离、角度、位置关系计算
- **动作验证**: 确保动作在当前情况下合法

## 环境要求

### 必需安装
```bash
# 安装Google Research Football
pip install gfootball

# 安装依赖
pip install numpy
```

### 可选安装
```bash
# 如果需要渲染
sudo apt-get install python3-opencv
```

## 运行方法

### 基本运行
```bash
cd src
python run.py
```

### 直接运行
```bash
cd src  
python main.py
```

### 自定义环境参数
修改`main.py`中的`create_environment()`函数:
```python
env = football_env.create_environment(
    env_name='11_vs_11_stochastic',  # 环境名称
    render=True,                     # 是否显示画面
    write_video=False,               # 是否录制视频
    # ... 其他参数
)
```

## 配置调优

### 战术参数
在`src/gfootball_agent/config.py`中调整:
- `MID_BLOCK_X_THRESHOLD`: Mid-Block防线位置
- `PRESSURE_DISTANCE`: 上抢触发距离  
- `SHOT_RANGE`: 射门有效范围
- `TIRED_THRESHOLD`: 疲劳阈值

### 距离阈值
```python
class Distance:
    BALL_CLOSE = 0.03          # 接近球的距离
    SHORT_PASS_RANGE = 0.2     # 短传范围
    LONG_PASS_RANGE = 0.6      # 长传范围
    SHOT_RANGE = 0.3           # 射门范围
```

## 主要模块说明

### Agent类 (`agent.py`)
- 管理11名球员的决策
- 处理粘性动作逻辑
- 记录动作历史

### 决策逻辑 (`decision_logic/`)
- **top_level_logic**: 根据游戏模式分发
- **normal_mode**: 常规比赛决策
- **set_pieces**: 定位球决策

### 角色决策 (`roles/`)
每个角色都有独立的决策逻辑:
- **进攻**: 持球/无球时的行为
- **防守**: 防守站位和上抢
- **争抢**: 无人控球时的争抢

### 工具模块 (`utils/`)
- **features.py**: 计算距离、角度、最佳位置等
- **actions.py**: 管理粘性动作、验证动作合法性

## 故障排除

### 常见问题

1. **ImportError: No module named 'gfootball'**
   - 解决: 安装Google Research Football环境
   ```bash
   pip install gfootball
   ```

2. **环境创建失败**
   - 检查是否安装了所有依赖
   - 尝试使用conda环境

3. **画面无法显示**
   - 安装OpenCV: `pip install opencv-python`
   - 或设置`render=False`

4. **性能问题**
   - 关闭渲染: `render=False`
   - 调整`max_steps`参数

### 调试模式
在`main.py`中添加调试信息:
```python
# 打印更多信息
if step % 10 == 0:
    print(f"步数: {step}, 球位置: {obs[0]['ball'][:2]}")
```

## 扩展开发

### 添加新战术
1. 在`config.py`中添加新的战术参数
2. 修改各角色的决策逻辑
3. 测试和调优

### 优化算法
1. 分析`action_history`找出问题
2. 调整`features.py`中的计算逻辑
3. 优化决策树的判断条件

### 性能监控
```python
import time
start_time = time.time()
# ... 执行决策 ...
print(f"决策耗时: {time.time() - start_time:.3f}秒")
```

## 许可证
本项目基于战术分析和编程实践，仅用于学习和研究目的。 