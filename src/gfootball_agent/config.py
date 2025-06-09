"""
足球智能体配置文件 - 存放所有配置、常量和阈值
"""

# ===================== 游戏模式常量 =====================
class GameMode:
    NORMAL = 0
    KICK_OFF = 1
    GOAL_KICK = 2
    FREE_KICK = 3
    CORNER = 4
    THROW_IN = 5
    PENALTY = 6

# ===================== 球员角色常量 =====================
class PlayerRole:
    GOALKEEPER = 0  # 守门员
    CENTRE_BACK = 1  # 中后卫
    LEFT_BACK = 2  # 左后卫
    RIGHT_BACK = 3  # 右后卫
    DEFENCE_MIDFIELD = 4  # 防守中场
    CENTRAL_MIDFIELD = 5  # 中中场
    LEFT_MIDFIELD = 6  # 左中场
    RIGHT_MIDFIELD = 7  # 右中场
    ATTACK_MIDFIELD = 8  # 攻击中场
    CENTRAL_FORWARD = 9  # 中锋

# ===================== 动作常量 =====================
class Action:
    IDLE = 0
    LEFT = 1
    TOP_LEFT = 2
    TOP = 3
    TOP_RIGHT = 4
    RIGHT = 5
    BOTTOM_RIGHT = 6
    BOTTOM = 7
    BOTTOM_LEFT = 8
    LONG_PASS = 9
    HIGH_PASS = 10
    SHORT_PASS = 11
    SHOT = 12
    SPRINT = 13
    RELEASE_DIRECTION = 14
    RELEASE_SPRINT = 15
    SLIDING = 16
    DRIBBLE = 17
    RELEASE_DRIBBLE = 18

# ===================== 场地坐标常量 =====================
class Field:
    # 场地边界
    LEFT_BOUNDARY = -1.0
    RIGHT_BOUNDARY = 1.0
    TOP_BOUNDARY = -0.42
    BOTTOM_BOUNDARY = 0.42
    
    # 球门位置
    LEFT_GOAL_X = -1.0
    RIGHT_GOAL_X = 1.0
    GOAL_TOP_Y = -0.044
    GOAL_BOTTOM_Y = 0.044
    
    # 关键区域
    CENTER_X = 0.0
    CENTER_Y = 0.0

# ===================== 距离阈值常量 =====================
class Distance:
    # 基础距离阈值
    BALL_CLOSE = 0.03  # 认为接近球的距离阈值
    BALL_VERY_CLOSE = 0.015  # 认为非常接近球的距离阈值
    TEAMMATE_CLOSE = 0.05  # 认为接近队友的距离
    OPPONENT_CLOSE = 0.04  # 认为接近对手的距离
    
    # 传球距离阈值
    SHORT_PASS_RANGE = 0.2  # 短传有效范围
    LONG_PASS_RANGE = 0.6  # 长传有效范围
    
    # 射门距离阈值
    SHOT_RANGE = 0.3  # 射门有效范围
    OPTIMAL_SHOT_RANGE = 0.15  # 最佳射门范围
    
    # 防守距离阈值
    PRESSURE_DISTANCE = 0.05  # 上抢压迫距离
    DEFENSIVE_LINE_DISTANCE = 0.1  # 防线紧凑性距离

# ===================== 角度阈值常量 =====================
class Angle:
    SHOT_ANGLE_THRESHOLD = 30  # 射门角度阈值（度）
    PASS_ANGLE_THRESHOLD = 45  # 传球角度阈值（度）

# ===================== 战术参数 =====================
class Tactics:
    # Mid-Block防守参数
    MID_BLOCK_X_THRESHOLD = -0.2  # Mid-Block防守的X坐标阈值
    DEFENSIVE_LINE_Y_SPREAD = 0.3  # 防线的Y轴展开范围
    
    # 进攻参数
    ATTACK_X_THRESHOLD = 0.2  # 进攻区域的X坐标阈值
    COUNTER_ATTACK_TRIGGER = 0.8  # 反击触发的控球权转换速度
    
    # 体能管理
    TIRED_THRESHOLD = 0.5  # 疲劳阈值
    SPRINT_ENERGY_CONSERVATION = 0.3  # 冲刺体能保护阈值

# ===================== 位置映射 =====================
class PositionMapping:
    """球员索引与角色的映射关系"""
    # 基于scenario_config.md的初始配置
    ROLE_TO_INDICES = {
        PlayerRole.GOALKEEPER: [0],
        PlayerRole.LEFT_BACK: [3],
        PlayerRole.CENTRE_BACK: [4, 5],
        PlayerRole.RIGHT_BACK: [6],
        PlayerRole.RIGHT_MIDFIELD: [1],
        PlayerRole.CENTRAL_MIDFIELD: [7, 8, 9],
        PlayerRole.LEFT_MIDFIELD: [10],
        PlayerRole.CENTRAL_FORWARD: [2]
    }
    
    @staticmethod
    def get_role_by_index(index, obs):
        """根据观测数据获取球员角色"""
        return obs['left_team_roles'][index]

# ===================== 粘性动作管理 =====================
class StickyActions:
    """粘性动作索引映射"""
    LEFT = 0
    TOP_LEFT = 1
    TOP = 2
    TOP_RIGHT = 3
    RIGHT = 4
    BOTTOM_RIGHT = 5
    BOTTOM = 6
    BOTTOM_LEFT = 7
    SPRINT = 8
    DRIBBLE = 9
    
    MOVEMENT_ACTIONS = [LEFT, TOP_LEFT, TOP, TOP_RIGHT, RIGHT, BOTTOM_RIGHT, BOTTOM, BOTTOM_LEFT] 