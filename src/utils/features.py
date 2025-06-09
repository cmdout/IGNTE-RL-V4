"""
特征工程模块 - 计算距离、角度等派生特征
"""

import numpy as np
import math
from src.gfootball_agent.config import Field, PlayerRole


def distance_to(pos1, pos2):
    """计算两个位置之间的欧几里得距离"""
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


def angle_between_vectors(v1, v2):
    """计算两个向量之间的角度（弧度）"""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # 处理数值误差
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def angle_to_goal(player_pos, goal_x=Field.RIGHT_GOAL_X):
    """计算球员到球门的角度"""
    goal_center = [goal_x, 0]
    direction_vector = np.array(goal_center) - np.array(player_pos)
    # 计算与X轴正方向的角度
    angle = math.atan2(direction_vector[1], direction_vector[0])
    return math.degrees(angle)


def get_ball_info(obs):
    """获取球的位置和状态信息"""
    return {
        'position': obs['ball'][:2],  # 只取x,y坐标
        'direction': obs['ball_direction'][:2],
        'owned_team': obs['ball_owned_team'],
        'owned_player': obs['ball_owned_player']
    }


def get_player_info(obs, player_index):
    """获取指定球员的详细信息"""
    return {
        'position': obs['left_team'][player_index],
        'direction': obs['left_team_direction'][player_index],
        'role': obs['left_team_roles'][player_index],
        'tired_factor': obs['left_team_tired_factor'][player_index],
        'active': obs['left_team_active'][player_index]
    }


def find_closest_player(reference_pos, team_positions, exclude_indices=None):
    """找到距离参考位置最近的球员"""
    if exclude_indices is None:
        exclude_indices = []
    
    min_distance = float('inf')
    closest_index = -1
    
    for i, pos in enumerate(team_positions):
        if i in exclude_indices:
            continue
        
        dist = distance_to(reference_pos, pos)
        if dist < min_distance:
            min_distance = dist
            closest_index = i
    
    return closest_index, min_distance


def find_closest_teammate(obs, player_index):
    """找到最近的队友"""
    player_pos = obs['left_team'][player_index]
    closest_idx, closest_dist = find_closest_player(
        player_pos, obs['left_team'], exclude_indices=[player_index]
    )
    return closest_idx, closest_dist


def find_closest_opponent(obs, player_index):
    """找到最近的对手"""
    player_pos = obs['left_team'][player_index]
    closest_idx, closest_dist = find_closest_player(
        player_pos, obs['right_team']
    )
    return closest_idx, closest_dist


def is_in_opponent_half(position):
    """判断位置是否在对方半场"""
    return position[0] > Field.CENTER_X


def is_in_own_half(position):
    """判断位置是否在己方半场"""
    return position[0] < Field.CENTER_X


def can_shoot(player_pos, ball_pos, obs):
    """判断是否处于合理的射门位置"""
    # 必须在对方半场
    if not is_in_opponent_half(player_pos):
        return False
    
    # 计算到球门的距离
    goal_center = [Field.RIGHT_GOAL_X, Field.CENTER_Y]
    distance_to_goal = distance_to(player_pos, goal_center)
    
    # 距离球门太远不适合射门
    from src.gfootball_agent.config import Distance
    if distance_to_goal > Distance.SHOT_RANGE:
        return False
    
    # 检查射门角度
    shot_angle = abs(angle_to_goal(player_pos))
    from src.gfootball_agent.config import Angle
    if shot_angle > Angle.SHOT_ANGLE_THRESHOLD:
        return False
    
    return True


def get_best_pass_target(obs, player_index):
    """找到最佳的传球目标 - 优化版本，更加激进的前传"""
    player_pos = obs['left_team'][player_index]
    ball_info = get_ball_info(obs)
    
    best_target = -1
    best_score = -1
    
    for i, teammate_pos in enumerate(obs['left_team']):
        if i == player_index:  # 跳过自己
            continue
            
        if not obs['left_team_active'][i]:  # 跳过非活跃球员
            continue
        
        # 计算传球距离
        pass_distance = distance_to(player_pos, teammate_pos)
        
        # 计算向前推进的程度
        forward_progress = teammate_pos[0] - player_pos[0]
        
        # 检查传球路线是否清晰
        is_clear_path = check_pass_path_clear(player_pos, teammate_pos, obs['right_team'])
        
        # 检查接球队友周围的空间
        teammate_space = get_space_around_player(obs, i)
        
        # 获取队友角色
        teammate_role = obs['left_team_roles'][i]
        
        # 综合评分 - 大幅提高前传权重
        score = 0
        
        # 1. 向前传球奖励（显著提高权重）
        if forward_progress > 0:
            score += forward_progress * 5  # 从2提高到5
            
            # 对于显著的前传给予额外奖励
            if forward_progress > 0.2:
                score += 3
            elif forward_progress > 0.1:
                score += 1.5
                
        # 2. 严厉惩罚回传，特别是在己方半场
        elif forward_progress < 0:
            penalty = abs(forward_progress) * 3  # 回传惩罚
            if is_in_own_half(player_pos):  # 在己方半场回传惩罚更重
                penalty *= 2
            score -= penalty
            
        # 3. 高价值目标奖励
        if teammate_role == PlayerRole.CENTRAL_FORWARD:
            score += 2.5  # 传给前锋高奖励
        elif teammate_role == PlayerRole.ATTACK_MIDFIELD:
            score += 2.0  # 传给攻击中场
        elif teammate_role in [PlayerRole.LEFT_MIDFIELD, PlayerRole.RIGHT_MIDFIELD]:
            if is_in_opponent_half(teammate_pos):
                score += 1.5  # 传给在前场的边路中场
                
        # 4. 路线清晰性
        if is_clear_path:
            score += 1.5  # 提高清晰路线的奖励
        else:
            score -= 1.0  # 路线不清晰的惩罚
            
        # 5. 队友周围空间奖励
        score += teammate_space * 2  # 空间越大奖励越高
        
        # 6. 在对方半场的位置奖励
        if is_in_opponent_half(teammate_pos):
            score += 2.0  # 提高在对方半场的奖励
            
        # 7. 距离因素优化
        from src.gfootball_agent.config import Distance
        if pass_distance < Distance.SHORT_PASS_RANGE:
            if Distance.SHORT_PASS_RANGE * 0.3 < pass_distance < Distance.SHORT_PASS_RANGE * 0.8:
                score += 0.8  # 适中距离的短传
        elif pass_distance < Distance.LONG_PASS_RANGE:
            if forward_progress > 0.3:  # 只有显著前传的长传才有奖励
                score += 1.0
                
        # 8. 避免传给受压的队友
        closest_opp_to_teammate, dist_to_closest_opp = find_closest_opponent(obs, i)
        if dist_to_closest_opp < Distance.PRESSURE_DISTANCE * 1.5:
            score -= 2.0  # 队友受压惩罚
        
        if score > best_score:
            best_score = score
            best_target = i
    
    return best_target


def get_space_around_player(obs, player_index):
    """计算球员周围的空间大小"""
    player_pos = obs['left_team'][player_index]
    
    # 计算最近对手的距离
    min_distance_to_opponent = float('inf')
    for opp_pos in obs['right_team']:
        dist = distance_to(player_pos, opp_pos)
        if dist < min_distance_to_opponent:
            min_distance_to_opponent = dist
    
    # 将距离转换为空间评分（0-1）
    max_useful_distance = 0.15  # 超过这个距离就认为空间很好了
    space_score = min(min_distance_to_opponent / max_useful_distance, 1.0)
    
    return space_score


def check_dribble_space(obs, player_index, direction_vector=None):
    """
    检查球员前方是否有盘带空间
    
    参数:
        obs: 观测数据
        player_index: 球员索引
        direction_vector: 期望前进的方向向量，None表示向前
    
    返回:
        (has_space, distance_to_obstacle): 是否有空间，到障碍的距离
    """
    player_pos = obs['left_team'][player_index]
    
    # 默认向前（向对方球门方向）
    if direction_vector is None:
        direction_vector = [1.0, 0.0]  # 向右（对方球门方向）
    
    # 标准化方向向量
    direction_norm = np.linalg.norm(direction_vector)
    if direction_norm == 0:
        return False, 0.0
    
    direction_unit = np.array(direction_vector) / direction_norm
    
    # 检查前方锥形区域
    cone_angle = 30  # 度
    cone_distance = 0.1  # 检查距离
    min_safe_distance = 0.05  # 最小安全距离
    
    # 计算锥形的边界向量
    angle_rad = math.radians(cone_angle / 2)
    left_boundary = [
        direction_unit[0] * math.cos(angle_rad) - direction_unit[1] * math.sin(angle_rad),
        direction_unit[0] * math.sin(angle_rad) + direction_unit[1] * math.cos(angle_rad)
    ]
    right_boundary = [
        direction_unit[0] * math.cos(-angle_rad) - direction_unit[1] * math.sin(-angle_rad),
        direction_unit[0] * math.sin(-angle_rad) + direction_unit[1] * math.cos(-angle_rad)
    ]
    
    # 检查锥形区域内是否有对手
    min_distance_to_opponent = float('inf')
    
    for opp_pos in obs['right_team']:
        # 计算到对手的向量
        to_opponent = np.array(opp_pos) - np.array(player_pos)
        distance_to_opp = np.linalg.norm(to_opponent)
        
        if distance_to_opp == 0:
            continue
        
        # 检查对手是否在锥形范围内
        to_opponent_unit = to_opponent / distance_to_opp
        
        # 计算与方向向量的角度
        dot_product = np.dot(to_opponent_unit, direction_unit)
        if dot_product > math.cos(angle_rad):  # 在锥形范围内
            if distance_to_opp < cone_distance:
                min_distance_to_opponent = min(min_distance_to_opponent, distance_to_opp)
    
    # 判断是否有足够空间
    has_space = min_distance_to_opponent > min_safe_distance
    
    return has_space, min_distance_to_opponent


def is_safe_to_clear_ball(obs, player_index):
    """
    判断是否应该解围（在危险区域且没有好的传球选择）
    
    参数:
        obs: 观测数据
        player_index: 球员索引
    
    返回:
        should_clear: 是否应该解围
    """
    player_pos = obs['left_team'][player_index]
    
    # 检查是否在危险区域（己方禁区或接近禁区）
    danger_zone_x = Field.LEFT_GOAL_X + 0.2  # 禁区 + 缓冲区
    is_in_danger_zone = player_pos[0] < danger_zone_x
    
    if not is_in_danger_zone:
        return False
    
    # 检查是否被对手紧逼
    closest_opp_idx, closest_opp_dist = find_closest_opponent(obs, player_index)
    from src.gfootball_agent.config import Distance
    is_under_pressure = closest_opp_dist < Distance.PRESSURE_DISTANCE * 1.5
    
    if not is_under_pressure:
        return False
    
    # 检查是否有安全的传球选择
    safe_pass_found = False
    
    for i, teammate_pos in enumerate(obs['left_team']):
        if i == player_index or not obs['left_team_active'][i]:
            continue
        
        # 不能传给守门员（如果自己不是守门员）
        teammate_role = obs['left_team_roles'][i]
        if teammate_role == PlayerRole.GOALKEEPER and obs['left_team_roles'][player_index] != PlayerRole.GOALKEEPER:
            continue
        
        # 检查队友是否在安全位置
        teammate_to_goal_dist = distance_to(teammate_pos, [Field.LEFT_GOAL_X, Field.CENTER_Y])
        player_to_goal_dist = distance_to(player_pos, [Field.LEFT_GOAL_X, Field.CENTER_Y])
        
        # 队友不能比自己更接近球门（避免乌龙球）
        if teammate_to_goal_dist < player_to_goal_dist:
            continue
        
        # 检查传球距离和路线
        pass_distance = distance_to(player_pos, teammate_pos)
        if pass_distance < Distance.SHORT_PASS_RANGE:
            if check_pass_path_clear(player_pos, teammate_pos, obs['right_team']):
                safe_pass_found = True
                break
    
    # 如果没有安全传球选择，应该解围
    return not safe_pass_found


def get_clearance_target_position():
    """获取解围的目标位置（对方半场边路）"""
    # 选择对方半场的边路位置
    import random
    
    target_x = 0.6 + random.random() * 0.3  # 对方半场
    target_y = (0.3 + random.random() * 0.1) * (1 if random.random() > 0.5 else -1)  # 边路
    
    return [target_x, target_y]


def check_pass_path_clear(start_pos, end_pos, opponent_positions, threshold=0.05):
    """检查传球路径是否被对手阻挡"""
    path_vector = np.array(end_pos) - np.array(start_pos)
    path_length = np.linalg.norm(path_vector)
    
    if path_length == 0:
        return True
    
    path_unit = path_vector / path_length
    
    for opp_pos in opponent_positions:
        # 计算对手到传球路径的距离
        to_opponent = np.array(opp_pos) - np.array(start_pos)
        projection_length = np.dot(to_opponent, path_unit)
        
        # 只考虑在传球路径上的对手
        if 0 <= projection_length <= path_length:
            closest_point = np.array(start_pos) + projection_length * path_unit
            distance_to_path = distance_to(opp_pos, closest_point)
            
            if distance_to_path < threshold:
                return False
    
    return True


def get_defensive_position(obs, player_index):
    """计算防守位置"""
    ball_pos = get_ball_info(obs)['position']
    player_role = obs['left_team_roles'][player_index]
    
    # 基础防守原则：在球和己方球门之间
    goal_center = [Field.LEFT_GOAL_X, Field.CENTER_Y]
    
    if player_role == PlayerRole.GOALKEEPER:
        # 守门员特殊处理
        return get_goalkeeper_position(ball_pos)
    elif player_role in [PlayerRole.CENTRE_BACK, PlayerRole.LEFT_BACK, PlayerRole.RIGHT_BACK]:
        # 后卫防线
        return get_defender_position(ball_pos, player_role, obs)
    else:
        # 中场防守位置
        return get_midfielder_defensive_position(ball_pos, player_role, obs)


def get_goalkeeper_position(ball_pos):
    """计算守门员的防守位置"""
    # 守门员在球和球门中心的连线上，但不离开球门区域太远
    goal_center = [Field.LEFT_GOAL_X, Field.CENTER_Y]
    
    # 计算球到球门的方向向量
    ball_to_goal = np.array(goal_center) - np.array(ball_pos)
    ball_to_goal_norm = np.linalg.norm(ball_to_goal)
    
    if ball_to_goal_norm == 0:
        return goal_center
    
    ball_to_goal_unit = ball_to_goal / ball_to_goal_norm
    
    # 守门员位置：从球门中心向球的方向移动一小段距离
    keeper_distance = min(0.05, ball_to_goal_norm * 0.3)
    keeper_pos = np.array(goal_center) - keeper_distance * ball_to_goal_unit
    
    # 确保守门员不会离开球门区域太远
    keeper_pos[0] = max(keeper_pos[0], Field.LEFT_GOAL_X + 0.02)
    
    return keeper_pos.tolist()


def get_defender_position(ball_pos, player_role, obs):
    """计算后卫的防守位置"""
    from src.gfootball_agent.config import Tactics
    
    # 防线的X坐标基于球的位置动态调整
    defensive_x = min(ball_pos[0] - 0.1, Tactics.MID_BLOCK_X_THRESHOLD)
    defensive_x = max(defensive_x, Field.LEFT_GOAL_X + 0.15)  # 不能太靠近自己球门
    
    # 根据球员角色确定Y坐标
    if player_role == PlayerRole.LEFT_BACK:
        defensive_y = -Tactics.DEFENSIVE_LINE_Y_SPREAD / 2
    elif player_role == PlayerRole.RIGHT_BACK:
        defensive_y = Tactics.DEFENSIVE_LINE_Y_SPREAD / 2
    else:  # 中后卫
        # 中后卫根据球的Y位置调整
        defensive_y = ball_pos[1] * 0.3  # 轻微跟随球的Y位置
    
    return [defensive_x, defensive_y]


def get_midfielder_defensive_position(ball_pos, player_role, obs):
    """计算中场球员的防守位置"""
    from src.gfootball_agent.config import Tactics
    
    # 中场防守位置稍微靠前
    defensive_x = ball_pos[0] - 0.05
    defensive_x = max(defensive_x, Tactics.MID_BLOCK_X_THRESHOLD)
    
    # 根据球员角色调整Y位置
    if player_role == PlayerRole.LEFT_MIDFIELD:
        defensive_y = -0.2
    elif player_role == PlayerRole.RIGHT_MIDFIELD:
        defensive_y = 0.2
    else:  # 中中场
        defensive_y = ball_pos[1] * 0.5  # 更多地跟随球的位置
    
    return [defensive_x, defensive_y]


def is_player_tired(obs, player_index):
    """判断球员是否疲劳"""
    from src.gfootball_agent.config import Tactics
    return obs['left_team_tired_factor'][player_index] > Tactics.TIRED_THRESHOLD


def get_movement_direction(current_pos, target_pos):
    """计算从当前位置到目标位置的移动方向"""
    direction = np.array(target_pos) - np.array(current_pos)
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm < 0.01:  # 已经很接近目标位置
        return None
    
    # 标准化方向向量
    direction_unit = direction / direction_norm
    
    # 将方向向量转换为8个方向之一
    angle = math.atan2(direction_unit[1], direction_unit[0])
    angle_degrees = math.degrees(angle)
    
    # 将角度转换为动作
    from src.gfootball_agent.config import Action
    if -22.5 <= angle_degrees < 22.5:
        return Action.RIGHT
    elif 22.5 <= angle_degrees < 67.5:
        return Action.BOTTOM_RIGHT
    elif 67.5 <= angle_degrees < 112.5:
        return Action.BOTTOM
    elif 112.5 <= angle_degrees < 157.5:
        return Action.BOTTOM_LEFT
    elif 157.5 <= angle_degrees or angle_degrees < -157.5:
        return Action.LEFT
    elif -157.5 <= angle_degrees < -112.5:
        return Action.TOP_LEFT
    elif -112.5 <= angle_degrees < -67.5:
        return Action.TOP
    elif -67.5 <= angle_degrees < -22.5:
        return Action.TOP_RIGHT
    
    return Action.IDLE


def debug_field_visualization(obs, title="球场站位"):
    """
    在命令行输出球场站位示意图
    
    参数:
        obs: 环境观测数据
        title: 显示标题
    
    说明:
        - 数字 0-10: 左队球员（我方）
        - 字母 A-K: 右队球员（对方）
        - *: 球的位置
        - |: 球门
        - .: 空地
    """
    # 球场ASCII艺术尺寸
    field_width = 80  # 字符宽度
    field_height = 30  # 字符高度
    
    # 创建空球场
    field = [['.' for _ in range(field_width)] for _ in range(field_height)]
    
    # 添加球门
    goal_top = field_height // 2 - 2
    goal_bottom = field_height // 2 + 2
    for y in range(goal_top, goal_bottom + 1):
        field[y][0] = '|'  # 左球门
        field[y][field_width - 1] = '|'  # 右球门
    
    # 添加中线
    center_x = field_width // 2
    for y in range(field_height):
        field[y][center_x] = '|'
    
    # 添加边界线
    for x in range(field_width):
        field[0][x] = '-'  # 上边界
        field[field_height - 1][x] = '-'  # 下边界
    
    def world_to_field_coords(world_x, world_y):
        """将世界坐标转换为球场ASCII坐标"""
        # 世界坐标范围: x[-1,1], y[-0.42,0.42]
        # 转换到球场坐标: x[1,field_width-2], y[1,field_height-2]
        field_x = int((world_x - Field.LEFT_BOUNDARY) / 
                     (Field.RIGHT_BOUNDARY - Field.LEFT_BOUNDARY) * 
                     (field_width - 2)) + 1
        field_y = int((world_y - Field.TOP_BOUNDARY) / 
                     (Field.BOTTOM_BOUNDARY - Field.TOP_BOUNDARY) * 
                     (field_height - 2)) + 1
        
        # 确保坐标在有效范围内
        field_x = max(1, min(field_width - 2, field_x))
        field_y = max(1, min(field_height - 2, field_y))
        
        return field_x, field_y
    
    # 添加球的位置
    ball_pos = obs['ball'][:2]  # 只取x,y坐标
    ball_x, ball_y = world_to_field_coords(ball_pos[0], ball_pos[1])
    field[ball_y][ball_x] = '*'
    
    # 添加左队球员（我方）- 用数字0-10表示
    for i, player_pos in enumerate(obs['left_team']):
        if obs['left_team_active'][i]:  # 只显示活跃球员
            px, py = world_to_field_coords(player_pos[0], player_pos[1])
            # 如果球员位置与球重合，显示球员编号
            if field[py][px] == '*':
                field[py][px] = str(i)  # 球员优先显示
            elif field[py][px] == '.':
                field[py][px] = str(i)
    
    # 添加右队球员（对方）- 用字母A-K表示
    opponent_symbols = 'ABCDEFGHIJK'
    for i, player_pos in enumerate(obs['right_team']):
        if obs['right_team_active'][i]:  # 只显示活跃球员
            px, py = world_to_field_coords(player_pos[0], player_pos[1])
            symbol = opponent_symbols[i] if i < len(opponent_symbols) else 'X'
            if field[py][px] == '.':
                field[py][px] = symbol
    
    # 输出球场
    print(f"\n{title}")
    print("=" * field_width)
    print("左球门 ← 我方进攻方向 →  右球门")
    print("数字0-10: 我方球员, 字母A-K: 对方球员, *: 球")
    print("-" * field_width)
    
    for row in field:
        print(''.join(row))
    
    print("-" * field_width)
    
    # 输出详细信息
    ball_info = get_ball_info(obs)
    print(f"球位置: ({ball_pos[0]:.3f}, {ball_pos[1]:.3f})")
    print(f"控球方: {['无人', '我方', '对方'][ball_info['owned_team'] + 1]}")
    
    if ball_info['owned_team'] == 0:  # 我方控球
        print(f"控球球员: {ball_info['owned_player']}")
    elif ball_info['owned_team'] == 1:  # 对方控球
        print(f"控球球员: {opponent_symbols[ball_info['owned_player']]}")
    
    print(f"比分: {obs['score'][0]} - {obs['score'][1]}")
    print("=" * field_width)


def debug_player_info(obs, player_index=None):
    """
    输出球员详细信息
    
    参数:
        obs: 环境观测数据
        player_index: 指定球员索引，None则显示所有球员
    """
    role_names = {
        PlayerRole.GOALKEEPER: "守门员",
        PlayerRole.CENTRE_BACK: "中后卫", 
        PlayerRole.LEFT_BACK: "左后卫",
        PlayerRole.RIGHT_BACK: "右后卫",
        PlayerRole.DEFENCE_MIDFIELD: "防守中场",
        PlayerRole.CENTRAL_MIDFIELD: "中中场",
        PlayerRole.LEFT_MIDFIELD: "左中场",
        PlayerRole.RIGHT_MIDFIELD: "右中场",
        PlayerRole.ATTACK_MIDFIELD: "攻击中场",
        PlayerRole.CENTRAL_FORWARD: "中锋"
    }
    
    print("\n我方球员信息:")
    print("-" * 60)
    
    if player_index is not None:
        # 显示指定球员信息
        players_to_show = [player_index]
    else:
        # 显示所有球员信息
        players_to_show = range(11)
    
    for i in players_to_show:
        if obs['left_team_active'][i]:
            player_info = get_player_info(obs, i)
            role_name = role_names.get(player_info['role'], "未知角色")
            
            print(f"球员 {i}: {role_name}")
            print(f"  位置: ({player_info['position'][0]:.3f}, {player_info['position'][1]:.3f})")
            print(f"  方向: ({player_info['direction'][0]:.3f}, {player_info['direction'][1]:.3f})")
            print(f"  疲劳度: {player_info['tired_factor']:.3f}")
            print(f"  活跃状态: {'是' if player_info['active'] else '否'}")
            
            # 计算与球的距离
            ball_pos = obs['ball'][:2]
            dist_to_ball = distance_to(player_info['position'], ball_pos)
            print(f"  距球距离: {dist_to_ball:.3f}")
            print()
    
    print("-" * 60) 