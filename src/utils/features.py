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
    from ..gfootball_agent.config import Distance
    if distance_to_goal > Distance.SHOT_RANGE:
        return False
    
    # 检查射门角度
    shot_angle = abs(angle_to_goal(player_pos))
    from ..gfootball_agent.config import Angle
    if shot_angle > Angle.SHOT_ANGLE_THRESHOLD:
        return False
    
    return True


def get_best_pass_target(obs, player_index):
    """找到最佳的传球目标"""
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
        
        # 计算队友是否在更好的位置（更靠近对方球门）
        forward_progress = teammate_pos[0] - player_pos[0]
        
        # 检查是否有对手阻挡传球路线
        is_clear_path = check_pass_path_clear(player_pos, teammate_pos, obs['right_team'])
        
        # 综合评分
        score = 0
        if forward_progress > 0:  # 向前传球加分
            score += forward_progress * 2
        if is_clear_path:  # 路线清晰加分
            score += 1
        if is_in_opponent_half(teammate_pos):  # 在对方半场加分
            score += 1
        
        # 距离适中的传球更好
        from ..gfootball_agent.config import Distance
        if Distance.SHORT_PASS_RANGE * 0.5 < pass_distance < Distance.SHORT_PASS_RANGE:
            score += 0.5
        
        if score > best_score:
            best_score = score
            best_target = i
    
    return best_target


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
    from ..gfootball_agent.config import Tactics
    
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
    from ..gfootball_agent.config import Tactics
    
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
    from ..gfootball_agent.config import Tactics
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