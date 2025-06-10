"""
特征工程模块 - 计算距离、角度等派生特征
"""

import numpy as np
import math
from src.gfootball_agent.config import Field, PlayerRole, Distance


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


def distance_to_line(p1, p2, p3):
    """Calculate distance from point p3 to line segment [p1, p2]."""
    # p1, p2, p3 are expected to be numpy arrays
    if np.array_equal(p1,p2):
        return np.linalg.norm(p3 - p1) # Line segment is a point

    d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

    # Check if the projection of p3 onto the line defined by p1 and p2
    # falls within the segment [p1, p2].
    dot_p1_p3 = np.dot(p3 - p1, p2 - p1)
    if dot_p1_p3 <= 0: # p3 projects outside segment, beyond p1
        return np.linalg.norm(p3 - p1)

    dot_p2_p3 = np.dot(p3 - p2, p1 - p2)
    if dot_p2_p3 <= 0: # p3 projects outside segment, beyond p2
        return np.linalg.norm(p3 - p2)

    return d


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


def get_best_pass_target(obs, player_index, game_context=None):
    """找到最佳的传球目标 - 增强版，考虑助攻潜力、穿透性、风险和游戏情境"""
    player_pos = obs['left_team'][player_index]
    player_role = obs['left_team_roles'][player_index]
    
    best_target = -1
    best_score = -float('inf') # Initialize with negative infinity for better comparison

    # Game context parameters (defaults if not provided)
    # These would ideally be passed via game_context object
    is_losing_late = False # Example: True if losing in last 15 mins
    is_winning_late = False # Example: True if winning in last 15 mins
    is_counter_attack = False # Example: True if just won ball in own half

    # if game_context:
    #     is_losing_late = game_context.get('is_losing_late', False)
    #     is_winning_late = game_context.get('is_winning_late', False)
    #     is_counter_attack = game_context.get('is_counter_attack', False)

    for i, teammate_pos in enumerate(obs['left_team']):
        if i == player_index:  # 跳过自己
            continue
            
        if not obs['left_team_active'][i]:  # 跳过非活跃球员
            continue
        
        teammate_role = obs['left_team_roles'][i]
        teammate_dir = obs['left_team_direction'][i] # Teammate's facing direction

        score = 0
        
        # === Core Pass Attributes ===
        pass_distance = distance_to(player_pos, teammate_pos)
        forward_progress = teammate_pos[0] - player_pos[0]

        # === Risk Assessment ===
        # 1. Path Clarity (already exists, but let's integrate its effect more directly)
        is_clear_path = check_pass_path_clear(player_pos, teammate_pos, obs['right_team'], threshold=0.04) # Stricter threshold
        if is_clear_path:
            score += 2.5 # Increased bonus for very clear path
        else:
            score -= 3.0 # Increased penalty for unclear path

        # 2. Pressure on Receiver
        _closest_opp_to_teammate_idx, dist_to_closest_opp_to_teammate = find_closest_opponent(obs, i)
        from src.gfootball_agent.config import Distance # Ensure Distance is imported
        if dist_to_closest_opp_to_teammate < Distance.PRESSURE_DISTANCE * 1.2: # Stricter pressure check
            score -= 2.5 # Higher penalty if receiver is under immediate pressure
        elif dist_to_closest_opp_to_teammate < Distance.PRESSURE_DISTANCE * 2.0:
            score -= 1.0 # Moderate penalty

        # 3. Number of opponents near pass trajectory (more advanced interception risk)
        opponents_near_path = 0
        for opp_idx, opp_pos in enumerate(obs['right_team']):
            if obs['right_team_active'][opp_idx]:
                # Simplified check: if opponent is close to the midpoint of the pass
                mid_point_pass = ((player_pos[0] + teammate_pos[0]) / 2, (player_pos[1] + teammate_pos[1]) / 2)
                if distance_to(opp_pos, mid_point_pass) < pass_distance * 0.4: # Opponent near mid-path
                     # And also not too far from the direct line
                    if distance_to_line(np.array(player_pos), np.array(teammate_pos), np.array(opp_pos)) < 0.1:
                        opponents_near_path += 1
        score -= opponents_near_path * 1.0 # Penalize for each opponent near the path


        # === Reward Assessment ===
        # 4. Forward Progress (already exists, re-weighting)
        if forward_progress > 0.02: # Smallest forward progress still useful
            score += forward_progress * 6.0 # Maintain high reward for forwardness
            if forward_progress > 0.25: # Significant forward progress
                score += 4.0
            elif forward_progress > 0.15:
                score += 2.0
        elif forward_progress < -0.05: # More tolerance for small backward passes if they are safe
            penalty = abs(forward_progress) * 4.0
            if is_in_own_half(player_pos):
                penalty *= 2.5 # Heavy penalty for backward passes in own half
            # Allow very short safe back passes from attackers under pressure
            if player_role == PlayerRole.CENTRAL_FORWARD and forward_progress > -0.08 and dist_to_closest_opp_to_teammate > Distance.PRESSURE_DISTANCE * 2.0 :
                penalty = 0 # Forgive short safe back pass for forward
            score -= penalty
        
        # 5. High Value Target (Role-based, already exists, slight re-weight)
        if teammate_role == PlayerRole.CENTRAL_FORWARD:
            score += 3.5
        elif teammate_role == PlayerRole.ATTACK_MIDFIELD:
            score += 2.5
        elif teammate_role in [PlayerRole.LEFT_MIDFIELD, PlayerRole.RIGHT_MIDFIELD] and is_in_opponent_half(teammate_pos):
            score += 2.0
        elif teammate_role in [PlayerRole.LEFT_BACK, PlayerRole.RIGHT_BACK] and is_in_opponent_half(teammate_pos) and forward_progress > 0.1: # Overlapping run
             score += 1.5

        # 6. Receiver Space (already exists, re-weight)
        teammate_space = get_space_around_player(obs, i) # Assuming this returns 0-1 value
        score += teammate_space * 2.5

        # 7. Position on Field (already exists, re-weight)
        if is_in_opponent_half(teammate_pos):
            score += 2.5
            if teammate_pos[0] > 0.6: # Deep in opponent territory
                score += 1.5

        # 8. Pass Distance Suitability (already exists, minor adjustments)
        if pass_distance < Distance.SHORT_PASS_RANGE * 0.8: # Good short pass
             if Distance.SHORT_PASS_RANGE * 0.2 < pass_distance : # Not too short
                score += 1.0
        elif pass_distance < Distance.LONG_PASS_RANGE * 0.8: # Good long pass
            if forward_progress > 0.25: # Prefer long passes to be significantly forward
                score += 1.5
            else:
                score -= 0.5 # Discourage long sideways/backward passes unless very strategic
        else: # Too long pass
            score -= 2.0


        # === New Scoring Components ===
        # 9. Assist Potential
        # Check if the receiver is in a good shooting position or can easily get into one
        if is_in_opponent_half(teammate_pos):
            dist_receiver_to_goal = distance_to(teammate_pos, [Field.RIGHT_GOAL_X, Field.CENTER_Y])
            if dist_receiver_to_goal < Distance.SHOT_RANGE:
                # Simplified check for being able to shoot soon
                # A more complex check would involve `can_shoot` for the teammate from `teammate_pos`
                score += 3.0
                if teammate_role == PlayerRole.CENTRAL_FORWARD:
                    score += 1.5 # Extra for forward in shooting pos
            if dist_receiver_to_goal < Distance.OPTIMAL_SHOT_RANGE:
                score += 2.0


        # 10. Line Breaking Pass
        # Check if the pass crosses a 'line' of opponents.
        # Simplified: if pass starts behind an opponent and ends in front of them significantly.
        # Or if it moves the ball from one third of the pitch to another bypassing opponents.
        num_opp_bypassed = 0
        for opp_idx, opp_pos in enumerate(obs['right_team']):
            if obs['right_team_active'][opp_idx]:
                # If opponent is between passer and receiver along X-axis, and pass goes beyond them
                if min(player_pos[0], teammate_pos[0]) < opp_pos[0] < max(player_pos[0], teammate_pos[0]):
                    # And opponent is reasonably close to the direct line of the pass
                    if distance_to_line(np.array(player_pos), np.array(teammate_pos), np.array(opp_pos)) < 0.2: # 0.2 is a wider corridor
                        num_opp_bypassed +=1
        if num_opp_bypassed > 0 and forward_progress > 0.1: # Must be a forward pass
            score += num_opp_bypassed * 1.5


        # 11. Receiver Readiness (e.g., facing opponent goal)
        # Player direction is a vector [x, y]. If x > 0, mostly facing opponent goal.
        if teammate_dir[0] > 0.1 and is_in_opponent_half(teammate_pos): # Facing generally towards opponent goal
            score += 1.0
        elif teammate_dir[0] < -0.1 and is_in_own_half(teammate_pos): # Facing own goal in own half (bad)
            score -= 0.5


        # === Contextual Adjustments (Examples) ===
        # if is_counter_attack and forward_progress > 0.2:
        #    score *= 1.5 # Amplify scores for good forward passes in counter attacks
        # if is_winning_late and forward_progress < 0 and teammate_space > 0.8:
        #    score += 2.0 # Safer possession passes when winning late are more valuable
        # if is_losing_late and forward_progress > 0.1:
        #    score *= 1.3 # More desperate for forward passes

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


def calculate_dynamic_defensive_line_x(obs, base_mid_block_x_threshold):
    """Calculates a dynamic X coordinate for the defensive line."""
    ball_pos_x = obs['ball'][0]
    # score_diff = obs['score'][0] - obs['score'][1] # Our score - Opponent score
    # steps_remaining = 3000 - obs['steps_left'] # Approximation, if steps_left is from 3000 to 0

    dynamic_threshold = base_mid_block_x_threshold

    # 1. Adjust based on ball position
    if ball_pos_x > 0.5: # Ball deep in opponent half
        dynamic_threshold = base_mid_block_x_threshold + 0.15 # Push up significantly
    elif ball_pos_x > 0.0: # Ball in opponent half, near center
        dynamic_threshold = base_mid_block_x_threshold + 0.08
    elif ball_pos_x < -0.7: # Ball very deep in our half
        dynamic_threshold = base_mid_block_x_threshold - 0.1 # Drop deeper
    elif ball_pos_x < -0.4: # Ball in our defensive third
        dynamic_threshold = base_mid_block_x_threshold - 0.05

    # Placeholder for game state adjustments (e.g. score, time)
    # if is_winning_late (e.g. score_diff > 0 and steps_remaining < 600):
    #     dynamic_threshold -= 0.07 # Drop deeper by an additional amount
    # elif is_losing_late (e.g. score_diff < 0 and steps_remaining < 600):
    #     dynamic_threshold += 0.07 # Push up more

    # Ensure line doesn't go past center too much, or too deep
    dynamic_threshold = min(dynamic_threshold, Field.CENTER_X + 0.1) # Max push up
    dynamic_threshold = max(dynamic_threshold, Field.LEFT_GOAL_X + 0.25) # Min depth (don't sit on GK)

    return dynamic_threshold


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


def get_defender_position(ball_pos, player_role, obs): # obs is needed now
    """计算后卫的防守位置 - 使用动态防线"""
    from src.gfootball_agent.config import Tactics # Keep import local
    
    # Calculate dynamic defensive line X
    base_threshold = Tactics.MID_BLOCK_X_THRESHOLD
    # Pass the full obs to the dynamic line calculation function
    defensive_x = calculate_dynamic_defensive_line_x(obs, base_threshold)
    
    # Original logic for Y spread based on role
    if player_role == PlayerRole.LEFT_BACK:
        defensive_y = -Tactics.DEFENSIVE_LINE_Y_SPREAD / 2.0 # Use float division
    elif player_role == PlayerRole.RIGHT_BACK:
        defensive_y = Tactics.DEFENSIVE_LINE_Y_SPREAD / 2.0
    else:  # 中后卫
        defensive_y = ball_pos[1] * 0.4 # Slightly more reactive to ball Y for CBs
        defensive_y = np.clip(defensive_y, -0.15, 0.15) # Keep CBs relatively central

    # Final position, ensuring defenders don't get pushed too far from their base X due to ball_pos[0] effect in old code
    # The dynamic_X already considers ball_pos, so direct use is fine.
    # However, ensure it's not excessively deep.
    final_x = max(defensive_x, Field.LEFT_GOAL_X + 0.18) # Min depth for defenders
    return [final_x, defensive_y]


def get_midfielder_defensive_position(ball_pos, player_role, obs): # obs is needed now
    """计算中场球员的防守位置 - 使用动态防线"""
    from src.gfootball_agent.config import Tactics # Keep import local

    base_threshold = Tactics.MID_BLOCK_X_THRESHOLD
    # Midfielders sit slightly ahead of the defensive line or at the same dynamic line
    # Let's use a slightly more advanced X for midfielders than the pure defensive line.
    defensive_line_x = calculate_dynamic_defensive_line_x(obs, base_threshold)
    
    midfield_x = defensive_line_x + 0.05 # Midfielders generally 0.05 ahead of calculated def line
    midfield_x = min(midfield_x, ball_pos[0] - 0.03) # But should stay behind or very close to ball
    midfield_x = max(midfield_x, defensive_line_x) # Cannot be behind the main defensive line
    midfield_x = min(midfield_x, Field.CENTER_X + 0.3) # Don't push too far up when 'defending'

    # Original Y logic for midfielders
    if player_role == PlayerRole.LEFT_MIDFIELD:
        defensive_y = -0.25 # Slightly wider for wing midfielders
    elif player_role == PlayerRole.RIGHT_MIDFIELD:
        defensive_y = 0.25
    elif player_role == PlayerRole.DEFENCE_MIDFIELD: # DMF specific
        midfield_x = defensive_line_x + 0.02 # DMF closer to defensive line
        defensive_y = ball_pos[1] * 0.4
        defensive_y = np.clip(defensive_y, -0.2, 0.2)
    else:  # CENTRAL_MIDFIELD, ATTACK_MIDFIELD (when defending)
        defensive_y = ball_pos[1] * 0.5
        defensive_y = np.clip(defensive_y, -0.25, 0.25)
        if player_role == PlayerRole.ATTACK_MIDFIELD: # AM can be a bit higher
            midfield_x = min(defensive_line_x + 0.1, ball_pos[0] - 0.02)


    final_x = max(midfield_x, Field.LEFT_GOAL_X + 0.28) # Min depth for midfielders
    return [final_x, defensive_y]


def is_player_tired(obs, player_index, fatigue_threshold=None): # Added fatigue_threshold
    """判断球员是否疲劳"""
    from src.gfootball_agent.config import Tactics # Keep import local if only used here

    threshold_to_use = fatigue_threshold if fatigue_threshold is not None else Tactics.TIRED_THRESHOLD
    return obs['left_team_tired_factor'][player_index] > threshold_to_use


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