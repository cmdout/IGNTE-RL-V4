"""
后卫的决策逻辑
""" 

from src.utils.features import (
    get_ball_info, get_player_info, distance_to, 
    get_defensive_position, find_closest_teammate,
    find_closest_opponent, get_best_pass_target,
    get_movement_direction, is_player_tired,
    is_in_opponent_half, can_shoot, debug_field_visualization
)
from src.utils.actions import action_manager, validate_action_for_situation
from src.gfootball_agent.config import Action, Distance, Field, PlayerRole, Tactics


def defender_decision(obs, player_index):
    """后卫决策逻辑"""
    ball_info = get_ball_info(obs)
    player_info = get_player_info(obs, player_index)
    ball_owned_team = ball_info['owned_team']
    
    # 根据控球权分发决策
    if ball_owned_team == 0:  # 我方控球
        return defender_offensive_logic(obs, player_index, ball_info, player_info)
    elif ball_owned_team == 1:  # 对方控球
        return defender_defensive_logic(obs, player_index, ball_info, player_info)
    else:  # 无人控球
        return defender_contention_logic(obs, player_index, ball_info, player_info)


def defender_offensive_logic(obs, player_index, ball_info, player_info):
    """后卫进攻逻辑"""
    player_pos = player_info['position']
    player_role = player_info['role']
    
    # 如果后卫持球
    if ball_info['owned_player'] == player_index:
        return defender_with_ball_logic(obs, player_index, ball_info, player_info)
    
    # 后卫无球时的进攻支援
    # 检查是否应该前插助攻
    if should_defender_attack(obs, player_index, player_role):
        return defender_attacking_movement(obs, player_index, player_info)
    else:
        # 保持安全位置，准备接应传球
        return defender_support_movement(obs, player_index, player_info)


def defender_with_ball_logic(obs, player_index, ball_info, player_info):
    """后卫持球时的决策逻辑 - 优化版本，优先考虑盘带"""
    player_pos = player_info['position']
    player_role = player_info['role']
    
    # 检查是否被对手逼抢
    closest_opponent_idx, closest_opponent_dist = find_closest_opponent(obs, player_index)
    
    # 如果被紧逼，优先安全出球
    if closest_opponent_dist < Distance.PRESSURE_DISTANCE:
        return defender_under_pressure(obs, player_index, ball_info, player_info)
    
    # 优先检查前方是否有盘带空间
    from src.utils.features import check_dribble_space
    has_dribble_space, space_distance = check_dribble_space(obs, player_index)
    
    # 如果前方有空间且不在危险区域，优先盘带
    danger_zone_x = Field.LEFT_GOAL_X + 0.15  # 禁区附近为危险区域
    if has_dribble_space and player_pos[0] > danger_zone_x:
        # 检查盘带是否比传球更有价值
        best_target = get_best_pass_target(obs, player_index)
        
        should_dribble = False
        
        if best_target == -1:
            # 没有好的传球选择，果断盘带
            should_dribble = True
        else:
            # 比较盘带和传球的价值
            target_pos = obs['left_team'][best_target]
            forward_progress_pass = target_pos[0] - player_pos[0]
            
            # 如果传球的前进幅度不大，优先盘带
            if forward_progress_pass < 0.12:  # 传球前进不明显
                should_dribble = True
            
            # 边后卫在边路有空间时，积极盘带助攻
            if player_role in [PlayerRole.LEFT_BACK, PlayerRole.RIGHT_BACK]:
                if abs(player_pos[1]) > 0.15 and space_distance > 0.08:
                    should_dribble = True
        
        if should_dribble:
            return defender_dribble_forward(obs, player_index, player_info)
    
    # 寻找最佳传球目标
    best_target = get_best_pass_target(obs, player_index)

    # debug_field_visualization(obs, title="后卫持球时的决策逻辑")
    
    if best_target != -1:
        target_pos = obs['left_team'][best_target]
        pass_distance = distance_to(player_pos, target_pos)
        target_role = obs['left_team_roles'][best_target]
        
        # 优先向前传球
        forward_progress = target_pos[0] - player_pos[0]
        
        if forward_progress > 0.1:  # 显著向前的传球
            if pass_distance < Distance.SHORT_PASS_RANGE:
                return Action.SHORT_PASS
            else:
                return Action.HIGH_PASS  # 高球传向前场
        elif pass_distance < Distance.SHORT_PASS_RANGE:
            # 横传或回传
            return Action.SHORT_PASS
    
    # 没有好的传球选择，带球前进寻找机会
    return defender_dribble_forward(obs, player_index, player_info)


def defender_under_pressure(obs, player_index, ball_info, player_info):
    """后卫被逼抢时的处理 - 优化版本，避免乌龙球"""
    player_pos = player_info['position']
    
    # 首先检查是否在危险区域，应该解围
    from src.utils.features import is_safe_to_clear_ball
    if is_safe_to_clear_ball(obs, player_index):
        # 解围到对方半场边路
        return Action.HIGH_PASS  # 高球解围
    
    # 寻找安全的传球目标，严格避免乌龙球
    best_target = -1
    best_score = -1
    
    for i, teammate_pos in enumerate(obs['left_team']):
        if i == player_index or not obs['left_team_active'][i]:
            continue
        
        teammate_role = obs['left_team_roles'][i]
        
        # 计算安全评分
        score = 0
        
        # 1. 绝对不传给守门员（除非自己是守门员）
        if teammate_role == PlayerRole.GOALKEEPER and player_info['role'] != PlayerRole.GOALKEEPER:
            continue
        
        # 2. 绝对不传给比自己更接近球门的队友
        teammate_to_goal_dist = distance_to(teammate_pos, [Field.LEFT_GOAL_X, Field.CENTER_Y])
        player_to_goal_dist = distance_to(player_pos, [Field.LEFT_GOAL_X, Field.CENTER_Y])
        
        if teammate_to_goal_dist <= player_to_goal_dist + 0.02:
            continue  # 跳过太接近球门的队友
        
        # 3. 队友距离对手的远近（安全性）
        closest_opp_to_teammate, dist_to_opp = find_closest_opponent(obs, i)
        score += dist_to_opp * 5  # 距离对手越远越安全
        
        # 4. 队友距离球门越远越安全
        score += teammate_to_goal_dist * 3
        
        # 5. 优先传给边路队友（边路相对安全）
        if abs(teammate_pos[1]) > 0.2:
            score += 1.0
        
        # 6. 不要传给Y位置太接近的队友（可能造成拥挤）
        if abs(teammate_pos[1] - player_pos[1]) < 0.1:
            score -= 0.5
        
        # 7. 传球距离不能太远（紧急情况）
        pass_distance = distance_to(player_pos, teammate_pos)
        if pass_distance > Distance.SHORT_PASS_RANGE * 1.5:
            score -= 1.0
        
        # 8. 检查传球路线
        from src.utils.features import check_pass_path_clear
        if not check_pass_path_clear(player_pos, teammate_pos, obs['right_team']):
            score -= 2.0  # 路线不清晰严重扣分
        
        if score > best_score:
            best_score = score
            best_target = i
    
    # 有安全的传球目标
    if best_target != -1 and best_score > 0:
        target_pos = obs['left_team'][best_target]
        pass_distance = distance_to(player_pos, target_pos)
        
        if pass_distance < Distance.SHORT_PASS_RANGE:
            return Action.SHORT_PASS
        else:
            return Action.LONG_PASS
    
    # 没有安全传球目标，必须解围
    return Action.HIGH_PASS  # 高球解围到前场


def defender_dribble_forward(obs, player_index, player_info):
    """后卫带球前进"""
    player_pos = player_info['position']
    player_role = player_info['role']
    
    # 确定前进目标位置
    if player_role == PlayerRole.CENTRE_BACK:
        # 中后卫谨慎前进，不超过中场线
        target_x = min(player_pos[0] + 0.1, Tactics.MID_BLOCK_X_THRESHOLD)
        target_y = player_pos[1]  # 保持在中路
    else:
        # 边后卫可以沿边路更积极地前进
        target_x = min(player_pos[0] + 0.15, Field.CENTER_X)
        
        # 保持在边路
        if player_role == PlayerRole.LEFT_BACK:
            target_y = min(player_pos[1], -0.2)
        else:  # RIGHT_BACK
            target_y = max(player_pos[1], 0.2)
    
    target_pos = [target_x, target_y]
    
    # 开始盘带
    current_sticky = obs['sticky_actions']
    if not current_sticky[9]:  # 没有在盘带
        return Action.DRIBBLE
    
    # 移动到目标位置
    movement_action = get_movement_direction(player_pos, target_pos)
    if movement_action:
        return movement_action
    
    return Action.IDLE


def defender_defensive_logic(obs, player_index, ball_info, player_info):
    """后卫防守逻辑"""
    ball_pos = ball_info['position']
    player_pos = player_info['position']
    player_role = player_info['role']
    
    # 计算理想防守位置
    target_pos = get_defensive_position(obs, player_index)
    
    # 检查是否是离球最近的后卫，需要上抢
    if should_defender_pressure(obs, player_index, ball_pos):
        return defender_pressure_logic(obs, player_index, ball_info, player_info)
    
    # 正常防守站位
    distance_to_target = distance_to(player_pos, target_pos)
    
    if distance_to_target > 0.02:  # 需要调整位置
        movement_action = get_movement_direction(player_pos, target_pos)
        
        # 如果需要快速回防，使用冲刺
        if (distance_to_target > 0.1 and 
            ball_pos[0] > player_pos[0] and  # 球在前方
            not is_player_tired(obs, player_index)):
            return Action.SPRINT
        
        if movement_action:
            return movement_action
    
    # 保持防守站位
    return Action.IDLE


def defender_pressure_logic(obs, player_index, ball_info, player_info):
    """后卫上抢逻辑"""
    ball_pos = ball_info['position']
    player_pos = player_info['position']
    
    # 计算到球的距离
    distance_to_ball = distance_to(player_pos, ball_pos)
    
    # 如果非常接近球，尝试铲球
    if distance_to_ball < Distance.BALL_VERY_CLOSE:
        return Action.SLIDING
    
    # 如果接近球，谨慎接近
    if distance_to_ball < Distance.BALL_CLOSE:
        movement_action = get_movement_direction(player_pos, ball_pos)
        if movement_action:
            return movement_action
    
    # 冲刺接近球
    if distance_to_ball > 0.05 and not is_player_tired(obs, player_index):
        return Action.SPRINT
    
    # 正常速度接近
    movement_action = get_movement_direction(player_pos, ball_pos)
    if movement_action:
        return movement_action
    
    return Action.IDLE


def defender_contention_logic(obs, player_index, ball_info, player_info):
    """后卫争抢逻辑（无人控球时）"""
    ball_pos = ball_info['position']
    player_pos = player_info['position']
    
    # 计算到球的距离
    distance_to_ball = distance_to(player_pos, ball_pos)
    
    # 检查是否是最接近球的后卫
    if is_closest_defender_to_ball(obs, player_index, ball_pos):
        # 积极争抢球
        if distance_to_ball < Distance.BALL_CLOSE:
            movement_action = get_movement_direction(player_pos, ball_pos)
            if movement_action:
                return movement_action
        else:
            # 冲刺向球
            return Action.SPRINT
    
    # 不是最近的，保持防守位置
    target_pos = get_defensive_position(obs, player_index)
    movement_action = get_movement_direction(player_pos, target_pos)
    
    if movement_action:
        return movement_action
    
    return Action.IDLE


def should_defender_attack(obs, player_index, player_role):
    """判断后卫是否应该前插助攻"""
    ball_info = get_ball_info(obs)
    ball_pos = ball_info['position']
    
    # 只有边后卫在特定情况下才前插
    if player_role not in [PlayerRole.LEFT_BACK, PlayerRole.RIGHT_BACK]:
        return False
    
    # 球必须在对方半场
    if not is_in_opponent_half(ball_pos):
        return False
    
    # 检查防线是否安全（有足够的中后卫覆盖）
    centre_backs_in_position = count_centre_backs_in_position(obs)
    if centre_backs_in_position < 2:
        return False
    
    # 检查该边路是否需要支援
    return check_flank_needs_support(obs, player_index, player_role)


def defender_attacking_movement(obs, player_index, player_info):
    """后卫前插助攻的移动"""
    player_pos = player_info['position']
    player_role = player_info['role']
    
    # 沿边路前插
    target_x = min(player_pos[0] + 0.2, Tactics.ATTACK_X_THRESHOLD)
    
    if player_role == PlayerRole.LEFT_BACK:
        target_y = -0.25  # 左边路
    else:  # RIGHT_BACK
        target_y = 0.25   # 右边路
    
    target_pos = [target_x, target_y]
    movement_action = get_movement_direction(player_pos, target_pos)
    
    # 积极前进，使用冲刺
    if distance_to(player_pos, target_pos) > 0.1 and not is_player_tired(obs, player_index):
        return Action.SPRINT
    
    if movement_action:
        return movement_action
    
    return Action.IDLE


def defender_support_movement(obs, player_index, player_info):
    """后卫支援移动（保持安全位置）"""
    player_pos = player_info['position']
    ball_info = get_ball_info(obs)
    ball_pos = ball_info['position']
    
    # 保持在安全的支援位置
    target_x = max(ball_pos[0] - 0.15, Field.LEFT_GOAL_X + 0.2)
    target_y = player_pos[1]  # 保持当前Y位置
    
    target_pos = [target_x, target_y]
    movement_action = get_movement_direction(player_pos, target_pos)
    
    if movement_action:
        return movement_action
    
    return Action.IDLE


def should_defender_pressure(obs, player_index, ball_pos):
    """判断后卫是否应该上抢"""
    player_pos = obs['left_team'][player_index]
    
    # 检查是否是离球最近的后卫
    min_distance = float('inf')
    closest_defender = -1
    
    for i, pos in enumerate(obs['left_team']):
        role = obs['left_team_roles'][i]
        if role in [PlayerRole.CENTRE_BACK, PlayerRole.LEFT_BACK, PlayerRole.RIGHT_BACK]:
            dist = distance_to(pos, ball_pos)
            if dist < min_distance:
                min_distance = dist
                closest_defender = i
    
    return closest_defender == player_index and min_distance < Distance.PRESSURE_DISTANCE * 2


def is_closest_defender_to_ball(obs, player_index, ball_pos):
    """判断是否是离球最近的后卫"""
    player_pos = obs['left_team'][player_index]
    player_distance = distance_to(player_pos, ball_pos)
    
    for i, pos in enumerate(obs['left_team']):
        if i == player_index:
            continue
        
        role = obs['left_team_roles'][i]
        if role in [PlayerRole.CENTRE_BACK, PlayerRole.LEFT_BACK, PlayerRole.RIGHT_BACK]:
            if distance_to(pos, ball_pos) < player_distance:
                return False
    
    return True


def count_centre_backs_in_position(obs):
    """统计在位的中后卫数量"""
    count = 0
    for i, pos in enumerate(obs['left_team']):
        role = obs['left_team_roles'][i]
        if role == PlayerRole.CENTRE_BACK:
            # 检查是否在合理的防守位置
            if pos[0] < Tactics.MID_BLOCK_X_THRESHOLD:
                count += 1
    return count


def check_flank_needs_support(obs, player_index, player_role):
    """检查边路是否需要支援"""
    # 简化判断：检查该边路的中场球员位置
    player_pos = obs['left_team'][player_index]
    
    if player_role == PlayerRole.LEFT_BACK:
        # 检查左中场位置
        for i, pos in enumerate(obs['left_team']):
            role = obs['left_team_roles'][i]
            if role == PlayerRole.LEFT_MIDFIELD:
                # 如果左中场前压很多，边后卫可以考虑支援
                if pos[0] > Tactics.ATTACK_X_THRESHOLD:
                    return True
    else:  # RIGHT_BACK
        # 检查右中场位置
        for i, pos in enumerate(obs['left_team']):
            role = obs['left_team_roles'][i]
            if role == PlayerRole.RIGHT_MIDFIELD:
                if pos[0] > Tactics.ATTACK_X_THRESHOLD:
                    return True
    
    return False 