"""
中场的决策逻辑
""" 

from src.utils.features import (
    get_ball_info, get_player_info, distance_to, 
    get_midfielder_defensive_position, find_closest_teammate,
    find_closest_opponent, get_best_pass_target,
    get_movement_direction, is_player_tired,
    is_in_opponent_half, can_shoot, is_in_own_half
)
from src.utils.actions import action_manager, validate_action_for_situation
from src.gfootball_agent.config import Action, Distance, Field, PlayerRole, Tactics


def midfielder_decision(obs, player_index):
    """中场球员决策逻辑"""
    ball_info = get_ball_info(obs)
    player_info = get_player_info(obs, player_index)
    ball_owned_team = ball_info['owned_team']
    
    # 根据控球权分发决策
    if ball_owned_team == 0:  # 我方控球
        return midfielder_offensive_logic(obs, player_index, ball_info, player_info)
    elif ball_owned_team == 1:  # 对方控球
        return midfielder_defensive_logic(obs, player_index, ball_info, player_info)
    else:  # 无人控球
        return midfielder_contention_logic(obs, player_index, ball_info, player_info)


def midfielder_offensive_logic(obs, player_index, ball_info, player_info):
    """中场球员进攻逻辑"""
    player_pos = player_info['position']
    player_role = player_info['role']
    
    # 如果中场球员持球
    if ball_info['owned_player'] == player_index:
        return midfielder_with_ball_logic(obs, player_index, ball_info, player_info)
    
    # 中场球员无球时的进攻跑位
    return midfielder_offensive_movement(obs, player_index, player_info)


def midfielder_with_ball_logic(obs, player_index, ball_info, player_info):
    """中场球员持球时的决策逻辑 - 优化版本，增加盘带优先级"""
    player_pos = player_info['position']
    player_role = player_info['role']
    
    # 首先检查射门机会
    if can_shoot(player_pos, ball_info['position'], obs):
        goal_distance = distance_to(player_pos, [Field.RIGHT_GOAL_X, Field.CENTER_Y])
        if goal_distance < Distance.OPTIMAL_SHOT_RANGE:
            return Action.SHOT
    
    # 检查是否被对手逼抢
    closest_opponent_idx, closest_opponent_dist = find_closest_opponent(obs, player_index)
    
    # 如果被紧逼，快速处理球
    if closest_opponent_dist < Distance.PRESSURE_DISTANCE:
        return midfielder_under_pressure(obs, player_index, ball_info, player_info)
    
    # 优先检查盘带机会（在传球之前）
    from src.utils.features import check_dribble_space
    has_dribble_space, space_distance = check_dribble_space(obs, player_index)
    
    # 在对方半场有空间时，积极盘带
    if has_dribble_space and space_distance > 0.06:
        should_dribble = False
        
        # 在对方半场更积极盘带
        if is_in_opponent_half(player_pos):
            should_dribble = True
        # 攻击型中场即使在己方半场也要积极
        elif player_role == PlayerRole.ATTACK_MIDFIELD and space_distance > 0.08:
            should_dribble = True
        # 边路中场在边路有空间时积极盘带
        elif player_role in [PlayerRole.LEFT_MIDFIELD, PlayerRole.RIGHT_MIDFIELD]:
            if abs(player_pos[1]) > 0.15 and space_distance > 0.07:
                should_dribble = True
        
        # 比较盘带和传球价值
        if not should_dribble:
            best_target = get_best_pass_target(obs, player_index)
            if best_target != -1:
                target_pos = obs['left_team'][best_target]
                forward_progress = target_pos[0] - player_pos[0]
                # 如果传球前进不明显，优先盘带
                if forward_progress < 0.08:
                    should_dribble = True
            else:
                # 没有好的传球选择，优先盘带
                should_dribble = True
        
        if should_dribble:
            return midfielder_dribble_logic(obs, player_index, player_info)
    
    # 寻找最佳传球机会
    best_target = get_best_pass_target(obs, player_index)
    
    if best_target != -1:
        target_pos = obs['left_team'][best_target]
        target_role = obs['left_team_roles'][best_target]
        pass_distance = distance_to(player_pos, target_pos)
        forward_progress = target_pos[0] - player_pos[0]
        
        # 优先考虑向前的传球
        if target_role == PlayerRole.CENTRAL_FORWARD and forward_progress > 0.05:
            # 传给前锋，根据距离选择传球方式
            if pass_distance < Distance.SHORT_PASS_RANGE:
                return Action.SHORT_PASS
            else:
                return Action.HIGH_PASS  # 高球找前锋
        
        # 传给其他位置的队友
        if forward_progress > 0.03:  # 向前传球
            if pass_distance < Distance.SHORT_PASS_RANGE:
                return Action.SHORT_PASS
            else:
                return Action.LONG_PASS
        
        # 横传或回传保持控球
        if pass_distance < Distance.SHORT_PASS_RANGE:
            return Action.SHORT_PASS
    
    # 没有好的传球选择，考虑盘带突破
    return midfielder_dribble_logic(obs, player_index, player_info)


def midfielder_under_pressure(obs, player_index, ball_info, player_info):
    """中场球员被逼抢时的处理"""
    player_pos = player_info['position']
    
    # 寻找最近的支援队友
    closest_teammate_idx, closest_teammate_dist = find_closest_teammate(obs, player_index)
    
    if closest_teammate_idx != -1 and closest_teammate_dist < Distance.SHORT_PASS_RANGE:
        # 快速短传给最近的队友
        return Action.SHORT_PASS
    
    # 寻找安全的传球目标
    safest_target = find_safest_pass_target(obs, player_index)
    
    if safest_target != -1:
        target_pos = obs['left_team'][safest_target]
        pass_distance = distance_to(player_pos, target_pos)
        
        if pass_distance < Distance.SHORT_PASS_RANGE:
            return Action.SHORT_PASS
        else:
            return Action.LONG_PASS
    
    # 实在没有好选择，尝试盘带摆脱
    return midfielder_escape_dribble(obs, player_index, player_info)


def midfielder_dribble_logic(obs, player_index, player_info):
    """中场球员盘带逻辑"""
    player_pos = player_info['position']
    player_role = player_info['role']
    
    # 确定盘带方向
    if is_in_opponent_half(player_pos):
        # 在对方半场，可以更积极地向球门方向盘带
        target_x = min(player_pos[0] + 0.1, Field.RIGHT_GOAL_X - 0.2)
        target_y = player_pos[1]
    else:
        # 在己方半场，谨慎地向前盘带
        target_x = min(player_pos[0] + 0.08, Field.CENTER_X)
        
        # 边路中场可以沿边路盘带
        if player_role == PlayerRole.LEFT_MIDFIELD:
            target_y = min(player_pos[1], -0.15)
        elif player_role == PlayerRole.RIGHT_MIDFIELD:
            target_y = max(player_pos[1], 0.15)
        else:
            target_y = player_pos[1]
    
    target_pos = [target_x, target_y]
    
    # 开始或继续盘带
    current_sticky = obs['sticky_actions']
    if not current_sticky[9]:  # 没有在盘带
        return Action.DRIBBLE
    
    # 移动到目标位置
    movement_action = get_movement_direction(player_pos, target_pos)
    if movement_action:
        return movement_action
    
    return Action.IDLE


def midfielder_escape_dribble(obs, player_index, player_info):
    """中场球员摆脱盘带"""
    player_pos = player_info['position']
    
    # 寻找压力最小的方向
    directions = [
        [0.05, 0], [0, 0.05], [0, -0.05], [-0.05, 0],  # 四个基本方向
        [0.04, 0.04], [0.04, -0.04], [-0.04, 0.04], [-0.04, -0.04]  # 四个斜向
    ]
    
    best_direction = None
    max_space = 0
    
    for direction in directions:
        test_pos = [player_pos[0] + direction[0], player_pos[1] + direction[1]]
        
        # 检查该方向的空间
        space = calculate_space_in_direction(obs, test_pos)
        if space > max_space:
            max_space = space
            best_direction = direction
    
    if best_direction:
        target_pos = [player_pos[0] + best_direction[0], player_pos[1] + best_direction[1]]
        
        # 开始盘带
        current_sticky = obs['sticky_actions']
        if not current_sticky[9]:
            return Action.DRIBBLE
        
        movement_action = get_movement_direction(player_pos, target_pos)
        if movement_action:
            return movement_action
    
    return Action.IDLE


def midfielder_defensive_logic(obs, player_index, ball_info, player_info):
    """中场球员防守逻辑"""
    ball_pos = ball_info['position']
    player_pos = player_info['position']
    player_role = player_info['role']
    
    # 计算理想防守位置
    target_pos = get_midfielder_defensive_position(ball_pos, player_role, obs)
    
    # 检查是否需要上抢
    if should_midfielder_pressure(obs, player_index, ball_pos):
        return midfielder_pressure_logic(obs, player_index, ball_info, player_info)
    
    # 移动到防守位置
    distance_to_target = distance_to(player_pos, target_pos)
    
    if distance_to_target > 0.03:
        movement_action = get_movement_direction(player_pos, target_pos)
        
        # 如果需要快速回防
        if (distance_to_target > 0.15 and 
            ball_pos[0] > player_pos[0] and  # 球在前方
            not is_player_tired(obs, player_index)):
            return Action.SPRINT
        
        if movement_action:
            return movement_action
    
    return Action.IDLE


def midfielder_pressure_logic(obs, player_index, ball_info, player_info):
    """中场球员上抢逻辑"""
    ball_pos = ball_info['position']
    player_pos = player_info['position']
    
    distance_to_ball = distance_to(player_pos, ball_pos)
    
    # 如果非常接近球，尝试铲球
    if distance_to_ball < Distance.BALL_VERY_CLOSE:
        return Action.SLIDING
    
    # 冲刺接近球
    if distance_to_ball > 0.08 and not is_player_tired(obs, player_index):
        return Action.SPRINT
    
    # 正常接近球
    movement_action = get_movement_direction(player_pos, ball_pos)
    if movement_action:
        return movement_action
    
    return Action.IDLE


def midfielder_contention_logic(obs, player_index, ball_info, player_info):
    """中场球员争抢逻辑（无人控球时）"""
    ball_pos = ball_info['position']
    player_pos = player_info['position']
    
    distance_to_ball = distance_to(player_pos, ball_pos)
    
    # 检查是否是最接近球的中场球员
    if is_closest_midfielder_to_ball(obs, player_index, ball_pos):
        # 积极争抢球
        if distance_to_ball < Distance.BALL_CLOSE:
            movement_action = get_movement_direction(player_pos, ball_pos)
            if movement_action:
                return movement_action
        else:
            # 冲刺向球
            if not is_player_tired(obs, player_index):
                return Action.SPRINT
            else:
                movement_action = get_movement_direction(player_pos, ball_pos)
                if movement_action:
                    return movement_action
    
    # 不是最近的，移动到支援位置
    target_pos = get_contention_support_position(obs, player_index, ball_pos)
    movement_action = get_movement_direction(player_pos, target_pos)
    
    if movement_action:
        return movement_action
    
    return Action.IDLE


def midfielder_offensive_movement(obs, player_index, player_info):
    """中场球员进攻时的无球跑位"""
    player_pos = player_info['position']
    player_role = player_info['role']
    ball_info = get_ball_info(obs)
    ball_pos = ball_info['position']
    
    # 根据角色确定跑位策略
    if player_role == PlayerRole.ATTACK_MIDFIELD:
        return attacking_midfielder_movement(obs, player_index, player_info, ball_pos)
    elif player_role in [PlayerRole.LEFT_MIDFIELD, PlayerRole.RIGHT_MIDFIELD]:
        return wing_midfielder_movement(obs, player_index, player_info, ball_pos)
    else:  # CENTRAL_MIDFIELD
        return central_midfielder_movement(obs, player_index, player_info, ball_pos)


def attacking_midfielder_movement(obs, player_index, player_info, ball_pos):
    """攻击型中场的跑位"""
    player_pos = player_info['position']
    
    if is_in_opponent_half(ball_pos):
        # 球在对方半场，积极前插寻找机会
        target_x = min(ball_pos[0] + 0.1, Field.RIGHT_GOAL_X - 0.15)
        target_y = ball_pos[1] * 0.7  # 跟随球的横向位置
    else:
        # 球在己方半场，保持中等位置准备接应
        target_x = max(ball_pos[0] + 0.05, Field.CENTER_X - 0.1)
        target_y = player_pos[1]
    
    target_pos = [target_x, target_y]
    movement_action = get_movement_direction(player_pos, target_pos)
    
    if movement_action:
        return movement_action
    
    return Action.IDLE


def wing_midfielder_movement(obs, player_index, player_info, ball_pos):
    """边路中场的跑位"""
    player_pos = player_info['position']
    player_role = player_info['role']
    
    # 确定边路方向
    target_y = -0.25 if player_role == PlayerRole.LEFT_MIDFIELD else 0.25
    
    if is_in_opponent_half(ball_pos):
        # 在对方半场时，边路中场可以更积极
        target_x = min(ball_pos[0] + 0.05, Field.RIGHT_GOAL_X - 0.2)
        
        # 如果球在中路，边路球员应该拉边
        if abs(ball_pos[1]) < 0.1:
            target_y = target_y  # 保持在边路
        else:
            # 如果球在同侧边路，可以内切支援
            if (player_role == PlayerRole.LEFT_MIDFIELD and ball_pos[1] < -0.1) or \
               (player_role == PlayerRole.RIGHT_MIDFIELD and ball_pos[1] > 0.1):
                target_y = ball_pos[1] * 0.5  # 向中路移动
    else:
        # 在己方半场时保持位置
        target_x = max(ball_pos[0] + 0.05, Tactics.MID_BLOCK_X_THRESHOLD + 0.05)
    
    target_pos = [target_x, target_y]
    movement_action = get_movement_direction(player_pos, target_pos)
    
    if movement_action:
        return movement_action
    
    return Action.IDLE


def central_midfielder_movement(obs, player_index, player_info, ball_pos):
    """中中场的跑位"""
    player_pos = player_info['position']
    
    # 中中场的位置调整更保守
    if is_in_opponent_half(ball_pos):
        # 适度前压，但不能太激进
        target_x = min(ball_pos[0], Field.CENTER_X + 0.1)
    else:
        # 在己方半场时，保持在中场中心区域
        target_x = max(ball_pos[0] - 0.05, Tactics.MID_BLOCK_X_THRESHOLD)
    
    # Y坐标跟随球的位置，但幅度较小
    target_y = ball_pos[1] * 0.3
    
    target_pos = [target_x, target_y]
    movement_action = get_movement_direction(player_pos, target_pos)
    
    if movement_action:
        return movement_action
    
    return Action.IDLE


def should_midfielder_pressure(obs, player_index, ball_pos):
    """判断中场球员是否应该上抢"""
    player_pos = obs['left_team'][player_index]
    distance_to_ball = distance_to(player_pos, ball_pos)
    
    # 只有最接近球的中场球员才上抢
    if not is_closest_midfielder_to_ball(obs, player_index, ball_pos):
        return False
    
    # 距离要合适
    return distance_to_ball < Distance.PRESSURE_DISTANCE * 1.5


def is_closest_midfielder_to_ball(obs, player_index, ball_pos):
    """判断是否是离球最近的中场球员"""
    player_pos = obs['left_team'][player_index]
    player_distance = distance_to(player_pos, ball_pos)
    
    midfielder_roles = [
        PlayerRole.CENTRAL_MIDFIELD, PlayerRole.LEFT_MIDFIELD, 
        PlayerRole.RIGHT_MIDFIELD, PlayerRole.ATTACK_MIDFIELD
    ]
    
    for i, pos in enumerate(obs['left_team']):
        if i == player_index:
            continue
        
        role = obs['left_team_roles'][i]
        if role in midfielder_roles:
            if distance_to(pos, ball_pos) < player_distance:
                return False
    
    return True


def find_safest_pass_target(obs, player_index):
    """寻找最安全的传球目标"""
    player_pos = obs['left_team'][player_index]
    safest_target = -1
    max_safety_score = 0
    
    for i, teammate_pos in enumerate(obs['left_team']):
        if i == player_index or not obs['left_team_active'][i]:
            continue
        
        # 计算安全性评分
        closest_opp_idx, dist_to_closest_opp = find_closest_opponent(obs, i)
        pass_distance = distance_to(player_pos, teammate_pos)
        
        # 安全性评分：距离对手越远越安全，传球距离适中更好
        safety_score = dist_to_closest_opp
        if Distance.SHORT_PASS_RANGE * 0.5 < pass_distance < Distance.SHORT_PASS_RANGE:
            safety_score += 0.02
        
        if safety_score > max_safety_score:
            max_safety_score = safety_score
            safest_target = i
    
    return safest_target


def calculate_space_in_direction(obs, test_pos):
    """计算指定位置周围的空间大小"""
    min_distance_to_opponent = float('inf')
    
    for opp_pos in obs['right_team']:
        dist = distance_to(test_pos, opp_pos)
        if dist < min_distance_to_opponent:
            min_distance_to_opponent = dist
    
    return min_distance_to_opponent


def get_contention_support_position(obs, player_index, ball_pos):
    """获取争抢时的支援位置"""
    player_role = obs['left_team_roles'][player_index]
    
    # 根据角色确定支援位置
    if player_role == PlayerRole.ATTACK_MIDFIELD:
        # 攻击型中场在球的前方支援
        target_x = ball_pos[0] + 0.05
        target_y = ball_pos[1]
    elif player_role in [PlayerRole.LEFT_MIDFIELD, PlayerRole.RIGHT_MIDFIELD]:
        # 边路中场在边路支援
        target_x = ball_pos[0]
        target_y = -0.2 if player_role == PlayerRole.LEFT_MIDFIELD else 0.2
    else:
        # 中中场在球的后方支援
        target_x = ball_pos[0] - 0.05
        target_y = ball_pos[1]
    
    return [target_x, target_y] 