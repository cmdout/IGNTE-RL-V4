"""
守门员的决策逻辑
""" 

from src.utils.features import (
    get_ball_info, get_player_info, distance_to, 
    get_goalkeeper_position, find_closest_teammate,
    find_closest_opponent, get_best_pass_target,
    get_movement_direction, is_player_tired
)
from src.utils.actions import action_manager, validate_action_for_situation
from src.gfootball_agent.config import Action, Distance, Field, PlayerRole


def goalkeeper_decision(obs, player_index):
    """守门员决策逻辑"""
    ball_info = get_ball_info(obs)
    player_info = get_player_info(obs, player_index)
    ball_owned_team = ball_info['owned_team']
    
    # 根据控球权分发决策
    if ball_owned_team == 0:  # 我方控球
        return goalkeeper_offensive_logic(obs, player_index, ball_info, player_info)
    elif ball_owned_team == 1:  # 对方控球
        return goalkeeper_defensive_logic(obs, player_index, ball_info, player_info)
    else:  # 无人控球
        return goalkeeper_contention_logic(obs, player_index, ball_info, player_info)


def goalkeeper_offensive_logic(obs, player_index, ball_info, player_info):
    """守门员进攻逻辑"""
    ball_pos = ball_info['position']
    player_pos = player_info['position']
    
    # 如果守门员持球
    if ball_info['owned_player'] == player_index:
        return goalkeeper_with_ball_logic(obs, player_index, ball_info, player_info)
    
    # 守门员无球时的进攻支援
    # 主要任务：保持合理位置，准备接应传球
    target_pos = get_goalkeeper_offensive_position(ball_pos)
    movement_action = get_movement_direction(player_pos, target_pos)
    
    if movement_action:
        return movement_action
    
    return Action.IDLE


def goalkeeper_with_ball_logic(obs, player_index, ball_info, player_info):
    """守门员持球时的决策逻辑 - 优化版本，避免危险传球"""
    player_pos = player_info['position']
    
    # 检查是否被对手紧逼
    closest_opponent_idx, closest_opponent_dist = find_closest_opponent(obs, player_index)
    is_under_pressure = closest_opponent_dist < Distance.PRESSURE_DISTANCE
    
    # 如果受压，优先检查是否应该解围
    if is_under_pressure:
        from src.utils.features import is_safe_to_clear_ball, get_clearance_target_position
        if is_safe_to_clear_ball(obs, player_index):
            # 解围到对方半场边路
            return Action.LONG_PASS
    
    # 寻找安全的传球目标
    best_target = get_safe_goalkeeper_pass_target(obs, player_index)
    
    if best_target != -1:
        target_pos = obs['left_team'][best_target]
        pass_distance = distance_to(player_pos, target_pos)
        target_role = obs['left_team_roles'][best_target]
        
        # 检查目标队友是否也受压
        target_closest_opp_idx, target_closest_opp_dist = find_closest_opponent(obs, best_target)
        target_under_pressure = target_closest_opp_dist < Distance.PRESSURE_DISTANCE * 1.2
        
        # 如果目标队友受压，考虑其他选择
        if target_under_pressure and not is_under_pressure:
            # 守门员没受压但队友受压，寻找其他目标或长传
            alternative_target = find_alternative_pass_target(obs, player_index, exclude=[best_target])
            if alternative_target != -1:
                return Action.LONG_PASS  # 长传给替代目标
        
        # 正常传球决策
        if pass_distance < Distance.SHORT_PASS_RANGE and not target_under_pressure:
            # 短传给后卫（确保后卫不受压）
            return Action.SHORT_PASS
        elif target_role in [PlayerRole.CENTRAL_MIDFIELD, PlayerRole.LEFT_MIDFIELD, PlayerRole.RIGHT_MIDFIELD]:
            # 长传到中场
            return Action.LONG_PASS
    
    # 没有好的传球选择，持球移动寻找机会
    return goalkeeper_move_with_ball(obs, player_index, player_info)


def get_safe_goalkeeper_pass_target(obs, player_index):
    """为守门员寻找安全的传球目标，避免乌龙球"""
    player_pos = obs['left_team'][player_index]
    
    best_target = -1
    best_score = -1
    
    for i, teammate_pos in enumerate(obs['left_team']):
        if i == player_index or not obs['left_team_active'][i]:
            continue
        
        teammate_role = obs['left_team_roles'][i]
        
        # 计算基础安全评分
        score = 0
        
        # 1. 绝对不传给距离己方球门更近的队友（防止乌龙球）
        teammate_to_goal_dist = distance_to(teammate_pos, [Field.LEFT_GOAL_X, Field.CENTER_Y])
        keeper_to_goal_dist = distance_to(player_pos, [Field.LEFT_GOAL_X, Field.CENTER_Y])
        
        if teammate_to_goal_dist <= keeper_to_goal_dist + 0.02:  # 加小的缓冲
            continue  # 跳过太接近球门的队友
        
        # 2. 队友距离己方球门越远越安全
        score += teammate_to_goal_dist * 3
        
        # 3. 优先传给后卫
        if teammate_role in [PlayerRole.CENTRE_BACK, PlayerRole.LEFT_BACK, PlayerRole.RIGHT_BACK]:
            score += 2.0
        elif teammate_role in [PlayerRole.CENTRAL_MIDFIELD, PlayerRole.LEFT_MIDFIELD, PlayerRole.RIGHT_MIDFIELD]:
            score += 1.5
        
        # 4. 检查队友周围的空间
        from src.utils.features import get_space_around_player
        teammate_space = get_space_around_player(obs, i)
        score += teammate_space * 2
        
        # 5. 传球距离因素
        pass_distance = distance_to(player_pos, teammate_pos)
        if Distance.SHORT_PASS_RANGE * 0.5 < pass_distance < Distance.SHORT_PASS_RANGE * 1.2:
            score += 1.0  # 适中距离
        
        # 6. 检查传球路线是否清晰
        from src.utils.features import check_pass_path_clear
        if check_pass_path_clear(player_pos, teammate_pos, obs['right_team']):
            score += 1.5
        else:
            score -= 1.0
        
        # 7. 避免传给在边线的队友（容易失误）
        if abs(teammate_pos[1]) > 0.35:
            score -= 0.5
        
        if score > best_score:
            best_score = score
            best_target = i
    
    return best_target


def find_alternative_pass_target(obs, player_index, exclude=None):
    """寻找替代的传球目标"""
    if exclude is None:
        exclude = []
    
    player_pos = obs['left_team'][player_index]
    
    for i, teammate_pos in enumerate(obs['left_team']):
        if i == player_index or i in exclude or not obs['left_team_active'][i]:
            continue
        
        # 寻找在中场的队友
        teammate_role = obs['left_team_roles'][i]
        if teammate_role in [PlayerRole.CENTRAL_MIDFIELD, PlayerRole.LEFT_MIDFIELD, PlayerRole.RIGHT_MIDFIELD]:
            # 检查是否在相对安全的位置
            if teammate_pos[0] > Field.CENTER_X - 0.2:  # 在中场或前场
                closest_opp_idx, closest_opp_dist = find_closest_opponent(obs, i)
                if closest_opp_dist > Distance.PRESSURE_DISTANCE:
                    return i
    
    return -1


def goalkeeper_move_with_ball(obs, player_index, player_info):
    """守门员持球移动"""
    player_pos = player_info['position']
    
    # 向前移动一小段距离，寻找传球机会
    target_x = min(player_pos[0] + 0.05, Field.LEFT_GOAL_X + 0.1)
    target_y = player_pos[1]  # 保持在中央区域
    
    target_pos = [target_x, target_y]
    movement_action = get_movement_direction(player_pos, target_pos)
    
    if movement_action:
        return movement_action
    
    return Action.IDLE


def goalkeeper_defensive_logic(obs, player_index, ball_info, player_info):
    """守门员防守逻辑"""
    ball_pos = ball_info['position']
    player_pos = player_info['position']
    
    # 计算理想防守位置
    target_pos = get_goalkeeper_position(ball_pos)
    
    # 检查是否需要出击
    if should_goalkeeper_rush(ball_pos, obs):
        return goalkeeper_rush_logic(obs, player_index, ball_info)
    
    # 正常防守站位
    distance_to_target = distance_to(player_pos, target_pos)
    
    if distance_to_target > 0.01:  # 需要调整位置
        movement_action = get_movement_direction(player_pos, target_pos)
        
        # 如果位置调整较大，可以使用冲刺
        if distance_to_target > 0.05 and not is_player_tired(obs, player_index):
            # 先冲刺
            return Action.SPRINT
        
        if movement_action:
            return movement_action
    
    return Action.IDLE


def goalkeeper_rush_logic(obs, player_index, ball_info):
    """守门员出击逻辑"""
    ball_pos = ball_info['position']
    player_pos = obs['left_team'][player_index]
    
    # 直接冲向球的位置
    movement_action = get_movement_direction(player_pos, ball_pos)
    
    # 检查是否接近球
    distance_to_ball = distance_to(player_pos, ball_pos)
    
    if distance_to_ball < Distance.BALL_VERY_CLOSE:
        # 尝试铲球或拿球
        return Action.SLIDING
    
    # 冲刺冲向球
    if distance_to_ball > 0.05 and not is_player_tired(obs, player_index):
        return Action.SPRINT
    
    if movement_action:
        return movement_action
    
    return Action.IDLE


def goalkeeper_contention_logic(obs, player_index, ball_info, player_info):
    """守门员争抢逻辑（无人控球时）"""
    ball_pos = ball_info['position']
    player_pos = player_info['position']
    
    # 检查球是否在己方禁区内
    if is_ball_in_penalty_area(ball_pos):
        # 球在禁区内，积极出击争抢
        distance_to_ball = distance_to(player_pos, ball_pos)
        
        if distance_to_ball < Distance.BALL_CLOSE:
            # 接近球时减速并准备控球
            movement_action = get_movement_direction(player_pos, ball_pos)
            if movement_action:
                return movement_action
        else:
            # 冲刺向球
            return Action.SPRINT
    
    # 球不在禁区，保持防守位置
    target_pos = get_goalkeeper_position(ball_pos)
    movement_action = get_movement_direction(player_pos, target_pos)
    
    if movement_action:
        return movement_action
    
    return Action.IDLE


def should_goalkeeper_rush(ball_pos, obs):
    """判断守门员是否应该出击"""
    # 球在己方禁区内
    if not is_ball_in_penalty_area(ball_pos):
        return False
    
    # 检查是否有对手控球并威胁球门
    ball_info = get_ball_info(obs)
    if ball_info['owned_team'] == 1:  # 对方控球
        # 计算对手到球门的距离
        goal_distance = distance_to(ball_pos, [Field.LEFT_GOAL_X, Field.CENTER_Y])
        
        # 如果对手在禁区内且距离球门很近，考虑出击
        if goal_distance < 0.1:
            return True
    
    # 无人控球且球在小禁区内
    if ball_info['owned_team'] == -1 and ball_pos[0] > Field.LEFT_GOAL_X - 0.05:
        return True
    
    return False


def is_ball_in_penalty_area(ball_pos):
    """判断球是否在己方禁区内"""
    # 简化的禁区定义（实际禁区更复杂）
    penalty_area_x = Field.LEFT_GOAL_X + 0.165  # 禁区长度约0.165
    penalty_area_y = 0.2  # 禁区宽度的一半
    
    return (ball_pos[0] < penalty_area_x and 
            abs(ball_pos[1]) < penalty_area_y)


def get_goalkeeper_offensive_position(ball_pos):
    """计算守门员在进攻时的位置"""
    # 进攻时守门员可以稍微向前，但不能离球门太远
    base_x = Field.LEFT_GOAL_X + 0.03
    base_y = Field.CENTER_Y
    
    # 根据球的位置进行微调
    if ball_pos[0] > Field.CENTER_X:  # 球在对方半场
        # 可以更积极一些
        base_x = Field.LEFT_GOAL_X + 0.05
    
    return [base_x, base_y] 