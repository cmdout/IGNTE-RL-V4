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
    """守门员持球时的决策逻辑"""
    player_pos = player_info['position']
    
    # 寻找最佳传球目标
    best_target = get_best_pass_target(obs, player_index)
    
    if best_target != -1:
        target_pos = obs['left_team'][best_target]
        pass_distance = distance_to(player_pos, target_pos)
        
        # 检查是否有对手逼抢
        closest_opponent_idx, closest_opponent_dist = find_closest_opponent(obs, player_index)
        
        # 如果被紧逼，优先快速出球
        if closest_opponent_dist < Distance.PRESSURE_DISTANCE:
            # 紧急情况，长传到安全区域
            if pass_distance > Distance.LONG_PASS_RANGE * 0.5:
                return Action.LONG_PASS
            else:
                return Action.SHORT_PASS
        
        # 正常情况下的传球选择
        if pass_distance < Distance.SHORT_PASS_RANGE:
            # 短传给后卫
            return Action.SHORT_PASS
        else:
            # 长传到中场
            return Action.LONG_PASS
    
    # 没有好的传球选择，持球移动寻找机会
    return goalkeeper_move_with_ball(obs, player_index, player_info)


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