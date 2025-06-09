"""
所有定位球模式的逻辑
""" 

from src.utils.features import (
    get_ball_info, get_player_info, distance_to, 
    find_closest_teammate, get_movement_direction,
    is_in_opponent_half, can_shoot
)
from src.gfootball_agent.config import Action, GameMode, PlayerRole, Field, Distance


def set_piece_decision(obs, player_index):
    """
    定位球模式决策
    根据具体的定位球类型分发到对应的处理函数
    """
    game_mode = obs['game_mode']
    
    if game_mode == GameMode.KICK_OFF:
        return kick_off_logic(obs, player_index)
    elif game_mode == GameMode.GOAL_KICK:
        return goal_kick_logic(obs, player_index)
    elif game_mode == GameMode.FREE_KICK:
        return free_kick_logic(obs, player_index)
    elif game_mode == GameMode.CORNER:
        return corner_logic(obs, player_index)
    elif game_mode == GameMode.THROW_IN:
        return throw_in_logic(obs, player_index)
    elif game_mode == GameMode.PENALTY:
        return penalty_logic(obs, player_index)
    else:
        # 默认返回IDLE
        return Action.IDLE


def kick_off_logic(obs, player_index):
    """开球逻辑"""
    ball_info = get_ball_info(obs)
    player_pos = obs['left_team'][player_index]
    ball_pos = ball_info['position']
    
    # 检查是否是主罚球员（最接近球的球员）
    if is_main_set_piece_taker(obs, player_index, ball_pos):
        # 主罚开球，短传给最近的队友
        closest_teammate_idx, closest_teammate_dist = find_closest_teammate(obs, player_index)
        
        if closest_teammate_idx != -1 and closest_teammate_dist < Distance.SHORT_PASS_RANGE:
            return Action.SHORT_PASS
        else:
            # 没有近距离队友，向前长传
            return Action.LONG_PASS
    
    # 非主罚球员，移动到接应位置
    return kick_off_support_movement(obs, player_index)


def kick_off_support_movement(obs, player_index):
    """开球时的支援跑位"""
    player_role = obs['left_team_roles'][player_index]
    player_pos = obs['left_team'][player_index]
    
    # 根据角色确定跑位位置
    if player_role == PlayerRole.GOALKEEPER:
        # 守门员保持在球门附近
        target_pos = [Field.LEFT_GOAL_X + 0.03, Field.CENTER_Y]
    elif player_role in [PlayerRole.CENTRE_BACK, PlayerRole.LEFT_BACK, PlayerRole.RIGHT_BACK]:
        # 后卫在己方半场准备接应
        target_x = Field.CENTER_X - 0.2
        target_y = player_pos[1]  # 保持当前Y位置
        target_pos = [target_x, target_y]
    elif player_role in [PlayerRole.CENTRAL_MIDFIELD, PlayerRole.LEFT_MIDFIELD, PlayerRole.RIGHT_MIDFIELD]:
        # 中场球员在中圈附近准备接球
        target_x = Field.CENTER_X - 0.1
        if player_role == PlayerRole.LEFT_MIDFIELD:
            target_y = -0.15
        elif player_role == PlayerRole.RIGHT_MIDFIELD:
            target_y = 0.15
        else:
            target_y = 0
        target_pos = [target_x, target_y]
    else:
        # 前锋保持在对方半场边缘
        target_pos = [Field.CENTER_X + 0.05, 0]
    
    movement_action = get_movement_direction(player_pos, target_pos)
    return movement_action if movement_action else Action.IDLE


def goal_kick_logic(obs, player_index):
    """球门球逻辑"""
    ball_info = get_ball_info(obs)
    player_pos = obs['left_team'][player_index]
    ball_pos = ball_info['position']
    player_role = obs['left_team_roles'][player_index]
    
    # 守门员主罚球门球
    if player_role == PlayerRole.GOALKEEPER and is_main_set_piece_taker(obs, player_index, ball_pos):
        # 寻找安全的传球目标
        safest_target = find_safest_goal_kick_target(obs, player_index)
        
        if safest_target != -1:
            target_pos = obs['left_team'][safest_target]
            pass_distance = distance_to(player_pos, target_pos)
            
            if pass_distance < Distance.SHORT_PASS_RANGE:
                return Action.SHORT_PASS
            else:
                return Action.LONG_PASS
        else:
            # 没有安全目标，大脚开向边路
            return Action.LONG_PASS
    
    # 其他球员移动到接应位置
    return goal_kick_support_movement(obs, player_index)


def goal_kick_support_movement(obs, player_index):
    """球门球支援跑位"""
    player_role = obs['left_team_roles'][player_index]
    player_pos = obs['left_team'][player_index]
    
    if player_role in [PlayerRole.CENTRE_BACK, PlayerRole.LEFT_BACK, PlayerRole.RIGHT_BACK]:
        # 后卫拉开接应
        target_x = Field.LEFT_GOAL_X + 0.2
        if player_role == PlayerRole.LEFT_BACK:
            target_y = -0.2
        elif player_role == PlayerRole.RIGHT_BACK:
            target_y = 0.2
        else:
            target_y = player_pos[1] * 0.5
        target_pos = [target_x, target_y]
    elif player_role in [PlayerRole.CENTRAL_MIDFIELD, PlayerRole.LEFT_MIDFIELD, PlayerRole.RIGHT_MIDFIELD]:
        # 中场拉开准备接长传
        target_x = Field.CENTER_X - 0.1
        if player_role == PlayerRole.LEFT_MIDFIELD:
            target_y = -0.25
        elif player_role == PlayerRole.RIGHT_MIDFIELD:
            target_y = 0.25
        else:
            target_y = 0
        target_pos = [target_x, target_y]
    else:
        # 前锋移动到中场
        target_pos = [Field.CENTER_X, 0]
    
    movement_action = get_movement_direction(player_pos, target_pos)
    return movement_action if movement_action else Action.IDLE


def free_kick_logic(obs, player_index):
    """任意球逻辑"""
    ball_info = get_ball_info(obs)
    player_pos = obs['left_team'][player_index]
    ball_pos = ball_info['position']
    
    # 检查是否是主罚球员
    if is_main_set_piece_taker(obs, player_index, ball_pos):
        # 检查是否可以直接射门
        if can_shoot(player_pos, ball_pos, obs):
            goal_distance = distance_to(ball_pos, [Field.RIGHT_GOAL_X, Field.CENTER_Y])
            
            # 在合适的射门距离内
            if goal_distance < Distance.SHOT_RANGE:
                return Action.SHOT
        
        # 不能射门，寻找传球机会
        best_target = find_free_kick_target(obs, player_index)
        
        if best_target != -1:
            target_pos = obs['left_team'][best_target]
            pass_distance = distance_to(ball_pos, target_pos)
            
            # 根据距离选择传球方式
            if pass_distance > Distance.LONG_PASS_RANGE * 0.5:
                return Action.HIGH_PASS  # 高球传入禁区
            else:
                return Action.SHORT_PASS
        
        # 默认高球传向前场
        return Action.HIGH_PASS
    
    # 非主罚球员移动到战术位置
    return free_kick_support_movement(obs, player_index)


def free_kick_support_movement(obs, player_index):
    """任意球支援跑位"""
    player_role = obs['left_team_roles'][player_index]
    player_pos = obs['left_team'][player_index]
    ball_pos = get_ball_info(obs)['position']
    
    if is_in_opponent_half(ball_pos):
        # 在对方半场的任意球，积极前插
        if player_role == PlayerRole.CENTRAL_FORWARD:
            # 前锋进入禁区抢点
            target_pos = [Field.RIGHT_GOAL_X - 0.1, 0]
        elif player_role in [PlayerRole.CENTRE_BACK]:
            # 高大的中后卫也可以前插抢点
            target_pos = [Field.RIGHT_GOAL_X - 0.15, player_pos[1]]
        elif player_role in [PlayerRole.CENTRAL_MIDFIELD, PlayerRole.ATTACK_MIDFIELD]:
            # 中场在禁区外准备抢第二落点
            target_pos = [Field.RIGHT_GOAL_X - 0.25, 0]
        else:
            # 其他球员保持位置
            target_pos = player_pos
    else:
        # 在己方半场的任意球，保持防守阵型
        target_pos = get_defensive_free_kick_position(player_role, player_pos)
    
    movement_action = get_movement_direction(player_pos, target_pos)
    return movement_action if movement_action else Action.IDLE


def corner_logic(obs, player_index):
    """角球逻辑"""
    ball_info = get_ball_info(obs)
    ball_pos = ball_info['position']
    player_pos = obs['left_team'][player_index]
    
    # 检查是否是主罚球员
    if is_main_set_piece_taker(obs, player_index, ball_pos):
        # 角球主罚，高球传入禁区
        return Action.HIGH_PASS
    
    # 非主罚球员进行角球跑位
    return corner_support_movement(obs, player_index)


def corner_support_movement(obs, player_index):
    """角球支援跑位"""
    player_role = obs['left_team_roles'][player_index]
    player_pos = obs['left_team'][player_index]
    ball_pos = get_ball_info(obs)['position']
    
    if player_role == PlayerRole.CENTRAL_FORWARD:
        # 前锋到前门柱抢点
        target_pos = [Field.RIGHT_GOAL_X - 0.02, Field.GOAL_TOP_Y + 0.02]
    elif player_role in [PlayerRole.CENTRE_BACK]:
        # 中后卫到后门柱抢点
        target_pos = [Field.RIGHT_GOAL_X - 0.02, Field.GOAL_BOTTOM_Y - 0.02]
    elif player_role in [PlayerRole.CENTRAL_MIDFIELD, PlayerRole.ATTACK_MIDFIELD]:
        # 中场在小禁区内抢点
        target_pos = [Field.RIGHT_GOAL_X - 0.08, 0]
    elif player_role == PlayerRole.GOALKEEPER:
        # 守门员保持在己方球门
        target_pos = [Field.LEFT_GOAL_X + 0.01, Field.CENTER_Y]
    else:
        # 其他球员在禁区外保护
        target_pos = [Field.RIGHT_GOAL_X - 0.2, player_pos[1]]
    
    movement_action = get_movement_direction(player_pos, target_pos)
    return movement_action if movement_action else Action.IDLE


def throw_in_logic(obs, player_index):
    """界外球逻辑"""
    ball_info = get_ball_info(obs)
    ball_pos = ball_info['position']
    player_pos = obs['left_team'][player_index]
    
    # 检查是否是主罚球员
    if is_main_set_piece_taker(obs, player_index, ball_pos):
        # 寻找最近的安全传球目标
        closest_teammate_idx, closest_teammate_dist = find_closest_teammate(obs, player_index)
        
        if closest_teammate_idx != -1 and closest_teammate_dist < Distance.SHORT_PASS_RANGE:
            return Action.SHORT_PASS
        else:
            # 向前场传球
            return Action.LONG_PASS
    
    # 非主罚球员移动到接应位置
    return throw_in_support_movement(obs, player_index)


def throw_in_support_movement(obs, player_index):
    """界外球支援跑位"""
    player_role = obs['left_team_roles'][player_index]
    player_pos = obs['left_team'][player_index]
    ball_pos = get_ball_info(obs)['position']
    
    # 简单的接应跑位：向球的方向移动一点
    if abs(ball_pos[1]) > 0.35:  # 球在边线附近
        target_x = ball_pos[0] - 0.05  # 向内场移动
        target_y = ball_pos[1] * 0.8   # 向中路靠拢
        target_pos = [target_x, target_y]
    else:
        # 保持当前位置
        target_pos = player_pos
    
    movement_action = get_movement_direction(player_pos, target_pos)
    return movement_action if movement_action else Action.IDLE


def penalty_logic(obs, player_index):
    """点球逻辑"""
    ball_info = get_ball_info(obs)
    ball_pos = ball_info['position']
    player_role = obs['left_team_roles'][player_index]
    
    # 守门员防守点球
    if player_role == PlayerRole.GOALKEEPER:
        # 随机选择扑救方向
        import random
        directions = [Action.LEFT, Action.RIGHT, Action.IDLE]
        return random.choice(directions)
    
    # 主罚球员射门
    if is_main_set_piece_taker(obs, player_index, ball_pos):
        return Action.SHOT
    
    # 其他球员保持位置
    return Action.IDLE


def is_main_set_piece_taker(obs, player_index, ball_pos):
    """判断是否是主罚球员（距离球最近的球员）"""
    player_pos = obs['left_team'][player_index]
    player_distance = distance_to(player_pos, ball_pos)
    
    # 检查是否是距离球最近的球员
    for i, pos in enumerate(obs['left_team']):
        if i == player_index:
            continue
        
        if distance_to(pos, ball_pos) < player_distance:
            return False
    
    return True


def find_safest_goal_kick_target(obs, player_index):
    """为球门球寻找最安全的传球目标"""
    player_pos = obs['left_team'][player_index]
    safest_target = -1
    max_safety = 0
    
    for i, teammate_pos in enumerate(obs['left_team']):
        if i == player_index or not obs['left_team_active'][i]:
            continue
        
        # 计算安全性（距离最近对手的距离）
        min_distance_to_opponent = float('inf')
        for opp_pos in obs['right_team']:
            dist = distance_to(teammate_pos, opp_pos)
            if dist < min_distance_to_opponent:
                min_distance_to_opponent = dist
        
        if min_distance_to_opponent > max_safety:
            max_safety = min_distance_to_opponent
            safest_target = i
    
    return safest_target


def find_free_kick_target(obs, player_index):
    """为任意球寻找最佳传球目标"""
    ball_pos = get_ball_info(obs)['position']
    best_target = -1
    best_score = -1
    
    for i, teammate_pos in enumerate(obs['left_team']):
        if i == player_index or not obs['left_team_active'][i]:
            continue
        
        # 评分标准：距离球门近 + 距离对手远 + 在对方半场
        score = 0
        
        # 距离球门越近越好
        goal_distance = distance_to(teammate_pos, [Field.RIGHT_GOAL_X, Field.CENTER_Y])
        score += (1.0 - goal_distance) * 2
        
        # 距离对手越远越好
        min_distance_to_opponent = min(distance_to(teammate_pos, opp_pos) for opp_pos in obs['right_team'])
        score += min_distance_to_opponent
        
        # 在对方半场加分
        if is_in_opponent_half(teammate_pos):
            score += 1
        
        if score > best_score:
            best_score = score
            best_target = i
    
    return best_target


def get_defensive_free_kick_position(player_role, current_pos):
    """获取防守任意球时的位置"""
    if player_role == PlayerRole.GOALKEEPER:
        return [Field.LEFT_GOAL_X + 0.01, Field.CENTER_Y]
    elif player_role in [PlayerRole.CENTRE_BACK, PlayerRole.LEFT_BACK, PlayerRole.RIGHT_BACK]:
        # 后卫回到防线
        return [Field.LEFT_GOAL_X + 0.15, current_pos[1]]
    else:
        # 中场和前锋适度回撤
        return [current_pos[0] - 0.05, current_pos[1]] 