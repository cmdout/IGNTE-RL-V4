"""
前锋决策逻辑
"""

from src.utils.features import (
    get_ball_info, get_player_info, distance_to, 
    find_closest_teammate, find_closest_opponent, 
    get_best_pass_target, get_movement_direction, 
    is_player_tired, is_in_opponent_half, can_shoot
)
from src.utils.actions import action_manager, validate_action_for_situation
from src.gfootball_agent.config import Action, Distance, Field, PlayerRole


def forward_decision(obs, player_index):
    """前锋决策逻辑"""
    ball_info = get_ball_info(obs)
    player_info = get_player_info(obs, player_index)
    ball_owned_team = ball_info['owned_team']
    
    # 根据控球权分发决策
    if ball_owned_team == 0:  # 我方控球
        return forward_offensive_logic(obs, player_index, ball_info, player_info)
    elif ball_owned_team == 1:  # 对方控球
        return forward_defensive_logic(obs, player_index, ball_info, player_info)
    else:  # 无人控球
        return forward_contention_logic(obs, player_index, ball_info, player_info)


def forward_offensive_logic(obs, player_index, ball_info, player_info):
    """前锋进攻逻辑"""
    player_pos = player_info['position']
    
    # 如果前锋持球
    if ball_info['owned_player'] == player_index:
        return forward_with_ball_logic(obs, player_index, ball_info, player_info)
    
    # 前锋无球时的跑位和策应
    return forward_off_ball_movement(obs, player_index, player_info)


def forward_with_ball_logic(obs, player_index, ball_info, player_info):
    """前锋持球时的决策逻辑"""
    player_pos = player_info['position']
    ball_pos = ball_info['position']
    
    # 优先考虑射门
    if can_shoot(player_pos, ball_pos, obs):
        goal_distance = distance_to(player_pos, [Field.RIGHT_GOAL_X, Field.CENTER_Y])
        
        # 在最佳射门范围内，直接射门
        if goal_distance < Distance.OPTIMAL_SHOT_RANGE:
            return Action.SHOT
        
        # 在射门范围内，检查射门角度和压力
        if goal_distance < Distance.SHOT_RANGE:
            closest_opponent_idx, closest_opponent_dist = find_closest_opponent(obs, player_index)
            
            # 如果没有被紧逼，可以射门
            if closest_opponent_dist > Distance.PRESSURE_DISTANCE:
                return Action.SHOT
    
    # 检查是否被对手逼抢
    closest_opponent_idx, closest_opponent_dist = find_closest_opponent(obs, player_index)
    
    if closest_opponent_dist < Distance.PRESSURE_DISTANCE:
        return forward_under_pressure(obs, player_index, ball_info, player_info)
    
    # 寻找更好的射门位置或传球机会
    return forward_create_opportunity(obs, player_index, ball_info, player_info)


def forward_under_pressure(obs, player_index, ball_info, player_info):
    """前锋被逼抢时的处理"""
    player_pos = player_info['position']
    
    # 寻找支援的队友
    best_target = get_best_pass_target(obs, player_index)
    
    if best_target != -1:
        target_pos = obs['left_team'][best_target]
        target_role = obs['left_team_roles'][best_target]
        pass_distance = distance_to(player_pos, target_pos)
        
        # 优先回传给中场，做球给后插上的队友
        if target_role in [PlayerRole.CENTRAL_MIDFIELD, PlayerRole.ATTACK_MIDFIELD]:
            # 检查后面是否有更好的射门角度
            goal_center = [Field.RIGHT_GOAL_X, Field.CENTER_Y]
            target_to_goal_dist = distance_to(target_pos, goal_center)
            player_to_goal_dist = distance_to(player_pos, goal_center)
            
            # 如果队友位置更好，回传做球
            if target_to_goal_dist < player_to_goal_dist + 0.05:
                if pass_distance < Distance.SHORT_PASS_RANGE:
                    return Action.SHORT_PASS
        
        # 一般情况下的传球
        if pass_distance < Distance.SHORT_PASS_RANGE:
            return Action.SHORT_PASS
    
    # 没有好的传球选择，尝试转身或护球
    return forward_protect_ball(obs, player_index, player_info)


def forward_create_opportunity(obs, player_index, ball_info, player_info):
    """前锋创造机会"""
    player_pos = player_info['position']
    goal_center = [Field.RIGHT_GOAL_X, Field.CENTER_Y]
    
    # 检查是否可以向球门方向盘带
    if can_dribble_towards_goal(obs, player_index, player_pos):
        return forward_dribble_to_goal(obs, player_index, player_info)
    
    # 寻找传球机会创造更好的射门角度
    best_target = get_best_pass_target(obs, player_index)
    
    if best_target != -1:
        target_pos = obs['left_team'][best_target]
        target_role = obs['left_team_roles'][best_target]
        
        # 如果有队友在更好的位置
        target_goal_distance = distance_to(target_pos, goal_center)
        player_goal_distance = distance_to(player_pos, goal_center)
        
        if target_goal_distance < player_goal_distance and is_in_opponent_half(target_pos):
            pass_distance = distance_to(player_pos, target_pos)
            if pass_distance < Distance.SHORT_PASS_RANGE:
                return Action.SHORT_PASS
    
    # 保持控球，等待更好的机会
    return Action.IDLE


def forward_protect_ball(obs, player_index, player_info):
    """前锋护球"""
    player_pos = player_info['position']
    
    # 寻找压力最小的方向
    escape_direction = find_escape_direction(obs, player_index, player_pos)
    
    if escape_direction:
        target_pos = [
            player_pos[0] + escape_direction[0],
            player_pos[1] + escape_direction[1]
        ]
        
        # 开始盘带
        current_sticky = obs['sticky_actions']
        if not current_sticky[9]:  # 没有在盘带
            return Action.DRIBBLE
        
        movement_action = get_movement_direction(player_pos, target_pos)
        if movement_action:
            return movement_action
    
    return Action.IDLE


def forward_dribble_to_goal(obs, player_index, player_info):
    """前锋向球门盘带"""
    player_pos = player_info['position']
    goal_center = [Field.RIGHT_GOAL_X, Field.CENTER_Y]
    
    # 计算向球门的方向，但避开对手
    direct_direction = [goal_center[0] - player_pos[0], goal_center[1] - player_pos[1]]
    direction_length = (direct_direction[0]**2 + direct_direction[1]**2)**0.5
    
    if direction_length > 0:
        # 标准化方向
        unit_direction = [direct_direction[0]/direction_length, direct_direction[1]/direction_length]
        
        # 调整方向以避开对手
        adjusted_direction = adjust_direction_to_avoid_opponents(obs, player_pos, unit_direction)
        
        target_pos = [
            player_pos[0] + adjusted_direction[0] * 0.05,
            player_pos[1] + adjusted_direction[1] * 0.05
        ]
        
        # 开始盘带
        current_sticky = obs['sticky_actions']
        if not current_sticky[9]:
            return Action.DRIBBLE
        
        movement_action = get_movement_direction(player_pos, target_pos)
        if movement_action:
            return movement_action
    
    return Action.IDLE


def forward_off_ball_movement(obs, player_index, player_info):
    """前锋无球跑位"""
    player_pos = player_info['position']
    ball_info = get_ball_info(obs)
    ball_pos = ball_info['position']
    
    # 根据球的位置选择跑位策略
    if is_in_opponent_half(ball_pos):
        # 球在对方半场，积极寻找射门机会
        return forward_attacking_run(obs, player_index, player_info, ball_pos)
    else:
        # 球在己方半场，准备接应反击
        return forward_counter_attack_run(obs, player_index, player_info, ball_pos)


def forward_attacking_run(obs, player_index, player_info, ball_pos):
    """前锋在对方半场的跑位 - 优化版本，创造更好的传球选项"""
    player_pos = player_info['position']
    goal_center = [Field.RIGHT_GOAL_X, Field.CENTER_Y]
    
    # 获取持球球员信息
    ball_info = get_ball_info(obs)
    ball_carrier_index = ball_info['owned_player']
    
    # 根据持球球员位置调整跑位策略
    if ball_carrier_index != -1:
        ball_carrier_pos = obs['left_team'][ball_carrier_index]
        ball_carrier_role = obs['left_team_roles'][ball_carrier_index]
        
        # 寻找最佳的接球位置，考虑创造传球线路
        best_position = find_best_receiving_position_enhanced(obs, player_index, ball_pos, ball_carrier_pos)
        
        if best_position:
            movement_action = get_movement_direction(player_pos, best_position)
            
            # 如果距离较远，使用冲刺
            distance_to_target = distance_to(player_pos, best_position)
            if distance_to_target > 0.08 and not is_player_tired(obs, player_index):
                return Action.SPRINT
            
            if movement_action:
                return movement_action
    
    # 默认智能跑位：寻找防守空隙
    target_position = find_defensive_gap(obs, player_index, ball_pos)
    
    if target_position:
        movement_action = get_movement_direction(player_pos, target_position)
        if movement_action:
            return movement_action
    
    # 兜底：向球门区域移动但避免越位
    target_x = min(ball_pos[0] + 0.08, Field.RIGHT_GOAL_X - 0.12)
    target_y = calculate_optimal_y_position(obs, player_pos, ball_pos)
    
    target_pos = [target_x, target_y]
    movement_action = get_movement_direction(player_pos, target_pos)
    
    if movement_action:
        return movement_action
    
    return Action.IDLE


def forward_counter_attack_run(obs, player_index, player_info, ball_pos):
    """前锋反击跑位"""
    player_pos = player_info['position']
    
    # 前锋在反击时应该拉开空间，准备接长传
    target_x = max(player_pos[0], Field.CENTER_X + 0.1)  # 保持在前场
    
    # 根据球的横向位置调整跑位
    if ball_pos[1] > 0.1:  # 球在右侧
        target_y = -0.1  # 跑向左侧，拉开空间
    elif ball_pos[1] < -0.1:  # 球在左侧
        target_y = 0.1   # 跑向右侧
    else:  # 球在中路
        target_y = 0     # 保持中路
    
    target_pos = [target_x, target_y]
    movement_action = get_movement_direction(player_pos, target_pos)
    
    if movement_action:
        return movement_action
    
    return Action.IDLE


def forward_defensive_logic(obs, player_index, ball_info, player_info):
    """前锋防守逻辑"""
    ball_pos = ball_info['position']
    player_pos = player_info['position']
    
    # 前锋的防守主要是对对方后卫进行骚扰性逼抢
    if should_forward_pressure(obs, player_index, ball_pos):
        return forward_pressure_logic(obs, player_index, ball_info, player_info)
    
    # 不进行深度防守，保持在前场的反击位置
    target_pos = get_forward_defensive_position(ball_pos)
    movement_action = get_movement_direction(player_pos, target_pos)
    
    if movement_action:
        return movement_action
    
    return Action.IDLE


def forward_pressure_logic(obs, player_index, ball_info, player_info):
    """前锋逼抢逻辑"""
    ball_pos = ball_info['position']
    player_pos = player_info['position']
    
    distance_to_ball = distance_to(player_pos, ball_pos)
    
    # 如果接近球，进行骚扰
    if distance_to_ball < Distance.BALL_CLOSE:
        return Action.SLIDING
    
    # 移动接近球，但不消耗太多体能
    if distance_to_ball < 0.15:
        movement_action = get_movement_direction(player_pos, ball_pos)
        if movement_action:
            return movement_action
    
    return Action.IDLE


def forward_contention_logic(obs, player_index, ball_info, player_info):
    """前锋争抢逻辑（无人控球时）"""
    ball_pos = ball_info['position']
    player_pos = player_info['position']
    
    distance_to_ball = distance_to(player_pos, ball_pos)
    
    # 如果球在前场且前锋是最接近的，积极争抢
    if is_in_opponent_half(ball_pos) and is_closest_forward_to_ball(obs, player_index, ball_pos):
        if distance_to_ball < Distance.BALL_CLOSE:
            movement_action = get_movement_direction(player_pos, ball_pos)
            if movement_action:
                return movement_action
        else:
            # 冲刺向球
            if not is_player_tired(obs, player_index):
                return Action.SPRINT
    
    # 否则保持前场位置
    target_pos = get_forward_positioning(ball_pos)
    movement_action = get_movement_direction(player_pos, target_pos)
    
    if movement_action:
        return movement_action
    
    return Action.IDLE


def can_dribble_towards_goal(obs, player_index, player_pos):
    """判断是否可以向球门盘带"""
    goal_center = [Field.RIGHT_GOAL_X, Field.CENTER_Y]
    
    # 检查前方是否有直接的路径
    direction_to_goal = [goal_center[0] - player_pos[0], goal_center[1] - player_pos[1]]
    distance_to_goal = (direction_to_goal[0]**2 + direction_to_goal[1]**2)**0.5
    
    if distance_to_goal == 0:
        return False
    
    # 检查前方是否有对手阻挡
    for i, opp_pos in enumerate(obs['right_team']):
        # 计算对手是否在前进路径上
        to_opponent = [opp_pos[0] - player_pos[0], opp_pos[1] - player_pos[1]]
        
        # 简化判断：如果对手在前方较近位置，不建议盘带
        if (to_opponent[0] > 0 and  # 对手在前方
            abs(to_opponent[1]) < 0.1 and  # 在合理的Y轴范围内
            (to_opponent[0]**2 + to_opponent[1]**2)**0.5 < 0.08):  # 距离较近
            return False
    
    return True


def find_escape_direction(obs, player_index, player_pos):
    """寻找逃脱方向"""
    directions = [
        [0.03, 0], [0, 0.03], [0, -0.03], [-0.03, 0],
        [0.02, 0.02], [0.02, -0.02], [-0.02, 0.02], [-0.02, -0.02]
    ]
    
    best_direction = None
    max_space = 0
    
    for direction in directions:
        test_pos = [player_pos[0] + direction[0], player_pos[1] + direction[1]]
        
        # 计算该方向的空间
        min_distance_to_opponent = float('inf')
        for opp_pos in obs['right_team']:
            dist = distance_to(test_pos, opp_pos)
            if dist < min_distance_to_opponent:
                min_distance_to_opponent = dist
        
        if min_distance_to_opponent > max_space:
            max_space = min_distance_to_opponent
            best_direction = direction
    
    return best_direction


def adjust_direction_to_avoid_opponents(obs, player_pos, desired_direction):
    """调整方向以避开对手"""
    # 检查期望方向上是否有对手
    test_pos = [player_pos[0] + desired_direction[0] * 0.05, 
                player_pos[1] + desired_direction[1] * 0.05]
    
    min_distance = float('inf')
    for opp_pos in obs['right_team']:
        dist = distance_to(test_pos, opp_pos)
        if dist < min_distance:
            min_distance = dist
    
    # 如果前方空间足够，保持原方向
    if min_distance > 0.04:
        return desired_direction
    
    # 尝试向左右偏移
    left_direction = [-desired_direction[1], desired_direction[0]]  # 90度逆时针旋转
    right_direction = [desired_direction[1], -desired_direction[0]]  # 90度顺时针旋转
    
    # 测试左右两个方向
    test_left = [player_pos[0] + left_direction[0] * 0.05, 
                 player_pos[1] + left_direction[1] * 0.05]
    test_right = [player_pos[0] + right_direction[0] * 0.05, 
                  player_pos[1] + right_direction[1] * 0.05]
    
    left_space = min(distance_to(test_left, opp_pos) for opp_pos in obs['right_team'])
    right_space = min(distance_to(test_right, opp_pos) for opp_pos in obs['right_team'])
    
    # 选择空间更大的方向
    if left_space > right_space:
        return left_direction
    else:
        return right_direction


def find_best_receiving_position(obs, player_index, ball_pos):
    """寻找最佳接球位置"""
    player_pos = obs['left_team'][player_index]
    goal_center = [Field.RIGHT_GOAL_X, Field.CENTER_Y]
    
    # 候选位置：在球的前方和侧方
    candidate_positions = [
        [ball_pos[0] + 0.08, ball_pos[1]],  # 球的前方
        [ball_pos[0] + 0.06, ball_pos[1] + 0.05],  # 右前方
        [ball_pos[0] + 0.06, ball_pos[1] - 0.05],  # 左前方
        [ball_pos[0] + 0.05, ball_pos[1] + 0.08],  # 右侧
        [ball_pos[0] + 0.05, ball_pos[1] - 0.08],  # 左侧
    ]
    
    best_position = None
    best_score = -1
    
    for pos in candidate_positions:
        # 确保位置在场地内
        if pos[0] > Field.RIGHT_BOUNDARY or pos[0] < Field.LEFT_BOUNDARY:
            continue
        if pos[1] > Field.BOTTOM_BOUNDARY or pos[1] < Field.TOP_BOUNDARY:
            continue
        
        # 计算评分
        score = 0
        
        # 距离球门越近越好
        distance_to_goal = distance_to(pos, goal_center)
        score += (1.0 - distance_to_goal) * 2
        
        # 距离对手越远越好
        min_distance_to_opponent = min(distance_to(pos, opp_pos) for opp_pos in obs['right_team'])
        score += min_distance_to_opponent * 3
        
        # 不要距离当前位置太远
        distance_to_current = distance_to(pos, player_pos)
        if distance_to_current < 0.15:
            score += 0.5
        
        if score > best_score:
            best_score = score
            best_position = pos
    
    return best_position


def find_best_receiving_position_enhanced(obs, player_index, ball_pos, ball_carrier_pos):
    """增强版寻找最佳接球位置，考虑传球线路"""
    player_pos = obs['left_team'][player_index]
    goal_center = [Field.RIGHT_GOAL_X, Field.CENTER_Y]
    
    # 扩展候选位置，包括更多的跑位选项
    candidate_positions = []
    
    # 前插到防守线身后
    for x_offset in [0.1, 0.12, 0.15]:
        for y_offset in [-0.05, 0, 0.05]:
            pos = [ball_pos[0] + x_offset, ball_pos[1] + y_offset]
            candidate_positions.append(pos)
    
    # 肋部跑位（防守队员之间的空隙）
    for y_offset in [-0.1, -0.15, 0.1, 0.15]:
        pos = [ball_pos[0] + 0.08, ball_pos[1] + y_offset]
        candidate_positions.append(pos)
    
    # 回撤接球（如果前方压力太大）
    if is_too_crowded_ahead(obs, ball_pos):
        for x_offset in [-0.02, -0.05]:
            for y_offset in [-0.08, 0, 0.08]:
                pos = [ball_carrier_pos[0] + x_offset, ball_carrier_pos[1] + y_offset]
                candidate_positions.append(pos)
    
    best_position = None
    best_score = -1
    
    for pos in candidate_positions:
        # 确保位置在场地内且合理
        if pos[0] > Field.RIGHT_BOUNDARY or pos[0] < Field.LEFT_BOUNDARY:
            continue
        if pos[1] > Field.BOTTOM_BOUNDARY or pos[1] < Field.TOP_BOUNDARY:
            continue
        if pos[0] < ball_carrier_pos[0] - 0.05:  # 避免过度回撤
            continue
        
        # 计算综合评分
        score = calculate_receiving_position_score(obs, pos, player_pos, ball_carrier_pos, goal_center)
        
        if score > best_score:
            best_score = score
            best_position = pos
    
    return best_position


def calculate_receiving_position_score(obs, position, current_pos, ball_carrier_pos, goal_center):
    """计算接球位置的评分"""
    score = 0
    
    # 1. 距离球门越近越好（主要因素）
    distance_to_goal = distance_to(position, goal_center)
    score += (1.5 - distance_to_goal) * 3
    
    # 2. 与对手的距离（安全性）
    min_distance_to_opponent = min(distance_to(position, opp_pos) for opp_pos in obs['right_team'])
    score += min_distance_to_opponent * 4
    
    # 3. 传球路线的清晰度
    from src.utils.features import check_pass_path_clear
    if check_pass_path_clear(ball_carrier_pos, position, obs['right_team']):
        score += 2.0
    else:
        score -= 1.0
    
    # 4. 移动距离的合理性
    move_distance = distance_to(current_pos, position)
    if move_distance > 0.2:
        score -= 1.0  # 移动距离太远扣分
    elif move_distance < 0.05:
        score -= 0.5  # 移动距离太近也不好
    
    # 5. 位置的战术价值
    if position[0] > ball_carrier_pos[0] + 0.05:  # 向前跑位奖励
        score += 1.5
    
    # 6. 避免越位
    if is_offside_position(obs, position):
        score -= 3.0
    
    # 7. 射门角度
    shot_angle = calculate_shot_angle(position, goal_center)
    if shot_angle < 45:  # 好的射门角度
        score += 1.0
    
    return score


def find_defensive_gap(obs, player_index, ball_pos):
    """寻找防守空隙"""
    player_pos = obs['left_team'][player_index]
    
    # 分析对手防线
    defenders = []
    for i, opp_pos in enumerate(obs['right_team']):
        role = obs['right_team_roles'][i]
        if role in [PlayerRole.CENTRE_BACK, PlayerRole.LEFT_BACK, PlayerRole.RIGHT_BACK]:
            defenders.append(opp_pos)
    
    if len(defenders) < 2:
        return None
    
    # 寻找防守队员之间的空隙
    gaps = []
    for i in range(len(defenders)):
        for j in range(i + 1, len(defenders)):
            gap_center = [
                (defenders[i][0] + defenders[j][0]) / 2,
                (defenders[i][1] + defenders[j][1]) / 2
            ]
            gap_width = distance_to(defenders[i], defenders[j])
            
            if gap_width > 0.15:  # 足够大的空隙
                gaps.append((gap_center, gap_width))
    
    # 选择最好的空隙
    best_gap = None
    best_score = -1
    
    for gap_center, gap_width in gaps:
        # 确保空隙在前方
        if gap_center[0] <= ball_pos[0]:
            continue
        
        score = gap_width * 2  # 空隙越大越好
        score += (gap_center[0] - ball_pos[0])  # 越靠前越好
        
        # 考虑到达空隙的难度
        distance_to_gap = distance_to(player_pos, gap_center)
        if distance_to_gap < 0.2:
            score += 0.5
        
        if score > best_score:
            best_score = score
            best_gap = gap_center
    
    return best_gap


def calculate_optimal_y_position(obs, player_pos, ball_pos):
    """计算最优的Y轴位置"""
    # 分析场上的分布，避免扎堆
    teammates_y = []
    for i, teammate_pos in enumerate(obs['left_team']):
        if obs['left_team_active'][i] and teammate_pos[0] > Field.CENTER_X - 0.1:
            teammates_y.append(teammate_pos[1])
    
    # 寻找人员稀少的区域
    y_positions = [-0.2, -0.1, 0, 0.1, 0.2]
    best_y = 0
    max_space = -1
    
    for y in y_positions:
        space = min(abs(y - teammate_y) for teammate_y in teammates_y) if teammates_y else 1.0
        if space > max_space:
            max_space = space
            best_y = y
    
    return best_y


def is_too_crowded_ahead(obs, ball_pos):
    """判断前方是否过于拥挤"""
    crowded_area_count = 0
    for opp_pos in obs['right_team']:
        if (opp_pos[0] > ball_pos[0] and 
            opp_pos[0] < ball_pos[0] + 0.15 and
            abs(opp_pos[1] - ball_pos[1]) < 0.2):
            crowded_area_count += 1
    
    return crowded_area_count >= 3


def is_offside_position(obs, position):
    """简单的越位判断"""
    # 找到最后一名对方防守球员的位置
    last_defender_x = min(opp_pos[0] for opp_pos in obs['right_team'])
    
    # 如果前锋位置超过最后一名防守球员，可能越位
    return position[0] > last_defender_x - 0.02


def calculate_shot_angle(position, goal_center):
    """计算射门角度"""
    goal_posts = [[Field.RIGHT_GOAL_X, 0.044], [Field.RIGHT_GOAL_X, -0.044]]
    
    # 计算到两个门柱的角度
    vector1 = [goal_posts[0][0] - position[0], goal_posts[0][1] - position[1]]
    vector2 = [goal_posts[1][0] - position[0], goal_posts[1][1] - position[1]]
    
    # 计算夹角
    import math
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = (vector1[0]**2 + vector1[1]**2)**0.5
    magnitude2 = (vector2[0]**2 + vector2[1]**2)**0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    cos_angle = dot_product / (magnitude1 * magnitude2)
    cos_angle = max(-1, min(1, cos_angle))  # 防止数值误差
    
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


def should_forward_pressure(obs, player_index, ball_pos):
    """判断前锋是否应该逼抢"""
    player_pos = obs['left_team'][player_index]
    distance_to_ball = distance_to(player_pos, ball_pos)
    
    # 只有在对方后场且距离较近时才逼抢
    if ball_pos[0] < -0.3:  # 球在对方后场
        return distance_to_ball < 0.2
    
    return False


def is_closest_forward_to_ball(obs, player_index, ball_pos):
    """判断是否是离球最近的前锋"""
    player_pos = obs['left_team'][player_index]
    player_distance = distance_to(player_pos, ball_pos)
    
    for i, pos in enumerate(obs['left_team']):
        if i == player_index:
            continue
        
        role = obs['left_team_roles'][i]
        if role == PlayerRole.CENTRAL_FORWARD:
            if distance_to(pos, ball_pos) < player_distance:
                return False
    
    return True


def get_forward_defensive_position(ball_pos):
    """获取前锋的防守位置"""
    # 前锋在防守时保持在中场前沿，随时准备反击
    target_x = max(ball_pos[0] - 0.1, Field.CENTER_X - 0.1)
    target_y = ball_pos[1] * 0.3  # 轻微跟随球的横向位置
    
    return [target_x, target_y]


def get_forward_positioning(ball_pos):
    """获取前锋的基本站位"""
    # 前锋基本保持在前场
    target_x = max(ball_pos[0], Field.CENTER_X + 0.05)
    target_y = ball_pos[1] * 0.5  # 跟随球的横向位置
    
    return [target_x, target_y] 