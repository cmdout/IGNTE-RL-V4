"""
中场的决策逻辑
""" 
import numpy as np

from src.utils.features import (
    get_ball_info, get_player_info, distance_to,
    get_midfielder_defensive_position, find_closest_teammate,
    find_closest_opponent, get_best_pass_target,
    get_movement_direction, is_player_tired,
    is_in_opponent_half, can_shoot, is_in_own_half,
    is_offside_position, check_pass_path_clear,
    calculate_shot_angle, distance_to_line, check_dribble_space # Added for utility functions
)
from src.utils.actions import action_manager, validate_action_for_situation
from src.gfootball_agent.config import Action, Distance, Field, PlayerRole, Tactics


def _evaluate_shot_utility_mid(obs, player_index, player_pos, ball_pos, player_role):
    """Evaluates the utility of taking a shot for a midfielder."""
    if not can_shoot(player_pos, ball_pos, obs):
        return -float('inf'), None

    utility = 0.0
    goal_center = [Field.RIGHT_GOAL_X, Field.CENTER_Y]
    distance_to_goal = distance_to(player_pos, goal_center)

    # Base utility based on distance - less aggressive than forward
    if distance_to_goal < Distance.OPTIMAL_SHOT_RANGE:
        utility += 8.0
    elif distance_to_goal < Distance.SHOT_RANGE:
        utility += 4.0
        utility -= (distance_to_goal - Distance.OPTIMAL_SHOT_RANGE) * 60 # Steeper penalty for midfielders
    else:
        return -float('inf'), None

    # Angle
    angle_deg = abs(calculate_shot_angle(player_pos, goal_center)) # from features
    if angle_deg < 20.0: utility += 2.0
    elif angle_deg < 35.0: utility += 1.0
    else: utility -= (angle_deg - 35.0) * 0.3

    # Opponent Blockers
    opponents_blocking = 0
    for opp_idx, opp_pos_val in enumerate(obs['right_team']):
        if obs['right_team_active'][opp_idx]:
            if distance_to_line(np.array(player_pos), np.array(goal_center), np.array(opp_pos_val)) < 0.07:
                 if distance_to(player_pos, opp_pos_val) < distance_to_goal :
                    opponents_blocking += 1
    utility -= opponents_blocking * 3.5

    # Role-specific adjustment
    if player_role == PlayerRole.ATTACK_MIDFIELD:
        utility += 4.0 # AMs are encouraged more
    elif player_role in [PlayerRole.CENTRAL_MIDFIELD, PlayerRole.LEFT_MIDFIELD, PlayerRole.RIGHT_MIDFIELD]:
        utility -= 2.0 # Other midfielders less likely to shoot
    else: # Defensive midfielders
        utility -= 5.0

    MIN_VIABLE_SHOT_UTILITY_MID = 2.0
    if utility < MIN_VIABLE_SHOT_UTILITY_MID:
         return -float('inf'), None
    return utility, Action.SHOT


def _evaluate_pass_utility_mid(obs, player_index, player_pos, player_role):
    """Evaluates the utility of making a pass for a midfielder."""
    best_target_index = get_best_pass_target(obs, player_index) # Assumes enhanced get_best_pass_target

    if best_target_index == -1:
        return -float('inf'), None

    utility = 5.0 # Base utility for finding a pass
    teammate_pos = obs['left_team'][best_target_index]
    teammate_role = obs['left_team_roles'][best_target_index]
    pass_distance = distance_to(player_pos, teammate_pos)
    forward_progress = teammate_pos[0] - player_pos[0]

    utility += forward_progress * 15.0 # Midfielders still value progression

    if is_in_opponent_half(teammate_pos): utility += 1.5
    if teammate_role == PlayerRole.CENTRAL_FORWARD: utility += 3.0 # High value to pass to forward
    elif teammate_role == PlayerRole.ATTACK_MIDFIELD and player_role != PlayerRole.ATTACK_MIDFIELD : utility += 1.5


    _ , dist_to_closest_opp_to_receiver = find_closest_opponent(obs, best_target_index)
    if dist_to_closest_opp_to_receiver < Distance.PRESSURE_DISTANCE * 1.3:
        utility -= 4.0 # Penalize passing to marked teammates heavily for midfielders

    pass_action = Action.SHORT_PASS
    if pass_distance > Distance.SHORT_PASS_RANGE * 0.85:
        if forward_progress > 0.15: # Good forward long pass
            pass_action = Action.LONG_PASS if abs(player_pos[1] - teammate_pos[1]) < 0.3 else Action.HIGH_PASS
            utility += 1.0
        else: # Sideways or backward long pass by midfielder can be for switching play
            if abs(player_pos[1] - teammate_pos[1]) > 0.3: # Significant switch in Y
                utility += 2.0 # Reward switching play
                pass_action = Action.LONG_PASS
            else:
                utility -= (pass_distance - Distance.SHORT_PASS_RANGE) * 15 # Penalize aimless long passes

    if pass_distance < Distance.SHORT_PASS_RANGE * 0.25: utility -= 1.5 # Too short often not useful

    MIN_VIABLE_PASS_UTILITY_MID = 0.0 # Midfielders should always try to pass if possible
    if utility < MIN_VIABLE_PASS_UTILITY_MID:
        # Even if low utility, if it's the only option, might still be taken by main logic
        # Return low score but valid action if a target exists
        return utility, pass_action

    return utility, pass_action


def _evaluate_dribble_utility_mid(obs, player_index, player_pos, player_role):
    """Evaluates the utility of dribbling for a midfielder."""
    utility = 0.0
    _ , closest_opp_dist = find_closest_opponent(obs, player_index)

    if closest_opp_dist < Distance.BALL_CLOSE * 1.2: utility -= 6.0
    elif closest_opp_dist < Distance.PRESSURE_DISTANCE * 0.8: utility -= 3.0
    else: utility += 2.0

    # Check space towards opponent goal (e.g., positive X direction)
    # Using check_dribble_space from features.py would be more robust here
    has_fwd_space, _ = check_dribble_space(obs, player_index, direction_vector=[0.1, 0.0]) # Check small fwd dribble
    if has_fwd_space:
        utility += 2.5
        if player_pos[0] > 0.0: utility += 1.0 # Bonus in opponent half
        if player_role == PlayerRole.ATTACK_MIDFIELD: utility += 1.5 # AM more likely to dribble
    else:
        utility -= 1.5

    # Wing midfielders might dribble more on flanks
    if player_role in [PlayerRole.LEFT_MIDFIELD, PlayerRole.RIGHT_MIDFIELD] and abs(player_pos[1]) > 0.2:
        utility += 1.0

    if player_role == PlayerRole.DEFENCE_MIDFIELD: utility -= 3.0 # DMF less likely to dribble

    MIN_VIABLE_DRIBBLE_UTILITY_MID = -2.0 # Can be slightly negative if it's to escape trouble
    if utility < MIN_VIABLE_DRIBBLE_UTILITY_MID:
         return -float('inf'), None

    return utility, Action.DRIBBLE


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
    """中场球员持球时的决策逻辑 - 基于效用评估"""
    player_pos = player_info['position']
    ball_pos = player_info['position'] # Same when player has ball
    player_role = player_info['role']

    _ , closest_opponent_dist = find_closest_opponent(obs, player_index)
    under_pressure = closest_opponent_dist < Distance.PRESSURE_DISTANCE

    if under_pressure:
        # Prioritize safe pass or escape dribble.
        pass_utility, pass_action = _evaluate_pass_utility_mid(obs, player_index, player_pos, player_role)
        if pass_utility > -float('inf'): # Pass target found
            best_target_idx = get_best_pass_target(obs, player_index) # Re-call to get index
            if best_target_idx != -1:
                _ , dist_receiver_opp = find_closest_opponent(obs, best_target_idx)
                if dist_receiver_opp > Distance.PRESSURE_DISTANCE * 1.8: # Receiver is very free
                    pass_utility += 10.0
        
        dribble_utility, dribble_action = _evaluate_dribble_utility_mid(obs, player_index, player_pos, player_role)
        # For escape dribble, utility might be based on just finding *any* space
        # The current _evaluate_dribble_utility_mid is basic. midfielder_escape_dribble is more specific.
        
        if pass_utility > dribble_utility + 2.0 and pass_action is not None : # Prefer pass if significantly better or dribble not viable
             return pass_action
        
        # Try specific escape dribble logic if general dribble utility is low or pass is bad
        escape_action = midfielder_escape_dribble(obs, player_index, player_info)
        if escape_action != Action.IDLE : # If escape found a way
            # check if sticky action for dribble is active, if not, set it
            if not obs['sticky_actions'][9]: return Action.DRIBBLE
            return escape_action # Return actual movement from escape

        if pass_action is not None and pass_utility > -5.0: # Fallback to any pass if escape failed
            return pass_action
        
        return Action.IDLE # Hold ball if nothing else

    # Not under immediate pressure: Evaluate all options
    shot_utility, shot_action = _evaluate_shot_utility_mid(obs, player_index, player_pos, ball_pos, player_role)
    pass_utility, pass_action = _evaluate_pass_utility_mid(obs, player_index, player_pos, player_role)
    dribble_utility, dribble_action = _evaluate_dribble_utility_mid(obs, player_index, player_pos, player_role)

    # print(f"P{player_index}({player_role}) Utils: Shot={shot_utility:.1f}, Pass={pass_utility:.1f}, Dribble={dribble_utility:.1f}")

    MIN_COMMIT_THRESHOLD_MID = 1.0 # General threshold for midfielders

    # Prioritize Pass for most midfielders, then Shot (for AM), then Dribble.
    if player_role == PlayerRole.ATTACK_MIDFIELD:
        if shot_utility > pass_utility and shot_utility > dribble_utility and shot_utility > MIN_COMMIT_THRESHOLD_MID:
            return shot_action
        elif pass_utility > shot_utility and pass_utility > dribble_utility and pass_utility > MIN_COMMIT_THRESHOLD_MID:
            return pass_action
        elif dribble_utility > MIN_COMMIT_THRESHOLD_MID: # Dribble if viable and other options not better
            # Use midfielder_dribble_logic for directed dribble
            return midfielder_dribble_logic(obs, player_index, player_info)
    else: # Other midfielders (CM, WM, DMF) - Pass is king
        if pass_utility > shot_utility + 3.0 and pass_utility > dribble_utility + 2.0 and pass_utility > MIN_COMMIT_THRESHOLD_MID: # Pass needs to be clearly better
            return pass_action
        elif shot_utility > MIN_COMMIT_THRESHOLD_MID and shot_utility > dribble_utility : # Rare shot for non-AM
            return shot_action
        elif dribble_utility > MIN_COMMIT_THRESHOLD_MID:
             return midfielder_dribble_logic(obs, player_index, player_info)

    # Fallback: if a pass was found, even if low utility, consider it over IDLE for midfielders
    if pass_action is not None and pass_utility > -float('inf'):
        return pass_action
        
    return midfielder_dribble_logic(obs, player_index, player_info) # Default to trying to dribble/progress if idle


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
            not is_player_tired(obs, player_index, fatigue_threshold=None)):
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
    if distance_to_ball > 0.08 and not is_player_tired(obs, player_index, fatigue_threshold=None):
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
            if not is_player_tired(obs, player_index, fatigue_threshold=None):
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
    """攻击型中场的跑位 - 增强版"""
    player_pos = player_info['position']
    goal_center = [Field.RIGHT_GOAL_X, Field.CENTER_Y]
    ball_carrier_index = obs['ball_owned_player']
    ball_carrier_pos = obs['left_team'][ball_carrier_index] if ball_carrier_index != -1 else player_pos

    candidate_positions = []
    # 1. Exploit space between opponent lines (the "hole")
    # Target X: Slightly ahead of the ball, but not too far unless it's a clear break.
    # Target Y: Central, but can drift if space opens up.
    for x_offset in [0.05, 0.1, 0.15]:
        for y_offset in [-0.1, 0, 0.1]: # Check central and slightly off-center
            pos = [ball_carrier_pos[0] + x_offset, ball_carrier_pos[1] + y_offset]
            if Field.CENTER_X < pos[0] < (Field.RIGHT_GOAL_X - 0.15): # In attacking half, not too close to goal line yet
                candidate_positions.append(pos)

    # 2. Late runs into the box / penalty area edge if ball is wide or forward
    if ball_pos[0] > 0.5 or abs(ball_pos[1]) > 0.25: # Ball is advanced or wide
        for x_offset in [0.0, 0.05]: # Arrive at edge of box or slightly inside
             # Target X around penalty box edge or slightly inside
            target_x_box = Field.RIGHT_GOAL_X - 0.22 + x_offset
            for y_offset in [-0.15, -0.05, 0.05, 0.15]: # Central areas of box approach
                pos = [target_x_box, y_offset]
                candidate_positions.append(pos)

    # 3. Default support position if ball is deeper
    if not is_in_opponent_half(ball_pos):
        pos = [max(ball_pos[0] + 0.05, Field.CENTER_X - 0.1), player_pos[1]] # Support from behind/level
        candidate_positions.append(pos)
        pos_alt_y = [max(ball_pos[0] + 0.05, Field.CENTER_X - 0.1), ball_pos[1] * 0.5]
        candidate_positions.append(pos_alt_y)


    best_position = None
    best_score = -float('inf')

    for pos_eval in candidate_positions:
        # Basic validation
        if not (Field.LEFT_BOUNDARY < pos_eval[0] < Field.RIGHT_BOUNDARY and                 Field.TOP_BOUNDARY < pos_eval[1] < Field.BOTTOM_BOUNDARY):
            continue
        if is_offside_position(obs, pos_eval): # Use the helper from features
            continue

        score = 0.0
        # Scoring similar to forward's, but weights adjusted for AM role

        # a. Distance to goal (less aggressive than fwd directly, but still important)
        dist_to_goal = distance_to(pos_eval, goal_center)
        score += (1.5 - dist_to_goal) * 2.0

        # b. Safety from opponents
        min_dist_to_opp_at_pos = min((distance_to(pos_eval, opp_p) for opp_idx, opp_p in enumerate(obs['right_team']) if obs['right_team_active'][opp_idx]), default=0.2)
        score += min_dist_to_opp_at_pos * 3.0

        # c. Pass path clarity (from ball_carrier to this pos_eval)
        if ball_carrier_index != -1 and check_pass_path_clear(ball_carrier_pos, pos_eval, obs['right_team']):
            score += 1.5

        # d. Value being ahead of the ball (but not necessarily too far ahead)
        if pos_eval[0] > ball_pos[0] + 0.02:
            score += 1.0
        if pos_eval[0] > ball_carrier_pos[0] + 0.02: # Also ahead of carrier
             score += 1.0

        # e. Value being in a central attacking channel
        if abs(pos_eval[1]) < 0.20 and pos_eval[0] > Field.CENTER_X + 0.2: # Central & attacking
            score += 1.0

        # f. Movement distance
        move_dist = distance_to(player_pos, pos_eval)
        if move_dist > 0.2: score -= 0.5
        if move_dist < 0.03: score -= 0.5 # Avoid staying static unless already optimal

        if score > best_score:
            best_score = score
            best_position = pos_eval

    if best_position:
        if distance_to(player_pos, best_position) > 0.08 and not is_player_tired(obs, player_index, fatigue_threshold=None):
            return Action.SPRINT
        return get_movement_direction(player_pos, best_position)

    # Fallback if no good candidates
    fallback_target_x = min(ball_pos[0] + 0.1, Field.RIGHT_GOAL_X - 0.15)
    fallback_target_y = ball_pos[1] * 0.7
    if is_offside_position(obs, [fallback_target_x, fallback_target_y]):
        fallback_target_x = ball_pos[0] - 0.05 # Drop back if offside
    
    return get_movement_direction(player_pos, [fallback_target_x, fallback_target_y]) if distance_to(player_pos, [fallback_target_x, fallback_target_y]) > 0.02 else Action.IDLE


def wing_midfielder_movement(obs, player_index, player_info, ball_pos):
    """边路中场的跑位 - 增强版"""
    player_pos = player_info['position']
    player_role = player_info['role']
    goal_center = [Field.RIGHT_GOAL_X, Field.CENTER_Y]
    ball_carrier_index = obs['ball_owned_player']
    ball_carrier_pos = obs['left_team'][ball_carrier_index] if ball_carrier_index != -1 else player_pos

    # Determine preferred wide Y based on role
    preferred_y_wide = -0.30 if player_role == PlayerRole.LEFT_MIDFIELD else 0.30
    # Opposite side Y for diagonal considerations
    # opposite_y_wide = -preferred_y_wide

    candidate_positions = []

    # 1. Maintain width, especially if ball is central or on opposite flank
    if ball_pos[1] * preferred_y_wide <= 0: # Ball central or on other side
        for x_offset in [0.0, 0.05, 0.1]: # Level with ball or slightly ahead
            pos = [ball_carrier_pos[0] + x_offset, preferred_y_wide]
            if pos[0] < Field.RIGHT_GOAL_X - 0.1: # Don't run into goal line
                 candidate_positions.append(pos)

    # 2. Support on the flank if ball is on the same side
    if ball_pos[1] * preferred_y_wide > 0: # Ball on same side
        for x_offset in [-0.05, 0.05, 0.1, 0.15]: # Options to receive or make a run past
            pos = [ball_carrier_pos[0] + x_offset, player_pos[1]] # Maintain current Y or slightly adjust
            candidate_positions.append(pos)
            pos_wide = [ball_carrier_pos[0] + x_offset, preferred_y_wide * 0.8] # Offer wide option
            candidate_positions.append(pos_wide)

    # 3. Diagonal runs towards goal if space opens or ball is advanced
    if ball_pos[0] > 0.4: # If ball is well into opponent half
        diag_target_x = ball_pos[0] + 0.15
        diag_target_y = preferred_y_wide * 0.5 # Cut inwards
        pos = [diag_target_x, diag_target_y]
        candidate_positions.append(pos)
        # Also consider run towards far post area if ball is on flank for a cross
        if abs(ball_pos[1]) > 0.2:
            pos_far_post = [Field.RIGHT_GOAL_X - 0.1, preferred_y_wide * -0.3] # Target far post area
            candidate_positions.append(pos_far_post)


    # 4. Check for fullback overlap potential (simplified)
    # This is a very basic check. A full implementation would need more context.
    fullback_role = PlayerRole.LEFT_BACK if player_role == PlayerRole.LEFT_MIDFIELD else PlayerRole.RIGHT_BACK
    for fb_idx, fb_player_role in enumerate(obs['left_team_roles']):
        if fb_player_role == fullback_role and obs['left_team_active'][fb_idx]:
            fb_pos = obs['left_team'][fb_idx]
            # If fullback is ahead of winger and on the flank
            if fb_pos[0] > player_pos[0] and abs(fb_pos[1] - preferred_y_wide) < 0.1:
                # Winger could tuck in to create space for overlapping fullback
                pos_tuck_in = [player_pos[0] + 0.05, preferred_y_wide * 0.5]
                candidate_positions.append(pos_tuck_in)
                break

    best_position = None
    best_score = -float('inf')

    for pos_eval in candidate_positions:
        if not (Field.LEFT_BOUNDARY < pos_eval[0] < Field.RIGHT_BOUNDARY and                 Field.TOP_BOUNDARY < pos_eval[1] < Field.BOTTOM_BOUNDARY):
            continue
        if is_offside_position(obs, pos_eval):
            continue

        score = 0.0
        # a. Maintaining width / good crossing position
        if abs(pos_eval[1] - preferred_y_wide) < 0.1 and pos_eval[0] > 0.2: # Is wide and advanced
            score += 2.0
        
        # b. Safety & Space
        min_dist_to_opp_at_pos = min((distance_to(pos_eval, opp_p) for opp_idx, opp_p in enumerate(obs['right_team']) if obs['right_team_active'][opp_idx]), default=0.2)
        score += min_dist_to_opp_at_pos * 2.5

        # c. Progressing play / supporting ball carrier
        if pos_eval[0] > ball_carrier_pos[0] - 0.05 : # Not dropping too far back
            score += 1.0
            if pos_eval[0] > ball_carrier_pos[0] + 0.05: # Ahead of carrier
                 score += 1.0

        # d. Pass path clarity
        if ball_carrier_index != -1 and check_pass_path_clear(ball_carrier_pos, pos_eval, obs['right_team']):
            score += 1.0

        # e. Proximity to goal for diagonal runs
        if abs(pos_eval[1]) < abs(preferred_y_wide * 0.8): # If it's an inward run
            dist_to_goal = distance_to(pos_eval, goal_center)
            score += (1.5 - dist_to_goal) * 1.0

        move_dist = distance_to(player_pos, pos_eval)
        if move_dist > 0.25: score -= 0.5
        if move_dist < 0.03: score -= 0.5

        if score > best_score:
            best_score = score
            best_position = pos_eval

    if best_position:
        if distance_to(player_pos, best_position) > 0.1 and not is_player_tired(obs, player_index, fatigue_threshold=None):
            return Action.SPRINT
        return get_movement_direction(player_pos, best_position)

    # Fallback logic
    target_x = min(ball_pos[0] + 0.05, Field.RIGHT_GOAL_X - 0.2)
    fallback_y = preferred_y_wide
    if abs(ball_pos[1]) < 0.1: # Ball central
        fallback_y = preferred_y_wide
    elif ball_pos[1] * preferred_y_wide > 0: # Ball on same side
        fallback_y = player_pos[1] # Hold current Y or slightly adjust towards ball

    final_pos = [target_x, fallback_y]
    if is_offside_position(obs, final_pos):
        final_pos[0] = ball_pos[0] - 0.05

    return get_movement_direction(player_pos, final_pos) if distance_to(player_pos, final_pos) > 0.02 else Action.IDLE


def central_midfielder_movement(obs, player_index, player_info, ball_pos):
    """中中场的跑位 - 增强版"""
    player_pos = player_info['position']
    # player_role = player_info['role'] # Not used currently, but could be for DMF vs CMF
    goal_center = [Field.RIGHT_GOAL_X, Field.CENTER_Y]
    ball_carrier_index = obs['ball_owned_player']
    ball_carrier_pos = obs['left_team'][ball_carrier_index] if ball_carrier_index != -1 else player_pos

    candidate_positions = []

    # 1. Offer safe passing options to defenders/other midfielders
    if ball_carrier_pos[0] < player_pos[0] or ball_carrier_pos[0] < Field.CENTER_X: # If ball is behind or deep
        for x_offset in [-0.1, -0.05, 0.0]: # Slightly behind or level with ball carrier
            for y_offset in [-0.15, 0, 0.15]: # Lateral options
                pos = [ball_carrier_pos[0] + x_offset, ball_carrier_pos[1] + y_offset]
                if pos[0] > Field.LEFT_BOUNDARY + 0.1: # Not too deep
                    candidate_positions.append(pos)

    # 2. Support attacks from a slightly deeper position
    if ball_pos[0] > Field.CENTER_X - 0.1: # If ball is central or attacking
        # Position self to recycle possession or make a second wave attack
        target_x_support = ball_pos[0] - 0.1 # Slightly behind the ball
        for y_offset in [-0.2, 0, 0.2]:
            pos = [target_x_support, ball_pos[1] + y_offset]
            candidate_positions.append(pos)
        # Also consider positions slightly ahead if clear space to progress
        pos_fwd_support = [ball_pos[0] + 0.05, ball_pos[1]]
        candidate_positions.append(pos_fwd_support)


    # 3. Default central positioning based on ball X
    default_x = np.clip(ball_pos[0] - 0.05, Tactics.MID_BLOCK_X_THRESHOLD, Field.CENTER_X + 0.2)
    default_y_options = [ball_pos[1] * 0.3, player_pos[1] * 0.5 + ball_pos[1] * 0.5, 0.0] # Mix of following ball and central
    for dy_opt in default_y_options:
        candidate_positions.append([default_x, np.clip(dy_opt, Field.TOP_BOUNDARY+0.05, Field.BOTTOM_BOUNDARY-0.05)])


    best_position = None
    best_score = -float('inf')

    for pos_eval in candidate_positions:
        if not (Field.LEFT_BOUNDARY < pos_eval[0] < Field.RIGHT_BOUNDARY and                 Field.TOP_BOUNDARY < pos_eval[1] < Field.BOTTOM_BOUNDARY):
            continue
        # CMs should generally not be offside, but check anyway
        if is_offside_position(obs, pos_eval) and pos_eval[0] > Field.CENTER_X + 0.3:
            continue

        score = 0.0
        # a. Good support angle / Centrality
        score -= abs(pos_eval[1]) * 0.5 # Prefer central positions slightly
        if ball_carrier_index != -1:
            # Angle between player-ball_carrier and ball_carrier-target_pos should not be too small (open angle)
            # This is complex; simplified: just ensure not directly behind carrier unless far
            if distance_to(pos_eval, ball_carrier_pos) > 0.05:
                 score +=1.0

        # b. Safety and Space
        min_dist_to_opp_at_pos = min((distance_to(pos_eval, opp_p) for opp_idx, opp_p in enumerate(obs['right_team']) if obs['right_team_active'][opp_idx]), default=0.2)
        score += min_dist_to_opp_at_pos * 2.0

        # c. Pass path clarity (to receive)
        if ball_carrier_index != -1 and check_pass_path_clear(ball_carrier_pos, pos_eval, obs['right_team']):
            score += 1.5

        # d. Readiness to transition (not too far forward, not too far back)
        if Tactics.MID_BLOCK_X_THRESHOLD -0.1 < pos_eval[0] < Field.CENTER_X + 0.25:
            score += 1.0
        else:
            score -= 0.5 # Penalize being too deep or too far up for a CM

        move_dist = distance_to(player_pos, pos_eval)
        if move_dist > 0.2: score -= 0.5
        # if move_dist < 0.03: score -= 0.5 # Allow CM to hold good central position

        if score > best_score:
            best_score = score
            best_position = pos_eval

    if best_position:
        # CMs generally don't sprint as much offensively unless it's a clear counter
        if distance_to(player_pos, best_position) > 0.15 and not is_player_tired(obs, player_index, fatigue_threshold=None) and ball_pos[0] < 0: # Sprint if ball behind and need to catch up
            return Action.SPRINT
        return get_movement_direction(player_pos, best_position)

    # Fallback
    final_pos = [np.clip(ball_pos[0] - 0.1, Tactics.MID_BLOCK_X_THRESHOLD, Field.CENTER_X + 0.1), ball_pos[1] * 0.3]
    return get_movement_direction(player_pos, final_pos) if distance_to(player_pos, final_pos) > 0.02 else Action.IDLE


def should_midfielder_pressure(obs, player_index, ball_pos):
    """判断中场球员是否应该上抢 - 增强版"""
    player_info = get_player_info(obs, player_index)
    player_pos = player_info['position']

    if is_player_tired(obs, player_index, fatigue_threshold=0.75): # Midfielders get tired, stricter
        return False

    distance_to_ball = distance_to(player_pos, ball_pos)

    # Only press if midfielder is the closest or one of the closest to the ball
    if not is_closest_midfielder_to_ball(obs, player_index, ball_pos, proximity_factor=1.2): # check if among the closest
        return False

    # Pressing zones and distances
    if ball_pos[0] > 0.1: # Ball in our attacking third (opponent defending deep)
        if distance_to_ball < Distance.PRESSURE_DISTANCE * 1.8: # More aggressive press high up
            return True
    elif ball_pos[0] > -0.4: # Ball in midfield zone or opponent's build-up third
        if distance_to_ball < Distance.PRESSURE_DISTANCE * 1.5:
            # Check if ball carrier is isolated or has poor facing direction
            ball_carrier_idx = obs['ball_owned_player']
            if obs['ball_owned_team'] == 1 and ball_carrier_idx != -1: # Opponent has ball
                opp_dir = obs['right_team_direction'][ball_carrier_idx]
                if opp_dir[0] > 0: # Facing their own goal (good trigger)
                    return True
                # Check if ball carrier is slow/immobile (sticky actions might indicate this)
                # opp_sticky = obs['right_team_sticky_actions'][ball_carrier_idx]
                # if not any(opp_sticky[s] for s in [StickyActions.SPRINT, StickyActions.DRIBBLE]):
                #    return True # If carrier is static
            else: # Default for this zone
                 return True
    # In own defensive third (ball_pos[0] <= -0.4), midfielders are more cautious
    # Might press if ball is very close AND it's a safe press (e.g. to prevent shot)
    elif ball_pos[0] <= -0.4 and distance_to_ball < Distance.PRESSURE_DISTANCE * 1.0:
        # More checks could be added here for defensive safety
        return True

    return False


def is_closest_midfielder_to_ball(obs, player_index, ball_pos, proximity_factor=1.0):
    """判断是否是离球最近（或接近最近）的中场球员"""
    player_pos = obs['left_team'][player_index]
    player_distance = distance_to(player_pos, ball_pos)
    
    min_dist_any_midfielder = player_distance

    midfielder_roles = [
        PlayerRole.CENTRAL_MIDFIELD, PlayerRole.LEFT_MIDFIELD, 
        PlayerRole.RIGHT_MIDFIELD, PlayerRole.ATTACK_MIDFIELD, PlayerRole.DEFENCE_MIDFIELD # Include DMF
    ]
    
    for i, pos in enumerate(obs['left_team']):
        if not obs['left_team_active'][i]:
            continue
        role = obs['left_team_roles'][i]
        if role in midfielder_roles:
            dist = distance_to(pos, ball_pos)
            if dist < min_dist_any_midfielder:
                min_dist_any_midfielder = dist
    
    return player_distance <= min_dist_any_midfielder * proximity_factor


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