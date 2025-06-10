"""
前锋决策逻辑
"""
import numpy as np

from src.utils.features import (
    get_ball_info, get_player_info, distance_to,
    find_closest_teammate, find_closest_opponent,
    get_best_pass_target, get_movement_direction,
    is_player_tired, is_in_opponent_half, can_shoot,
    check_pass_path_clear, distance_to_line, is_offside_position, # Added for new score function
)
from src.utils.actions import action_manager, validate_action_for_situation
from src.gfootball_agent.config import Action, Distance, Field, PlayerRole


def _evaluate_shot_utility(obs, player_index, player_pos, ball_pos):
    """Evaluates the utility of taking a shot."""
    if not can_shoot(player_pos, ball_pos, obs): # Basic condition: can the player even attempt a shot?
        return -float('inf'), None # Action is None as it's not viable

    utility = 0.0
    goal_center = [Field.RIGHT_GOAL_X, Field.CENTER_Y]
    distance_to_goal = distance_to(player_pos, goal_center)

    # 1. Distance to goal (the closer, the much better)
    if distance_to_goal < Distance.OPTIMAL_SHOT_RANGE: # Optimal range
        utility += 15.0
    elif distance_to_goal < Distance.SHOT_RANGE: # Decent range
        utility += 7.0
        utility -= (distance_to_goal - Distance.OPTIMAL_SHOT_RANGE) * 50 # Penalize further distances within shot range
    else: # Outside effective shot range (should be caught by can_shoot, but as safeguard)
        return -float('inf'), None

    # 2. Shot Angle (degrees, smaller is better if direct, but can_shoot already filters bad angles)
    angle_deg = abs(calculate_shot_angle(player_pos, goal_center)) # calculate_shot_angle is in features
    if angle_deg < 15.0:
        utility += 3.0
    elif angle_deg < 30.0:
        utility += 1.5
    else: # angle > 30 is already borderline by can_shoot
        utility -= (angle_deg - 30.0) * 0.2


    # 3. Opponent Pressure / Blockers
    # Count opponents between player and goal center (simplified cone)
    opponents_blocking = 0
    for opp_idx, opp_pos_val in enumerate(obs['right_team']):
        if obs['right_team_active'][opp_idx]:
            # Check if opponent is in a cone towards goal and closer than goal
            if distance_to_line(np.array(player_pos), np.array(goal_center), np.array(opp_pos_val)) < 0.06: # Narrow cone
                 if distance_to(player_pos, opp_pos_val) < distance_to_goal :
                    opponents_blocking += 1
    utility -= opponents_blocking * 3.0 # Penalize for each blocker

    # 4. Player Role (Forwards are expected to shoot)
    # player_role = obs['left_team_roles'][player_index] # Not strictly needed for forward, they are a forward
    utility += 2.0 # Base bonus for being a forward considering a shot

    # 5. Current momentum / facing direction (simplified)
    player_dir = obs['left_team_direction'][player_index]
    # If player is generally moving/facing towards goal
    if player_dir[0] > 0.5: # Strong X-component towards opponent goal
        utility += 1.0

    # Minimum utility threshold for a shot to be considered somewhat viable
    # This is separate from comparing with other actions.
    MIN_VIABLE_SHOT_UTILITY = 5.0
    if utility < MIN_VIABLE_SHOT_UTILITY:
         return -float('inf'), None # Not a good shot even if it's the "best" of bad options

    return utility, Action.SHOT


def _evaluate_pass_utility(obs, player_index, player_pos):
    """Evaluates the utility of making a pass."""
    # get_best_pass_target now returns a score along with the target index.
    # We need to modify get_best_pass_target in features.py to return this score.
    # For now, let's assume it returns (best_target_index, pass_score)
    # And we'll need to determine pass type (short, long, high) based on best_target.

    # This function will be called AFTER get_best_pass_target is modified
    # to return a meaningful score. For this subtask, we'll use a placeholder
    # for pass_score and determine pass_type based on distance.

    best_target_index = get_best_pass_target(obs, player_index) # Assuming current version from features.py
                                                                # which was modified to return a better score.
                                                                # The "score" is implicit in its choice.
                                                                # We need to call it to get the target.

    if best_target_index == -1:
        return -float('inf'), None

    # The "utility" of the pass can be a scaled version of the score from get_best_pass_target
    # or re-evaluated here. Let's use the fact that get_best_pass_target picked this target.
    # We need a proxy for the quality of this chosen pass.
    # Let's calculate a simplified utility here based on the chosen target.

    utility = 0.0
    teammate_pos = obs['left_team'][best_target_index]
    pass_distance = distance_to(player_pos, teammate_pos)
    forward_progress = teammate_pos[0] - player_pos[0]

    # Base utility from choosing this pass (could be higher if get_best_pass_target score was available)
    utility += 5.0 # Base value for finding a pass target

    if forward_progress > 0.1: utility += forward_progress * 20.0 # Value forward passes
    else: utility -= abs(forward_progress) * 10.0 # Penalize backward

    if is_in_opponent_half(teammate_pos): utility += 2.0

    _ , dist_to_closest_opp_to_receiver = find_closest_opponent(obs, best_target_index)
    if dist_to_closest_opp_to_receiver < Distance.PRESSURE_DISTANCE * 1.5:
        utility -= 3.0

    pass_action = Action.SHORT_PASS # Default
    if pass_distance > Distance.SHORT_PASS_RANGE * 0.9: # Threshold might need tuning
        if forward_progress > 0.2 and abs(player_pos[1] - teammate_pos[1]) < 0.3 : # Long, relatively straight forward pass
            pass_action = Action.LONG_PASS
        elif forward_progress > 0.15 : # Could be a through ball or high ball to forward
             pass_action = Action.HIGH_PASS # Prefer high pass for longer forward options
        else: # If not very forward, but long, maybe short pass is safer if possible, or it's a bad option
             utility -= (pass_distance - Distance.SHORT_PASS_RANGE) * 10 # Penalize long non-progressive passes

    # If it's a very short pass, it's usually safe but not very progressive
    if pass_distance < Distance.SHORT_PASS_RANGE * 0.3:
        utility -= 1.0 # Slight penalty for very short, potentially pointless passes unless under pressure

    # MIN_VIABLE_PASS_UTILITY, e.g. to avoid passing to marked player if dribble is slightly better
    MIN_VIABLE_PASS_UTILITY = 2.0
    if utility < MIN_VIABLE_PASS_UTILITY:
        return -float('inf'), None

    return utility, pass_action


def _evaluate_dribble_utility(obs, player_index, player_pos):
    """Evaluates the utility of dribbling."""
    utility = 0.0

    # Check space for dribbling (e.g., using a simplified check or check_dribble_space from features)
    # For simplicity, let's check immediate surroundings
    _ , closest_opp_dist = find_closest_opponent(obs, player_index)

    if closest_opp_dist < Distance.BALL_CLOSE * 1.5 : # Very close opponent, risky dribble
        utility -= 5.0
    elif closest_opp_dist < Distance.PRESSURE_DISTANCE:
        utility -= 2.0
    else: # Some space
        utility += 3.0

    # Potential to progress towards goal
    # Check space towards opponent goal (e.g., positive X direction)
    # This is a very simplified check for forward dribble space.
    # A proper check_dribble_space(obs, player_index, direction_vector=[1,0]) would be better.
    # For now, assume if no immediate opponent in front, some utility.
    # Let's simulate a check for space towards goal:
    can_dribble_fwd = True
    # Check if an opponent is directly in front (within a narrow cone and short distance)
    for opp_idx, opp_p in enumerate(obs['right_team']):
        if obs['right_team_active'][opp_idx]:
            if opp_p[0] > player_pos[0] and abs(opp_p[1] - player_pos[1]) < 0.1 and                distance_to(player_pos, opp_p) < 0.1:
                can_dribble_fwd = False
                break
    if can_dribble_fwd:
        utility += 3.0
        if player_pos[0] > 0.0 : utility +=1.5 # Bonus for dribbling in opponent half
    else: # No clear path forward
        utility -= 2.0


    # Is the player a good dribbler? (Not available in obs, so assume generic ability)
    # Forwards might be slightly more encouraged to take on a player if space is tight elsewhere.
    utility += 1.0

    # MIN_VIABLE_DRIBBLE_UTILITY
    MIN_VIABLE_DRIBBLE_UTILITY = 1.0
    if utility < MIN_VIABLE_DRIBBLE_UTILITY:
        return -float('inf'), None

    return utility, Action.DRIBBLE # Action.DRIBBLE is a sticky action, actual movement decided later


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
    """前锋持球时的决策逻辑 - 基于效用评估"""
    player_pos = player_info['position']
    ball_pos = ball_info['position'] # ball_pos is same as player_pos when player has ball

    _ , closest_opponent_dist = find_closest_opponent(obs, player_index)
    under_pressure = closest_opponent_dist < Distance.PRESSURE_DISTANCE * 1.1 # Slightly more sensitive pressure check

    if under_pressure:
        # If under pressure, prioritize quick, safe actions.
        # Original forward_under_pressure logic: pass to support or protect ball.
        # Let's try to get a safe pass out first.
        pass_utility, pass_action = _evaluate_pass_utility(obs, player_index, player_pos)
        # Make pass utility much higher if it's a safe way out of pressure
        if pass_utility > -float('inf'): # if a pass target was found
             # Check if this pass is to a teammate who is NOT under pressure
            best_target_idx = get_best_pass_target(obs, player_index) # Re-call to get index for safety check
            if best_target_idx != -1:
                _ , dist_receiver_opp = find_closest_opponent(obs, best_target_idx)
                if dist_receiver_opp > Distance.PRESSURE_DISTANCE * 1.5: # Receiver is relatively free
                    pass_utility += 10.0 # Big bonus for safe pass under pressure
        
        if pass_utility > 0: # If a decent safe pass exists
             # Ensure pass_action is not None (it should be set if utility > -inf)
            return pass_action if pass_action is not None else Action.SHORT_PASS


        # If no good pass, try to protect or a very quick escape dribble (not a long run)
        # The original forward_protect_ball tried to find an escape direction
        # This is complex to score as a "dribble utility" in the same way.
        # For now, if pass fails under pressure, fall back to protect/simple escape.
        return forward_protect_ball(obs, player_index, player_info) # Keep original for now

    # Not under immediate pressure, evaluate all options
    shot_utility, shot_action = _evaluate_shot_utility(obs, player_index, player_pos, ball_pos)
    pass_utility, pass_action = _evaluate_pass_utility(obs, player_index, player_pos)
    dribble_utility, dribble_action = _evaluate_dribble_utility(obs, player_index, player_pos)

    # Debug: print utilities
    # print(f"P{player_index} Utilities: Shot={shot_utility:.2f}, Pass={pass_utility:.2f}, Dribble={dribble_utility:.2f}")

    # Decide based on utility
    # Max utility must be reasonably high to commit to an action
    MIN_COMMIT_THRESHOLD = 2.0 # General threshold for any action

    if shot_utility > pass_utility and shot_utility > dribble_utility and shot_utility > MIN_COMMIT_THRESHOLD:
        return shot_action
    elif pass_utility > shot_utility and pass_utility > dribble_utility and pass_utility > MIN_COMMIT_THRESHOLD:
        return pass_action
    elif dribble_utility > MIN_COMMIT_THRESHOLD : # Dribble if it's the best or only viable option above threshold
        # If dribble utility is highest, or others are too low.
        # Before returning Action.DRIBBLE, determine direction or use existing dribble logic.
        # The _evaluate_dribble_utility is very basic.
        # Let's use the original forward_create_opportunity's dribble logic if dribble is chosen.
        # This part needs careful integration.
        # For now, if dribble utility is highest and viable:
        if dribble_utility > pass_utility and dribble_utility > shot_utility:
            if can_dribble_towards_goal(obs, player_index, player_pos):
                 return forward_dribble_to_goal(obs, player_index, player_info) # Use existing smart dribble
            else: # If cannot dribble to goal, but general dribble utility was high (e.g. to create space)
                 return Action.DRIBBLE # Generic dribble, rely on movement logic to find direction
        elif pass_utility <= MIN_COMMIT_THRESHOLD and shot_utility <= MIN_COMMIT_THRESHOLD: # Dribble as last resort if viable
            if can_dribble_towards_goal(obs, player_index, player_pos):
                 return forward_dribble_to_goal(obs, player_index, player_info)
            else:
                 return Action.DRIBBLE


    # If no action has high enough utility, or under pressure and no good pass:
    # Fallback to a more general create_opportunity or hold ball.
    # The original forward_create_opportunity had its own if/else.
    # Let's try to pass as a default if a reasonable target exists, even if utility wasn't highest.
    if pass_action and pass_utility > -float('inf'): # Check if a pass target was found at all
        # This might be a pass that didn't win the utility contest but is better than IDLE
        # Re-evaluate if this pass is "good enough" as a fallback
        # For simplicity, if a pass target exists from _evaluate_pass_utility, take it.
        if pass_utility > -5.0 : # Arbitrary low threshold to ensure it's not a terrible pass idea
            return pass_action


    return Action.IDLE # Default if nothing else makes sense


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
    # goal_center = [Field.RIGHT_GOAL_X, Field.CENTER_Y] # Not directly used here now

    ball_info = get_ball_info(obs)
    ball_carrier_index = ball_info['owned_player']
    
    best_position_to_move_to = None

    if ball_carrier_index != -1 and obs['left_team_active'][ball_carrier_index]: # Check if ball carrier is active
        ball_carrier_pos = obs['left_team'][ball_carrier_index]
        # ball_carrier_role = obs['left_team_roles'][ball_carrier_index] # Not used currently

        #寻找最佳的接球位置，考虑创造传球线路
        #This function now uses the improved calculate_receiving_position_score
        best_position_to_move_to = find_best_receiving_position_enhanced(obs, player_index, ball_pos, ball_carrier_pos)
        
        if best_position_to_move_to:
            movement_action = get_movement_direction(player_pos, best_position_to_move_to)
            distance_to_target = distance_to(player_pos, best_position_to_move_to)

            if distance_to_target < 0.02 : # Already very close to optimal spot
                return Action.IDLE # Hold position if already optimal

            if distance_to_target > 0.08 and not is_player_tired(obs, player_index, fatigue_threshold=None):
                return Action.SPRINT
            
            if movement_action:
                return movement_action
            else:
                # 防御性编程：如果movement_action为None，返回IDLE
                return Action.IDLE

    # Fallback logic if no good receiving position from find_best_receiving_position_enhanced
    # or if ball_carrier is not valid.
    # Try to find a defensive gap to exploit if direct receiving spots are poor.
    if not best_position_to_move_to:
        target_position_gap = find_defensive_gap(obs, player_index, ball_pos)
        if target_position_gap:
            # Check if moving to this gap is reasonable
            if distance_to(player_pos, target_position_gap) < 0.3: # Not too far
                movement_action = get_movement_direction(player_pos, target_position_gap)
                if movement_action:
                    if distance_to(player_pos, target_position_gap) > 0.1 and not is_player_tired(obs, player_index, fatigue_threshold=None):
                        return Action.SPRINT
                    return movement_action
                else:
                    # 防御性编程：如果movement_action为None，返回IDLE
                    return Action.IDLE

    # Further fallback: Maintain a generally threatening position if other heuristics don't yield a good move.
    # Move towards an area slightly ahead of the ball, off-center to create options.
    # Avoids being directly behind the ball or too far wide initially.
    default_target_x = min(ball_pos[0] + 0.1, Field.RIGHT_GOAL_X - 0.10) # Stay ahead of ball, shy of goal line

    # Try to find a less crowded Y position
    optimal_y = calculate_optimal_y_position(obs, player_pos, ball_pos) # This function helps find less crowded lanes

    # If player is already far right/left, maybe come slightly central, or vice-versa
    if abs(player_pos[1]) > 0.3: # If player is very wide
        default_target_y = player_pos[1] * 0.7 # Come slightly more central
    elif abs(optimal_y - player_pos[1]) < 0.05 : # Already in a good y-lane
        default_target_y = player_pos[1]
    else:
        default_target_y = optimal_y

    # Ensure target is not offside
    final_target_pos = [default_target_x, default_target_y]
    if is_offside_position(obs, final_target_pos):
        # Adjust X to be onside with the second to last defender or ball X if ball is further
        last_def_x = Field.RIGHT_GOAL_X
        second_last_def_x = Field.RIGHT_GOAL_X
        opp_x_positions = sorted([opp[0] for opp_idx, opp in enumerate(obs['right_team']) if obs['right_team_active'][opp_idx]])
        if len(opp_x_positions) > 1:
            second_last_def_x = opp_x_positions[1] # second lowest X is second last defender towards our goal
        elif len(opp_x_positions) == 1:
            second_last_def_x = opp_x_positions[0]

        onside_x = min(second_last_def_x - 0.03, ball_pos[0] - 0.01 if ball_pos[0] < second_last_def_x else second_last_def_x - 0.03)
        final_target_pos[0] = min(final_target_pos[0], onside_x)


    if distance_to(player_pos, final_target_pos) > 0.02:
        movement_action = get_movement_direction(player_pos, final_target_pos)
        if movement_action:
            return movement_action
        else:
            # 防御性编程：如果movement_action为None，返回IDLE
            return Action.IDLE

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
    else:
        # 防御性编程：如果movement_action为None，返回IDLE
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
    else:
        # 防御性编程：如果movement_action为None，返回IDLE
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
        else:
            # 防御性编程：如果movement_action为None，返回IDLE
            return Action.IDLE
    
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
                # 防御性编程：如果movement_action为None，返回IDLE
                return Action.IDLE
        else:
            # 冲刺向球
            if not is_player_tired(obs, player_index, fatigue_threshold=None):
                return Action.SPRINT
    
    # 否则保持前场位置
    target_pos = get_forward_positioning(ball_pos)
    movement_action = get_movement_direction(player_pos, target_pos)
    
    if movement_action:
        return movement_action
    else:
        # 防御性编程：如果movement_action为None，返回IDLE
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


def calculate_receiving_position_score(obs, position_to_evaluate, current_player_pos, ball_carrier_pos, goal_center):
    """计算接球位置的评分 - 增强版"""
    score = 0.0 # Use float for scores
    
    # --- Factors related to the proposed 'position_to_evaluate' ---
    
    # 1. Distance to Goal (Major factor)
    distance_to_goal = distance_to(position_to_evaluate, goal_center)
    score += (1.5 - distance_to_goal) * 3.5 # Slightly increased weight

    # 2. Safety from Opponents at the position
    min_dist_to_opp_at_pos = float('inf')
    num_opp_very_close_at_pos = 0
    for opp_pos_idx, opp_pos_val in enumerate(obs['right_team']):
        if obs['right_team_active'][opp_pos_idx]:
            dist = distance_to(position_to_evaluate, opp_pos_val)
            if dist < min_dist_to_opp_at_pos:
                min_dist_to_opp_at_pos = dist
            if dist < Distance.PRESSURE_DISTANCE * 0.8: # How many opponents are extremely close
                num_opp_very_close_at_pos +=1

    score += min_dist_to_opp_at_pos * 4.5 # Increased weight for safety
    if num_opp_very_close_at_pos > 0:
        score -= num_opp_very_close_at_pos * 2.0 # Penalize if position is too crowded

    # 3. Clarity of Pass Path from Ball Carrier to 'position_to_evaluate'
    if check_pass_path_clear(ball_carrier_pos, position_to_evaluate, obs['right_team'], threshold=0.04): # Stricter threshold
        score += 2.5 # Increased bonus
    else:
        score -= 2.0 # Increased penalty

    # 4. Tactical Value of Position
    if position_to_evaluate[0] > ball_carrier_pos[0] + 0.03:  # Forward position relative to ball carrier
        score += 2.0 # Increased bonus
        if position_to_evaluate[0] > Field.CENTER_X + 0.25 : # Deep in opponent's half
             score += 1.0

    # 5. Offside Check (Crucial)
    if is_offside_position(obs, position_to_evaluate): # Assumes is_offside_position is robust
        score -= 10.0 # Heavy penalty for offside positions to strongly discourage them

    # 6. Shot Angle from 'position_to_evaluate'
    # Assumes calculate_shot_angle returns degrees
    shot_angle = calculate_shot_angle(position_to_evaluate, goal_center)
    if shot_angle < 35.0: # Good angle
        score += 1.5
    elif shot_angle > 60.0: # Poor angle
        score -= 0.5

    # --- Factors related to the 'current_player_pos' moving to 'position_to_evaluate' ---

    # 7. Movement Feasibility / Path to Position
    # Is the path for the *player* to move to this position relatively clear?
    # This is a simplified check, focusing on immediate obstacles.
    num_opp_on_path_to_pos = 0
    for opp_idx, opp_pos_val in enumerate(obs['right_team']):
        if obs['right_team_active'][opp_idx]:
            if distance_to_line(np.array(current_player_pos), np.array(position_to_evaluate), np.array(opp_pos_val)) < 0.05:
                 # Check if opponent is actually between current and target
                player_to_target_vector = np.array(position_to_evaluate) - np.array(current_player_pos)
                player_to_opp_vector = np.array(opp_pos_val) - np.array(current_player_pos)
                if np.dot(player_to_target_vector, player_to_opp_vector) > 0 and np.linalg.norm(player_to_opp_vector) < np.linalg.norm(player_to_target_vector):
                    num_opp_on_path_to_pos += 1
    score -= num_opp_on_path_to_pos * 1.0

    # 8. Movement Distance
    move_distance = distance_to(current_player_pos, position_to_evaluate)
    if move_distance > 0.25: # Too far to move quickly
        score -= 1.5
    elif move_distance < 0.03: # Too close, already there or negligible move
        score -= 1.0 # Slight penalty to encourage seeking better new positions unless current is great

    # 9. Creating Space / Dragging Defenders
    # If the position is further from other teammates (especially ball carrier), it might create space.
    # This is a simple heuristic.
    avg_dist_to_teammates = 0
    active_teammates = 0
    # Need to get player_index for the current player to exclude self if needed
    # This assumes player_index for 'current_player_pos' is available or implicitly handled
    # For now, let's assume obs['ball_owned_player'] is not the current player for this calculation
    # or that the current player is identified by another means if not the ball carrier.
    # This logic might need player_index of the forward making the run.
    # Simplified: exclude ball carrier from "other teammates" for space creation assessment
    for tm_idx, tm_pos in enumerate(obs['left_team']):
        if obs['left_team_active'][tm_idx] and tm_idx != obs['ball_owned_player'] :
            avg_dist_to_teammates += distance_to(position_to_evaluate, tm_pos)
            active_teammates +=1
    if active_teammates > 0:
        avg_dist_to_teammates /= active_teammates
        if avg_dist_to_teammates > 0.2: # Position is relatively isolated from other teammates
            score += 0.5 # Small bonus for potentially dragging defenders or finding open space

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
        # 如果防守球员不足，返回一个默认的前插位置，而不是None
        default_x = min(ball_pos[0] + 0.1, Field.RIGHT_GOAL_X - 0.15)
        default_y = ball_pos[1] * 0.5  # 轻微跟随球的横向位置
        return [default_x, default_y]
    
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
    
    # 如果没有找到合适的空隙，返回默认位置
    if best_gap is None:
        default_x = min(ball_pos[0] + 0.08, Field.RIGHT_GOAL_X - 0.15)
        default_y = 0  # 中路位置
        return [default_x, default_y]
    
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
    """判断前锋是否应该逼抢 - 增强版"""
    player_info = get_player_info(obs, player_index)
    player_pos = player_info['position']

    if is_player_tired(obs, player_index, fatigue_threshold=0.7): # Higher fatigue tolerance for forward press
        return False # Don't press if very tired

    distance_to_ball = distance_to(player_pos, ball_pos)
    
    # Base condition: Ball in opponent's territory, and forward is reasonably close
    press_condition = False
    if ball_pos[0] < -0.25 and distance_to_ball < 0.25: # Slightly wider X range, slightly larger distance
        press_condition = True
    elif ball_pos[0] < 0.0 and distance_to_ball < 0.15: # Press more aggressively if ball is closer to center in opp half
        press_condition = True

    if not press_condition:
        return False

    # Bonus for support: Check if other forwards/attacking midfielders are also relatively high up
    num_supporting_attackers = 0
    for i, teammate_pos in enumerate(obs['left_team']):
        if i == player_index or not obs['left_team_active'][i]:
            continue
        role = obs['left_team_roles'][i]
        if role in [PlayerRole.CENTRAL_FORWARD, PlayerRole.ATTACK_MIDFIELD] and teammate_pos[0] > -0.5:
            if distance_to(player_pos, teammate_pos) < 0.3: # Teammate is somewhat nearby
                 num_supporting_attackers +=1
    
    if num_supporting_attackers > 0 and distance_to_ball < 0.22: # Slightly more willing if supported
        return True

    # Default to press_condition if no specific support found but base conditions met
    return press_condition


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