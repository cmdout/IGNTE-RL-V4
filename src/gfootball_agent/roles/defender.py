"""
后卫的决策逻辑
""" 
import numpy as np

from src.utils.features import (
    get_ball_info, get_player_info, distance_to, 
    get_defensive_position,
    find_closest_opponent, get_best_pass_target,
    get_movement_direction, is_player_tired,
    is_in_opponent_half, can_shoot,
    check_pass_path_clear, # Added for utility functions
    check_dribble_space, is_safe_to_clear_ball # Added for utility functions
)
from src.utils.actions import action_manager, validate_action_for_situation
from src.gfootball_agent.config import Action, Distance, Field, PlayerRole, Tactics


def _evaluate_shot_utility_def(obs, player_index, player_pos, ball_pos, player_role):
    """Evaluates the utility of taking a shot for a defender. Should be extremely low."""
    # Defenders generally should not shoot.
    # Only consider if player is somehow extremely far up and in a miracle situation.
    if player_pos[0] > 0.6 and can_shoot(player_pos, ball_pos, obs): # Very advanced for a defender
        # Extremely basic utility, still very low to discourage it.
        distance_to_goal = distance_to(player_pos, [Field.RIGHT_GOAL_X, Field.CENTER_Y])
        if distance_to_goal < Distance.SHOT_RANGE:
            return 1.0 - (distance_to_goal * 5), Action.SHOT # Tiny utility
    return -float('inf'), None


def _evaluate_pass_utility_def(obs, player_index, player_pos, player_role):
    """Evaluates the utility of making a pass for a defender."""
    best_target_index = get_best_pass_target(obs, player_index)

    if best_target_index == -1:
        return -float('inf'), None

    utility = 10.0 # Base utility: passing is default good action for defenders
    teammate_pos = obs['left_team'][best_target_index]
    teammate_role = obs['left_team_roles'][best_target_index]
    pass_distance = distance_to(player_pos, teammate_pos)
    forward_progress = teammate_pos[0] - player_pos[0]

    # Safety is paramount for defenders
    _ , dist_to_closest_opp_to_receiver = find_closest_opponent(obs, best_target_index)
    if dist_to_closest_opp_to_receiver < Distance.PRESSURE_DISTANCE * 1.5: # Receiver under some pressure
        utility -= 5.0
    if dist_to_closest_opp_to_receiver < Distance.BALL_CLOSE: # Receiver under heavy pressure
        utility -= 10.0

    # Path clarity
    if not check_pass_path_clear(player_pos, teammate_pos, obs['right_team'], threshold=0.035): # Stricter for defenders
        utility -= 8.0

    # Prefer passes to midfielders or fullbacks if building out
    if teammate_role in [PlayerRole.CENTRAL_MIDFIELD, PlayerRole.DEFENCE_MIDFIELD, PlayerRole.ATTACK_MIDFIELD, PlayerRole.LEFT_MIDFIELD, PlayerRole.RIGHT_MIDFIELD]:
        utility += 3.0
    elif teammate_role in [PlayerRole.LEFT_BACK, PlayerRole.RIGHT_BACK] and teammate_role != player_role: # Pass to other fullback
        utility += 2.0
    elif teammate_role == PlayerRole.GOALKEEPER: # Backpass to GK
        if player_pos[0] < Field.LEFT_GOAL_X + 0.2: # Only if deep and GK is clear
             _ , gk_opp_dist = find_closest_opponent(obs, best_target_index)
             if gk_opp_dist > 0.15 : utility += 1.0 # Risky, small utility
             else: utility -= 10.0 # Very risky
        else: utility -= 5.0 # Generally avoid passing back to GK from higher up
    elif teammate_role == PlayerRole.CENTRAL_FORWARD: # Long ball to forward
        if forward_progress > 0.3 and pass_distance > 0.3: # Must be a clear long ball
            utility += 1.0 # Small bonus, can be an option
        else:
            utility -= 3.0 # Short passes to heavily marked forward by defender is usually bad

    # Penalize risky forward passes from deep
    if forward_progress > 0.1 and player_pos[0] < -0.5: # Forward pass from deep defense
        utility -= (forward_progress * 15.0) # Moderate penalty, prefer safer build-up

    pass_action = Action.SHORT_PASS
    if pass_distance > Distance.SHORT_PASS_RANGE:
        if forward_progress > 0.25 and teammate_role == PlayerRole.CENTRAL_FORWARD : # Long ball to forward
            pass_action = Action.HIGH_PASS
        elif forward_progress > 0.15 : # Other constructive long passes
            pass_action = Action.LONG_PASS
        else: # Long sideways/backward passes from defenders are usually not great unless clearing
            utility -= (pass_distance - Distance.SHORT_PASS_RANGE) * 10
            pass_action = Action.LONG_PASS # Default to long if it's far.

    # MIN_VIABLE_PASS_UTILITY_DEF - defenders should try to make a pass if one is remotely sensible
    MIN_VIABLE_PASS_UTILITY_DEF = -5.0 # Can be negative if it's better than losing ball
    if utility < MIN_VIABLE_PASS_UTILITY_DEF and not (pass_action == Action.HIGH_PASS and forward_progress > 0.3) : # Exception for clearances
         return -float('inf'), None # Not a good pass

    return utility, pass_action


def _evaluate_dribble_utility_def(obs, player_index, player_pos, player_role):
    """Evaluates the utility of dribbling for a defender."""
    utility = 0.0
    _ , closest_opp_dist = find_closest_opponent(obs, player_index)

    if closest_opp_dist < Distance.BALL_CLOSE * 1.1: utility -= 10.0 # Very risky
    elif closest_opp_dist < Distance.PRESSURE_DISTANCE * 0.9: utility -= 7.0
    else: utility += 1.0 # Some space is a small plus, but not primary goal

    has_fwd_space, space_dist = check_dribble_space(obs, player_index, direction_vector=[0.1,0.0])
    if has_fwd_space and space_dist > 0.1: # Significant clear space ahead
        utility += 3.0
        if player_role in [PlayerRole.LEFT_BACK, PlayerRole.RIGHT_BACK] and abs(player_pos[1]) > 0.15: # Fullbacks on flank
            utility += 2.0
    else: # No clear forward space
        utility -= 3.0

    if player_role == PlayerRole.CENTRE_BACK:
        utility -= 5.0 # CBs strongly discouraged from dribbling

    # If player is very deep in own half, dribbling is extra risky
    if player_pos[0] < Field.LEFT_GOAL_X + 0.2:
        utility -= 4.0

    MIN_VIABLE_DRIBBLE_UTILITY_DEF = -2.0 # Defender dribble must be clearly safe or better than terrible pass
    if utility < MIN_VIABLE_DRIBBLE_UTILITY_DEF:
         return -float('inf'), None

    return utility, Action.DRIBBLE


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
    """后卫持球时的决策逻辑 - 基于效用评估"""
    player_pos = player_info['position']
    ball_pos = player_info['position']
    player_role = player_info['role']

    # If under severe pressure, the original defender_under_pressure is good (prioritizes safety/clearance)
    _ , closest_opponent_dist = find_closest_opponent(obs, player_index)
    if closest_opponent_dist < Distance.PRESSURE_DISTANCE * 0.8 : # Very tight pressure
        # print(f"Defender {player_index} under severe pressure, using original logic.")
        return defender_under_pressure(obs, player_index, ball_info, player_info) # Original handles this well

    # Evaluate options if not under extreme immediate pressure
    # Shot utility is practically nil for defenders, but we can call it for completeness.
    shot_utility, shot_action = _evaluate_shot_utility_def(obs, player_index, player_pos, ball_pos, player_role)
    pass_utility, pass_action = _evaluate_pass_utility_def(obs, player_index, player_pos, player_role)
    dribble_utility, dribble_action = _evaluate_dribble_utility_def(obs, player_index, player_pos, player_role)
    
    # print(f"P{player_index}({player_role}) Utils: Shot={shot_utility:.1f}, Pass={pass_utility:.1f}, Dribble={dribble_utility:.1f}")

    # Defenders: Pass > Safe Dribble > Clearance (High Pass) > Risky Dribble / Bad Pass > Shot
    MIN_COMMIT_THRESHOLD_DEF_PASS = 0.0 # Prefer any decent pass
    MIN_COMMIT_THRESHOLD_DEF_DRIBBLE = 1.0 # Dribble needs some positive utility

    if pass_utility >= dribble_utility and pass_utility > MIN_COMMIT_THRESHOLD_DEF_PASS:
        return pass_action
    # Consider dribble only if it's clearly better than a bad pass or no pass
    elif dribble_utility > pass_utility and dribble_utility > MIN_COMMIT_THRESHOLD_DEF_DRIBBLE :
        # Use existing defender_dribble_forward for controlled dribbling
        return defender_dribble_forward(obs, player_index, player_info)
    elif pass_utility > -float('inf') and pass_action is not None: # Fallback to any available pass
        return pass_action
    
    # If no good pass or dribble, and not under EXTREME pressure (handled above),
    # but still might be risky, consider a clearance type high pass.
    # The _evaluate_pass_utility_def might return a HIGH_PASS with decent utility if it's a long forward clearance.
    # If pass_utility was low but action was HIGH_PASS, it might be a clearance attempt.
    if pass_action == Action.HIGH_PASS and pass_utility > -3.0 : # Low utility but it's a clearance
        return Action.HIGH_PASS

    # Ultimate fallback: if original defender_under_pressure wasn't triggered but situation is still bad.
    # This could be a less immediate pressure but no good options.
    if is_safe_to_clear_ball(obs, player_index): # from features.py
         return Action.HIGH_PASS

    return defender_dribble_forward(obs, player_index, player_info) # Last resort: try to carry forward a bit.


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
            not is_player_tired(obs, player_index, fatigue_threshold=None)):
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
    if distance_to_ball > 0.05 and not is_player_tired(obs, player_index, fatigue_threshold=None):
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
    if distance_to(player_pos, target_pos) > 0.1 and not is_player_tired(obs, player_index, fatigue_threshold=None):
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
    """判断后卫是否应该上抢 - 增强版"""
    player_info = get_player_info(obs, player_index)
    player_pos = player_info['position']
    player_role = player_info['role']

    if is_player_tired(obs, player_index, fatigue_threshold=0.8): # Defenders press less when tired
        return False

    distance_to_ball = distance_to(player_pos, ball_pos)

    # If ball is very far, or defender is CB and ball is wide and not deep, don't press
    if distance_to_ball > Distance.PRESSURE_DISTANCE * 2.5:
        return False
    if player_role == PlayerRole.CENTRE_BACK and abs(ball_pos[1]) > 0.3 and ball_pos[0] > -0.5: # CB less likely to press wide unless ball is deep
        if distance_to_ball > Distance.PRESSURE_DISTANCE * 1.5:
             return False

    # Check if this defender is the "designated presser" or has good immediate support
    # This simplifies to checking if they are clearly the closest or have a nearby defensive partner
    is_closest_def = is_closest_defender_to_ball(obs, player_index, ball_pos, proximity_factor=1.1)

    if not is_closest_def :
        # If not the absolute closest, only press if very close and ball is in dangerous spot
        if not (distance_to_ball < Distance.BALL_CLOSE and ball_pos[0] < Field.LEFT_GOAL_X + 0.3): # Danger zone
            return False

    # If ball is central and near penalty box, CBs might step up
    if abs(ball_pos[1]) < 0.20 and ball_pos[0] < Field.LEFT_GOAL_X + 0.25: # Central, ~18yd box edge
        if distance_to_ball < Distance.PRESSURE_DISTANCE * 1.2:
            return True

    # Fullbacks can be slightly more aggressive, especially if ball is on their flank
    if player_role in [PlayerRole.LEFT_BACK, PlayerRole.RIGHT_BACK]:
        is_ball_on_flank = (ball_pos[1] < -0.15 and player_role == PlayerRole.LEFT_BACK) or                            (ball_pos[1] > 0.15 and player_role == PlayerRole.RIGHT_BACK)
        if is_ball_on_flank and distance_to_ball < Distance.PRESSURE_DISTANCE * 1.8:
            return True

    # General condition for CBs or if other conditions not met
    if distance_to_ball < Distance.PRESSURE_DISTANCE * 1.5:
        # Check for cover. If no other defender is relatively close behind, be cautious.
        num_covering_teammates = 0
        for i, teammate_pos in enumerate(obs['left_team']):
            if i == player_index or not obs['left_team_active'][i]:
                continue
            tm_role = obs['left_team_roles'][i]
            if tm_role in [PlayerRole.CENTRE_BACK, PlayerRole.LEFT_BACK, PlayerRole.RIGHT_BACK, PlayerRole.DEFENCE_MIDFIELD]:
                if teammate_pos[0] < player_pos[0] and distance_to(player_pos, teammate_pos) < 0.15:
                    num_covering_teammates +=1

        if num_covering_teammates > 0 or player_pos[0] < Field.LEFT_GOAL_X + 0.2: # Press if covered or very deep
            return True

    return False


def is_closest_defender_to_ball(obs, player_index, ball_pos, proximity_factor=1.0):
    """判断是否是离球最近（或接近最近）的后卫"""
    player_pos = obs['left_team'][player_index]
    player_distance = distance_to(player_pos, ball_pos)
    min_dist_any_defender = player_distance

    defender_roles = [PlayerRole.CENTRE_BACK, PlayerRole.LEFT_BACK, PlayerRole.RIGHT_BACK]
    
    for i, pos in enumerate(obs['left_team']):
        if not obs['left_team_active'][i]:
            continue
        role = obs['left_team_roles'][i]
        if role in defender_roles:
            dist = distance_to(pos, ball_pos)
            if dist < min_dist_any_defender:
                min_dist_any_defender = dist

    return player_distance <= min_dist_any_defender * proximity_factor


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