"""
动作管理模块 - 封装动作ID和粘性动作逻辑
"""

from src.gfootball_agent.config import Action, StickyActions


class ActionManager:
    """动作管理器 - 处理粘性动作和动作转换"""
    
    def __init__(self):
        self.last_actions = {}  # 记录每个球员的上一个动作
    
    def get_action_with_sticky_management(self, player_index, desired_action, obs):
        """
        根据期望动作和当前粘性动作状态，返回实际应该执行的动作
        """
        current_sticky = obs['sticky_actions']
        
        # 如果期望动作是IDLE，保持当前状态
        if desired_action == Action.IDLE:
            return Action.IDLE
        
        # 处理移动动作
        if self._is_movement_action(desired_action):
            return self._handle_movement_action(desired_action, current_sticky)
        
        # 处理冲刺动作
        if desired_action == Action.SPRINT:
            return self._handle_sprint_action(current_sticky)
        
        # 处理停止冲刺
        if desired_action == Action.RELEASE_SPRINT:
            return Action.RELEASE_SPRINT
        
        # 处理盘带动作
        if desired_action == Action.DRIBBLE:
            return self._handle_dribble_action(current_sticky)
        
        # 处理停止盘带
        if desired_action == Action.RELEASE_DRIBBLE:
            return Action.RELEASE_DRIBBLE
        
        # 处理停止移动
        if desired_action == Action.RELEASE_DIRECTION:
            return Action.RELEASE_DIRECTION
        
        # 其他动作（传球、射门、铲球等）直接执行
        return desired_action
    
    def _is_movement_action(self, action):
        """判断是否为移动动作"""
        movement_actions = [
            Action.LEFT, Action.TOP_LEFT, Action.TOP, Action.TOP_RIGHT,
            Action.RIGHT, Action.BOTTOM_RIGHT, Action.BOTTOM, Action.BOTTOM_LEFT
        ]
        return action in movement_actions
    
    def _handle_movement_action(self, desired_action, current_sticky):
        """处理移动动作"""
        # 检查当前是否已经在执行这个方向的移动
        desired_sticky_index = self._action_to_sticky_index(desired_action)
        
        if desired_sticky_index is not None and current_sticky[desired_sticky_index]:
            # 已经在执行期望的移动方向，返回IDLE保持状态
            return Action.IDLE
        else:
            # 需要改变移动方向或开始移动
            return desired_action
    
    def _handle_sprint_action(self, current_sticky):
        """处理冲刺动作"""
        if current_sticky[StickyActions.SPRINT]:
            # 已经在冲刺，返回IDLE保持状态
            return Action.IDLE
        else:
            # 开始冲刺
            return Action.SPRINT
    
    def _handle_dribble_action(self, current_sticky):
        """处理盘带动作"""
        if current_sticky[StickyActions.DRIBBLE]:
            # 已经在盘带，返回IDLE保持状态
            return Action.IDLE
        else:
            # 开始盘带
            return Action.DRIBBLE
    
    def _action_to_sticky_index(self, action):
        """将动作转换为对应的粘性动作索引"""
        action_to_sticky_map = {
            Action.LEFT: StickyActions.LEFT,
            Action.TOP_LEFT: StickyActions.TOP_LEFT,
            Action.TOP: StickyActions.TOP,
            Action.TOP_RIGHT: StickyActions.TOP_RIGHT,
            Action.RIGHT: StickyActions.RIGHT,
            Action.BOTTOM_RIGHT: StickyActions.BOTTOM_RIGHT,
            Action.BOTTOM: StickyActions.BOTTOM,
            Action.BOTTOM_LEFT: StickyActions.BOTTOM_LEFT,
        }
        return action_to_sticky_map.get(action)
    
    def should_stop_current_movement(self, current_sticky):
        """判断是否应该停止当前移动"""
        # 检查是否有移动方向的粘性动作处于激活状态
        for sticky_idx in StickyActions.MOVEMENT_ACTIONS:
            if current_sticky[sticky_idx]:
                return True
        return False
    
    def get_current_movement_direction(self, current_sticky):
        """获取当前移动方向"""
        for sticky_idx in StickyActions.MOVEMENT_ACTIONS:
            if current_sticky[sticky_idx]:
                return self._sticky_index_to_action(sticky_idx)
        return None
    
    def _sticky_index_to_action(self, sticky_index):
        """将粘性动作索引转换为对应的动作"""
        sticky_to_action_map = {
            StickyActions.LEFT: Action.LEFT,
            StickyActions.TOP_LEFT: Action.TOP_LEFT,
            StickyActions.TOP: Action.TOP,
            StickyActions.TOP_RIGHT: Action.TOP_RIGHT,
            StickyActions.RIGHT: Action.RIGHT,
            StickyActions.BOTTOM_RIGHT: Action.BOTTOM_RIGHT,
            StickyActions.BOTTOM: Action.BOTTOM,
            StickyActions.BOTTOM_LEFT: Action.BOTTOM_LEFT,
        }
        return sticky_to_action_map.get(sticky_index, Action.IDLE)


def combine_actions(primary_action, secondary_action=None):
    """
    组合动作（如移动+冲刺）
    注意：在GRF中，某些动作可以同时执行，但这里简化为优先级处理
    """
    # 主要动作优先
    if primary_action in [Action.SHOT, Action.LONG_PASS, Action.HIGH_PASS, Action.SHORT_PASS, Action.SLIDING]:
        return primary_action
    
    # 如果有辅助动作且主动作是移动，可以考虑组合
    if secondary_action == Action.SPRINT and primary_action != Action.IDLE:
        # 在某些情况下可能需要先冲刺再移动，但这里简化处理
        return primary_action
    
    return primary_action


def is_ball_action(action):
    """判断是否为与球相关的动作"""
    ball_actions = [
        Action.SHOT, Action.LONG_PASS, Action.HIGH_PASS, 
        Action.SHORT_PASS, Action.DRIBBLE, Action.RELEASE_DRIBBLE
    ]
    return action in ball_actions


def is_defensive_action(action):
    """判断是否为防守动作"""
    defensive_actions = [Action.SLIDING]
    return action in defensive_actions


def validate_action_for_situation(action, obs, player_index):
    """
    验证动作是否适用于当前情况
    返回: (is_valid, corrected_action)
    """
    ball_owned_team = obs['ball_owned_team']
    ball_owned_player = obs['ball_owned_player']
    
    # 球相关动作需要球员持球
    if is_ball_action(action):
        if ball_owned_team != 0 or ball_owned_player != player_index:
            # 球员没有控球，不能执行球相关动作
            return False, Action.IDLE
    
    # 防守动作不应该在控球时使用
    if is_defensive_action(action):
        if ball_owned_team == 0 and ball_owned_player == player_index:
            # 球员控球时不应该铲球
            return False, Action.IDLE
    
    return True, action


# 创建全局动作管理器实例
action_manager = ActionManager() 