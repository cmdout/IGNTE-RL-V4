"""
定义核心Agent类,负责调用决策逻辑
""" 

from src.gfootball_agent.decision_logic.top_level_logic import get_player_action
from src.utils.actions import action_manager, validate_action_for_situation


class FootballAgent:
    """
    足球智能体核心类
    负责管理11名球员的决策并返回动作数组
    """
    
    def __init__(self):
        """初始化智能体"""
        self.team_size = 11
        self.action_history = {}  # 记录每个球员的动作历史
        
    def get_actions(self, obs_list):
        """
        获取所有球员的动作
        
        参数:
            obs_list: 观测数据列表，每个元素对应一个球员的观测
        
        返回:
            actions: 动作列表，每个元素对应一个球员的动作
        """
        actions = []
        
        # 为每个球员生成动作
        for player_index in range(self.team_size):
            if player_index < len(obs_list):
                obs = obs_list[player_index]
                action = self._get_single_player_action(obs, player_index)
                actions.append(action)
            else:
                # 如果观测数据不足，返回默认动作
                actions.append(0)  # IDLE
        
        return actions
    
    def _get_single_player_action(self, obs, player_index):
        """
        获取单个球员的动作
        
        参数:
            obs: 球员的观测数据
            player_index: 球员索引
        
        返回:
            action: 球员应该执行的动作
        """
        # 调用顶层决策逻辑获取期望动作
        desired_action = get_player_action(obs, player_index)
        
        # 验证动作的合法性
        is_valid, corrected_action = validate_action_for_situation(
            desired_action, obs, player_index
        )
        
        if not is_valid:
            desired_action = corrected_action
        
        # 通过动作管理器处理粘性动作
        final_action = action_manager.get_action_with_sticky_management(
            player_index, desired_action, obs
        )
        
        # 记录动作历史
        self._record_action_history(player_index, final_action)
        
        return final_action
            
    
    def _record_action_history(self, player_index, action):
        """
        记录球员的动作历史
        
        参数:
            player_index: 球员索引
            action: 执行的动作
        """
        if player_index not in self.action_history:
            self.action_history[player_index] = []
        
        self.action_history[player_index].append(action)
        
        # 只保留最近的10个动作
        if len(self.action_history[player_index]) > 10:
            self.action_history[player_index].pop(0)
    
    def get_action_history(self, player_index):
        """
        获取球员的动作历史
        
        参数:
            player_index: 球员索引
        
        返回:
            history: 动作历史列表
        """
        return self.action_history.get(player_index, [])
    
    def reset(self):
        """重置智能体状态"""
        self.action_history.clear()


# 创建全局智能体实例
agent = FootballAgent() 