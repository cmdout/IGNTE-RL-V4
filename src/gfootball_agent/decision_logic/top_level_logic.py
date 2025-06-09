"""
顶层决策逻辑 - 根据游戏模式分发任务
"""

from src.gfootball_agent.config import GameMode
from src.gfootball_agent.decision_logic.normal_mode import normal_mode_decision
from src.gfootball_agent.decision_logic.set_pieces import set_piece_decision


def get_player_action(obs, player_index):
    """
    顶层决策函数 - 根据当前游戏模式分发到对应的决策逻辑
    
    参数:
        obs: 观测数据
        player_index: 球员索引
    
    返回:
        action: 球员应该执行的动作
    """
    game_mode = obs['game_mode']
    
    # 根据游戏模式选择对应的决策逻辑
    if game_mode == GameMode.NORMAL:
        # 常规比赛模式
        return normal_mode_decision(obs, player_index)
    else:
        # 定位球模式（开球、球门球、任意球、角球、界外球、点球）
        return set_piece_decision(obs, player_index) 