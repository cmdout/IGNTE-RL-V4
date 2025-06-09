"""
常规比赛模式决策逻辑
"""

from src.gfootball_agent.config import PlayerRole
from src.gfootball_agent.roles.goalkeeper import goalkeeper_decision
from src.gfootball_agent.roles.defender import defender_decision
from src.gfootball_agent.roles.midfielder import midfielder_decision
from ..roles.forward import forward_decision


def normal_mode_decision(obs, player_index):
    """
    常规比赛模式下的球员决策
    根据球员角色分发到对应的决策函数
    """
    player_role = obs['left_team_roles'][player_index]
    
    # 根据球员角色选择对应的决策逻辑
    if player_role == PlayerRole.GOALKEEPER:
        return goalkeeper_decision(obs, player_index)
    
    elif player_role in [PlayerRole.CENTRE_BACK, PlayerRole.LEFT_BACK, PlayerRole.RIGHT_BACK]:
        return defender_decision(obs, player_index)
    
    elif player_role in [PlayerRole.CENTRAL_MIDFIELD, PlayerRole.LEFT_MIDFIELD, 
                         PlayerRole.RIGHT_MIDFIELD, PlayerRole.ATTACK_MIDFIELD,
                         PlayerRole.DEFENCE_MIDFIELD]:
        return midfielder_decision(obs, player_index)
    
    elif player_role == PlayerRole.CENTRAL_FORWARD:
        return forward_decision(obs, player_index)
    
    else:
        # 未知角色，默认使用中场逻辑
        return midfielder_decision(obs, player_index) 