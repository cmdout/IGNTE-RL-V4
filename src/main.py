"""
项目主入口，负责初始化环境和运行主循环
"""

import gfootball.env as football_env
from src.gfootball_agent.agent import agent
from src.gfootball_agent.config import Action, PlayerRole
import time

def create_environment(args):
    """创建Google Research Football环境"""
    env = football_env.create_environment(
        env_name='11_vs_11_stochastic',
        representation='raw',
        rewards='scoring',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=args.render,
        write_video=args.write_video,
        dump_frequency=0,
        logdir=args.logdir,
        extra_players=None,
        number_of_left_players_agent_controls=11,
        number_of_right_players_agent_controls=0
    )
    return env


def get_action_name(action):
    """获取动作名称"""
    for name, value in vars(Action).items():
        if value == action:
            return name
    return "UNKNOWN"


def get_role_name(role):
    """获取角色名称"""
    for name, value in vars(PlayerRole).items():
        if value == role:
            return name
    return "UNKNOWN"


def run_episode(env, max_steps=3000):
    """
    运行一个完整的比赛回合
    
    参数:
        env: 足球环境
        max_steps: 最大步数
    
    返回:
        episode_reward: 回合总奖励
        episode_length: 回合长度
    """
    obs = env.reset()
    agent.reset()
    
    episode_reward = 0
    episode_length = 0
    
    print("开始新的比赛回合...")
    
    for step in range(max_steps):
        # 获取所有球员的动作
        actions = agent.get_actions(obs)
        # for player_index, action in enumerate(actions):
        #     role = obs[0]['left_team_roles'][player_index]
        #     role_name = get_role_name(role)
        #     action_name = get_action_name(action)
        #     print(f"  球员 {player_index}: 角色={role_name}, 动作={action_name}")
        
        # 执行动作
        obs, rewards, done, info = env.step(actions)
        
        # 计算奖励
        # time.sleep(0.1)
        total_reward = sum(rewards) / 11
        episode_reward += total_reward
        episode_length += 1
        
        # 打印关键信息
        if step % 100 == 0 or total_reward != 0:
            print(f"步数: {step}, 奖励: {total_reward:.3f}, 比分: {obs[0]['score']}")
            # 打印每个球员的信息
            # for player_index, action in enumerate(actions):
            #     role = obs['left_team_roles'][player_index]
            #     role_name = get_role_name(role)
            #     action_name = get_action_name(action)
            #     print(f"  球员 {player_index}: 角色={role_name}, 动作={action_name}")
        
        # 检查比赛是否结束
        if done:
            print(f"比赛结束! 总步数: {episode_length}, 总奖励: {episode_reward:.3f}")
            break
    
    return episode_reward, episode_length


def main(args):
    """主函数"""
    print("初始化Google Research Football环境...")
    
    # 创建环境
    env = create_environment(args)
    print("环境创建成功!")
    
    # 运行比赛
    for episode in range(args.num_episodes):
        print(f"\n=== 第 {episode + 1} 局比赛 ===")
        
        episode_reward, episode_length = run_episode(env, args.max_steps)
        
        print(f"第 {episode + 1} 局结束:")
        print(f"  总奖励: {episode_reward:.3f}")
        print(f"  总步数: {episode_length}")
        print("-" * 50)
    
    print("所有比赛结束!")
        


if __name__ == "__main__":
    main() 