# @Time: 2024/12/27
# @Average Reward: -0.1 In 10 matches, you Win 2, Tie 5 and Lose 3, scored 6 goals, and conceded 7 goals.
# @Model: Deepseek v3
# @Description: 使用Coder-Planner-Summarizer迭代多次
import gfootball.env as football_env
import logging
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def create_football_env():
    env = football_env.create_environment(
        env_name='11_vs_11_easy_stochastic',
        stacked=False,
        representation='raw',
        rewards='scoring',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        write_video=False,
        logdir="dump",
        render=True,
        number_of_left_players_agent_controls=11,
        number_of_right_players_agent_controls=0,
        other_config_options={'action_set': 'full'}
    )
    return env
# Placeholder for Observation:
class PlayerObservationWrapper:
    def __init__(self, observation):
        self.observation = observation

        # Ball information
        self.ball_position = observation['ball'][:2]  # x, y position of the ball
        self.ball_direction = observation['ball_direction'][:2]  # x, y direction of the ball
        self.ball_owned_team = observation['ball_owned_team']
        self.ball_owned_player = observation['ball_owned_player']

        # Player information
        self.active_player = observation['active']  # Index of the controlled player
        self.player_position = observation['left_team'][self.active_player]
        self.player_direction = observation['left_team_direction'][self.active_player]
        self.player_tired_factor = observation['left_team_tired_factor'][self.active_player]
        self.player_yellow_card = observation['left_team_yellow_card'][self.active_player]
        self.player_active = observation['left_team_active'][self.active_player]
        self.player_role = observation['left_team_roles'][self.active_player]

        # Team information
        self.left_team_positions = observation['left_team']
        self.left_team_directions = observation['left_team_direction']
        self.right_team_positions = observation['right_team']
        self.right_team_directions = observation['right_team_direction']

        # Game state
        self.game_mode = observation['game_mode']
        self.score = observation['score']
        self.steps_left = observation['steps_left']
        self.sticky_actions = observation['sticky_actions']

        # Precomputed distances
        self.distance_to_ball = self.compute_distance(self.player_position, self.ball_position)
        self.distances_to_teammates = [
            self.compute_distance(self.player_position, teammate_pos)
            for i, teammate_pos in enumerate(self.left_team_positions) if i != self.active_player
        ]
        self.distances_to_opponents = [
            self.compute_distance(self.player_position, opponent_pos)
            for opponent_pos in self.right_team_positions
        ]

    @staticmethod
    def compute_distance(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def is_ball_owned_by_player(self):
        return self.ball_owned_team == 0 and self.ball_owned_player == self.active_player

    def is_ball_owned_by_team(self, team=0):
        return self.ball_owned_team == team

    def is_ball_free(self):
        return self.ball_owned_team == -1


class ObservationWrapper:
    def __init__(self, observations):
        self.player_observations = [PlayerObservationWrapper(obs) for obs in observations]


class ActionWrapper:
    def __init__(self, env):
        self.env = env

    def step(self, actions):
        return self.env.step(actions)

    def write_dump(self):
        self.env.write_dump('shutdown')

# Placeholder for Helpful Function:
def get_movement_action(current_pos, target_pos):
    direction = target_pos - current_pos
    angle = np.arctan2(direction[1], direction[0])
    
    if -np.pi/4 <= angle < np.pi/4:
        return 5  # action_right
    elif np.pi/4 <= angle < 3*np.pi/4:
        return 4  # action_top_right
    elif 3*np.pi/4 <= angle or angle < -3*np.pi/4:
        return 3  # action_top
    elif -3*np.pi/4 <= angle < -np.pi/4:
        return 2  # action_top_left
    elif -np.pi/4 <= angle < 0:
        return 1  # action_left
    elif 0 <= angle < np.pi/4:
        return 1  # action_left
    else:
        return 0  # action_idle

# Placeholder for individual player action functions:

#Player Index: 0 Position: [-1.01102936 -0.        ]
def goalkeeper_actions(player_obs):
    if player_obs.is_ball_owned_by_player():
        # Clear the ball long up the field
        return 9  # action_long_pass
    else:
        # Stay in position to block shots
        return 0  # action_idle

#Player Index: 1 Position: [0.         0.02032536]
def right_midfielder_actions(player_obs, obs_wrapper):
    if player_obs.is_ball_owned_by_player():
        # Look for a quick counter-attack
        target_pos = np.array([1.0, player_obs.left_team_positions[1][1]])
        return get_movement_action(player_obs.left_team_positions[1], target_pos)
    else:
        # Drop deep to defend
        target_pos = np.array([-0.5, player_obs.left_team_positions[1][1]])
        return get_movement_action(player_obs.left_team_positions[1], target_pos)

#Player Index: 2 Position: [ 0.         -0.02032536]
def central_forward_actions(player_obs, obs_wrapper):
    if player_obs.is_ball_owned_by_player():
        # Take a shot if in a good position
        if player_obs.left_team_positions[2][0] > 0.7 and abs(player_obs.left_team_positions[2][1]) < 0.2:
            return 12  # action_shot
        else:
            # Dribble towards the goal
            target_pos = np.array([1.0, player_obs.left_team_positions[2][1]])
            return get_movement_action(player_obs.left_team_positions[2], target_pos)
    else:
        # Stay high up the pitch for counter-attacks
        target_pos = np.array([0.5, player_obs.left_team_positions[2][1]])
        return get_movement_action(player_obs.left_team_positions[2], target_pos)

#Player Index: 3 Position: [-0.4266544  -0.19894461]
def left_back_actions(player_obs, obs_wrapper):
    if player_obs.is_ball_owned_by_player():
        # Clear the ball long up the field
        return 9  # action_long_pass
    else:
        # Drop deep to defend
        target_pos = np.array([-0.8, player_obs.left_team_positions[3][1]])
        return get_movement_action(player_obs.left_team_positions[3], target_pos)

#Player Index: 4 Position: [-0.50551468 -0.06459399]
def left_centre_back_actions(player_obs, obs_wrapper):
    if player_obs.is_ball_owned_by_player():
        # Clear the ball long up the field
        return 9  # action_long_pass
    else:
        # Drop deep to defend
        target_pos = np.array([-0.9, player_obs.left_team_positions[4][1]])
        return get_movement_action(player_obs.left_team_positions[4], target_pos)

#Player Index: 5 Position: [-0.50551468  0.06459298]
def right_centre_back_actions(player_obs, obs_wrapper):
    if player_obs.is_ball_owned_by_player():
        # Clear the ball long up the field
        return 9  # action_long_pass
    else:
        # Drop deep to defend
        target_pos = np.array([-0.9, player_obs.left_team_positions[5][1]])
        return get_movement_action(player_obs.left_team_positions[5], target_pos)

#Player Index: 6 Position: [-0.4266544   0.19894461]
def right_back_actions(player_obs, obs_wrapper):
    if player_obs.is_ball_owned_by_player():
        # Clear the ball long up the field
        return 9  # action_long_pass
    else:
        # Drop deep to defend
        target_pos = np.array([-0.8, player_obs.left_team_positions[6][1]])
        return get_movement_action(player_obs.left_team_positions[6], target_pos)

#Player Index: 7 Position: [-0.18624374 -0.10739919]
def left_central_midfielder_actions(player_obs, obs_wrapper):
    if player_obs.is_ball_owned_by_player():
        # Look for a quick counter-attack
        target_pos = np.array([1.0, player_obs.left_team_positions[7][1]])
        return get_movement_action(player_obs.left_team_positions[7], target_pos)
    else:
        # Drop deep to defend
        target_pos = np.array([-0.5, player_obs.left_team_positions[7][1]])
        return get_movement_action(player_obs.left_team_positions[7], target_pos)

#Player Index: 8 Position: [-0.27052519 -0.        ]
def central_midfielder_actions(player_obs, obs_wrapper):
    if player_obs.is_ball_owned_by_player():
        # Look for a quick counter-attack
        target_pos = np.array([1.0, player_obs.left_team_positions[8][1]])
        return get_movement_action(player_obs.left_team_positions[8], target_pos)
    else:
        # Drop deep to defend
        target_pos = np.array([-0.6, player_obs.left_team_positions[8][1]])
        return get_movement_action(player_obs.left_team_positions[8], target_pos)

#Player Index: 9 Position: [-0.18624374  0.10739919]
def right_central_midfielder_actions(player_obs, obs_wrapper):
    if player_obs.is_ball_owned_by_player():
        # Look for a quick counter-attack
        target_pos = np.array([1.0, player_obs.left_team_positions[9][1]])
        return get_movement_action(player_obs.left_team_positions[9], target_pos)
    else:
        # Drop deep to defend
        target_pos = np.array([-0.5, player_obs.left_team_positions[9][1]])
        return get_movement_action(player_obs.left_team_positions[9], target_pos)

#Player Index: 10 Position: [-0.01011029 -0.2196155 ]
def left_midfielder_actions(player_obs, obs_wrapper):
    if player_obs.is_ball_owned_by_player():
        # Look for a quick counter-attack
        target_pos = np.array([1.0, player_obs.left_team_positions[10][1]])
        return get_movement_action(player_obs.left_team_positions[10], target_pos)
    else:
        # Drop deep to defend
        target_pos = np.array([-0.4, player_obs.left_team_positions[10][1]])
        return get_movement_action(player_obs.left_team_positions[10], target_pos)
def advanced_strategy(obs_wrapper):
    actions = []
    for player_obs in obs_wrapper.player_observations:
        index = player_obs.active_player

        if index == 0:  # Goalkeeper (GK)
            action = goalkeeper_actions(player_obs)

        elif index == 1:  # Center Backs (CB1)
            action = right_midfielder_actions(player_obs, obs_wrapper)

        elif index == 2:  # Center Backs (CB2)
            action = central_forward_actions(player_obs, obs_wrapper)

        elif index == 3:  # Left Back (LB)
            action = left_back_actions(player_obs, obs_wrapper)

        elif index == 4:  # Right Back (RB)
            action = left_centre_back_actions(player_obs, obs_wrapper)

        elif index == 5:  # Defensive Midfielder (DM)
            action = right_centre_back_actions(player_obs, obs_wrapper)

        elif index == 6:  # Central Midfielder (CM)
            action = right_back_actions(player_obs, obs_wrapper)

        elif index == 7:  # Left Midfielder (LM)
            action = left_central_midfielder_actions(player_obs, obs_wrapper)

        elif index == 8:  # Right Midfielder (RM)
            action = central_midfielder_actions(player_obs, obs_wrapper)

        elif index == 9:  # Attacking Midfielder (AM)
            action = right_central_midfielder_actions(player_obs, obs_wrapper)

        elif index == 10:  # Central Forward (CF)
            action = left_midfielder_actions(player_obs, obs_wrapper)

        else:
            action = 0  # Default action (Idle)

        actions.append(action)

    return actions

def main():
    env = create_football_env()
    action_wrapper = ActionWrapper(env)
    observations = env.reset()
    obs_wrapper = ObservationWrapper(observations)
    left_reward = right_reward = 0
    while True:
        actions = advanced_strategy(obs_wrapper)
        observations, rewards, dones, infos = action_wrapper.step(actions)
        obs_wrapper = ObservationWrapper(observations)
        if rewards[0] == 1:
            left_reward += 1
        elif rewards[0] == -1:
            right_reward += 1
        if dones:
            break
    return left_reward, right_reward

if __name__ == '__main__':
    # 抑制 gfootball 库的 INFO 级别日志
    football_logger = logging.getLogger('gfootball')
    football_logger.setLevel(logging.WARNING)
    # 使用 ProcessPoolExecutor 并行运行 main 函数
    # with ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(main) for _ in range(10)]
    #     for future in futures:
    #         left_reward, right_reward = future.result()
    #         print("left_reward:{}, right_reward:{}".format(left_reward, right_reward))
    main()