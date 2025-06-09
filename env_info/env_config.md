# Observations & Actions


## Raw observations

Environment exposes following `raw` observations:

- Ball information:
    - `ball` - [x, y, z] position of the ball.
    - `ball_direction` - [x, y, z] ball movement vector.
    - `ball_rotation` - [x, y, z] rotation angles in radians.
    - `ball_owned_team` - {-1, 0, 1}, -1 = ball not owned, 0 = left team, 1 = right team.
    - `ball_owned_player` - {0..N-1} integer denoting index of the player owning the ball.
- Left team:
    - `left_team` - N-elements vector with [x, y] positions of players.
    - `left_team_direction` - N-elements vector with [x, y] movement vectors of players.
    - `left_team_tired_factor` - N-elements vector of floats in the range {0..1}. 0 means player is not tired at all.
    - `left_team_yellow_card` - N-elements vector of integers denoting number of yellow cards a given player has (0 or 1).
    - `left_team_active` - N-elements vector of booleans denoting whether a given player is playing the game (False means player got a red card).
    - `left_team_roles` - N-elements vector denoting roles of players. The meaning is:
        - `0` = e_PlayerRole_GK - goalkeeper,
        - `1` = e_PlayerRole_CB - centre back,
        - `2` = e_PlayerRole_LB - left back,
        - `3` = e_PlayerRole_RB - right back,
        - `4` = e_PlayerRole_DM - defence midfield,
        - `5` = e_PlayerRole_CM - central midfield,
        - `6` = e_PlayerRole_LM - left midfield,
        - `7` = e_PlayerRole_RM - right midfield,
        - `8` = e_PlayerRole_AM - attack midfield,
        - `9` = e_PlayerRole_CF - central front,
- Right team:
    - `right_team` - same as for left team.
    - `right_team_direction` - same as for left team.
    - `right_team_tired_factor` - same as for left team.
    - `right_team_yellow_card` - same as for left team.
    - `right_team_active` - same as for left team.
    - `right_team_roles` - same as for left team.
- Controlled player information:
    - `active` - {0..N-1} integer denoting index of the controlled players.
    - `designated` - {0..N-1} integer denoting index of the designated player - the player leading the game, for example the one owning the ball. In non-multiagent mode it is always equal to `active`.
    - `sticky_actions` - 10-elements vectors of 0s or 1s denoting whether corresponding action is active:
        - `0` - `action_left`
        - `1` - `action_top_left`
        - `2` - `action_top`
        - `3` - `action_top_right`
        - `4` - `action_right`
        - `5` - `action_bottom_right`
        - `6` - `action_bottom`
        - `7` - `action_bottom_left`
        - `8` - `action_sprint`
        - `9` - `action_dribble`
- Match state:
    - `score` - pair of integers denoting number of goals for left and right teams, respectively.
    - `steps_left` - how many steps are left till the end of the match.
    - `game_mode` - current game mode, one of:
        - `0` = `e_GameMode_Normal`
        - `1` = `e_GameMode_KickOff`
        - `2` = `e_GameMode_GoalKick`
        - `3` = `e_GameMode_FreeKick`
        - `4` = `e_GameMode_Corner`
        - `5` = `e_GameMode_ThrowIn`
        - `6` = `e_GameMode_Penalty`
- Screen:
    - `frame` - three vectors of RGB pixels representing rendered
    screen. It is only exposed when rendering is enabled (`render` flag). Size
    of each vector is width by height of the rendered window, 1280 by 720 by default.

Where `N` is the number of players on the team.

*   Bottom left/right corner of the field is located at `[-1, 0.42]`
    and `[1, 0.42]`, respectively.
*   Top left/right corner of the field is located at `[-1, -0.42]`
    and `[1, -0.42]`, respectively.
*   Left/right goal is located at -1 and 1 X coordinate, respectively. They
    span between `-0.044` and `0.044` in Y coordinates.
*   Speed vectors represent a change in the position of the object within a
    single step.

In case of controlling `M` players (see multi-agent section), environment
returns a list of `M` observations, one per each controlled player. For
convenience, even if you control players on the right team, observations are
mirrored (center of your goal is at `[-1, 0]`).

## Actions

### Default action set

The default action set consists of 19 actions:

*   Idle actions

    *   `action_idle` = 0, a no-op action, sticky actions are not affected (player maintains his directional movement etc.).

*   Movement actions

    *   `action_left` = 1, run to the left, sticky action.
    *   `action_top_left` = 2, run to the top-left, sticky action.
    *   `action_top` = 3, run to the top, sticky action.
    *   `action_top_right` = 4, run to the top-right, sticky action.
    *   `action_right` = 5, run to the right, sticky action.
    *   `action_bottom_right` = 6, run to the bottom-right, sticky action.
    *   `action_bottom` = 7, run to the bottom, sticky action.
    *   `action_bottom_left` = 8, run to the bottom-left, sticky action.

*   Passing / Shooting

    *   `action_long_pass` = 9, perform a long pass to the player on your team. Player to pass the ball to is auto-determined based on the movement direction.
    *   `action_high_pass` = 10, perform a high pass, similar to `action_long_pass`.
    *   `action_short_pass` = 11, perform a short pass, similar to `action_long_pass`.
    *   `action_shot` = 12, perform a shot, always in the direction of the opponent's goal.

*   Other actions

    *   `action_sprint` = 13, start sprinting, sticky action. Player moves faster, but has worse ball handling.
    *   `action_release_direction` = 14, reset current movement direction.
    *   `action_release_sprint` = 15, stop sprinting.
    *   `action_sliding` = 16, perform a slide (effective when not having a ball).
    *   `action_dribble` = 17, start dribbling (effective when having a ball), sticky action. Player moves slower, but it is harder to take over the ball from him.
    *   `action_release_dribble` = 18, stop dribbling.