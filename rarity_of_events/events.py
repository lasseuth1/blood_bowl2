import numpy as np


def get_events(vars, last_vars):

    num_events = 6
    if len(last_vars) == 0:
        last_vars = [0] * num_events  # Used initially

    events = np.zeros(num_events)

    # Change in score
    if vars[0] > last_vars[0]:
        events[0] = 1

    # Change in own number of players
    if vars[1] > last_vars[1]:
        events[1] = 1

    # Change in standing players
    if vars[2] > last_vars[2]:
        events[2] = 1

    # Change in if team has acquired ball or still has ball
    if vars[3] > last_vars[3]:
        events[3] = 1

    # Change in number of re-rolls
    if vars[4] > last_vars[4]:
        events[4] = 1

    # Change in player layer (MOVEMENT)
    own_players_curr = vars[5]
    own_players_last = last_vars[5]
    if not np.array_equal(own_players_curr, own_players_last):
        events[5] = 1

    return events


def get_bb_vars(obs):
    ball_layer = obs['board']['balls']
    own_player_layer = obs['board']['own players']

    item_index = np.where(ball_layer == 1)
    team_has_ball = own_player_layer[item_index]  # Gets index value of the index where the ball is

    vars = [obs['state']['own score'],  # 0. Change in score
            np.count_nonzero(obs['board']['own players'] == 1),  # 1. Own number of players on pitch
            np.count_nonzero(obs['board']['standing players'] == 1),  # 2. Standing players
            team_has_ball,  # 3. Team has ball
            obs['state']['own reroll available'],  # 4. Number of re-rolls
            obs['board']['own players']  # 5. Own-player layer (movement-event)
            ]

    return vars
