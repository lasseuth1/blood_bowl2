import torch
from multiprocessing import Process, Pipe
import numpy as np
import ffai.ai
from ffai.ai.env import FFAIEnv
import math
import uuid
import tkinter as tk
from ffai.core.game import *
from ffai.core.load import *
from ffai.ai.bots import RandomBot
from ffai.ai.layers import *
import matplotlib.pyplot as plt
import rarity_of_events.arguments as args
import rarity_of_events.utils as utils


def worker(remote, parent_remote, env):
    global player_id
    parent_remote.close()

    episode_cnt = 0.0
    step_cnt = 0
    round_count = 0

    episode_touchdown_cnt = 0
    episode_pass_cnt = 0
    # episode_casualty_KO_cnt = 0
    # episode_knockdown_count = 0
    # episode_get_ball_count = 0
    # episode_caused_turnover = 0
    # episode_failing_dodge = 0
    # episode_push_crowd = 0


    episode_step_count = 0

    total_win_cnt = 0
    total_touchdown_cnt = 0
    total_pass_cnt = 0
    # total_casualty_KO_cnt = 0
    # total_knockeddown_cnt = 0
    # total_get_ball_count = 0
    # total_caused_turnover = 0
    # total_failing_dodge = 0
    # total_push_crowd = 0

    vars = []
    last_turn_vars = []
    round_count = 0
    half_count = 0
    first_step_round = True
    start_half = True
    info = None

    def get_variables(obs):

        opp_endzone_x = 1
        variables = [info['touchdowns'] if info is not None else 0,  # 0. own score
                     info['opp_touchdowns'] if info is not None else 0,  # 1. Opponent score
                     check_team_has_ball(obs),  # 2. Team has ball
                     env.game.get_ball_position(),  # 3. Ball position
                     opp_endzone_x,  # 4. X-coordinate for opponent endzone
                     len(env.game.state.reports) if env.game.state.reports is not None else 0, # 5. report size
                     check_opponent_has_ball(obs)  # 6. checks if the opponent has ball
                     ]

        return variables

    def get_events(vars, last_vars, outcomes, outcomes_turn, done, first_step, start_half):
        """
        0: Winning game
        1: Touchdowns
        2: Casualty inflicted or knocking out an opponent
        3: Knocking down an opponent player
        4: Pushing opponents out of bounds (crowd surf)
        5: Causing the opponent to loose the ball
        6: Opponent failing a dodge
        7: Successful pass
        8: Successful handoff
        9: Closer to endzone per step
        10: Getting the ball

        """
        reward = 0
        events = np.zeros(args.num_events)

        if np.count_nonzero(last_vars) == 0:
            return events

        # 0: WINNING GAME
        if done:
            if last_vars[0] > last_vars[1]:
                events[0] = 1
                reward += 5

        # 1 TOUCHDOWNS
        if vars[0] > last_vars[0]:
            events[1] = 1
            reward += 4

        # 2: CAUSE CASUALTY OR KNOCK-OUT OPPONENT
        for outcome in reversed(outcomes):
            if outcome.outcome_type.value in [73, 43]:
                opp_players = env.game.state.away_team.players
                if outcome.player in opp_players and outcome.opp_player is not None:
                    events[2] = 1
                    reward += 3

        # 3: KNOCK OPPONENT DOWN
        for outcome in reversed(outcomes):
            if outcome.outcome_type.value == 39:
                opp_players = env.game.state.away_team.players
                if outcome.player in opp_players and outcome.opp_player is not None:
                    events[3] = 1
                    reward += 2

        # 4: PUSH INTO CROWD
        for outcome in reversed(outcomes):
            if outcome.outcome_type.value == 78:
                opp_players = env.game.state.away_team.players
                if outcome.player in opp_players:
                    events[4] = 1
                    reward += 3

        # 5: CAUSING THE OPPONENT TO LOOSE BALL
        if not start_half and not done:
            if not vars[6] and last_vars[6]:
                    events[5] = 1
                    reward += 2

        # 6: OPPONENT FAILING A DODGE
        if first_step:
            for outcome in reversed(outcomes_turn):
                if outcome.outcome_type.value == 49:
                    opp_players = env.game.state.away_team.players
                    if outcome.player in opp_players:
                        events[6] = 1
                        reward += 1

        # 7: SUCCESSFUL PASS
        # 8: HANDOFF
        outcome_idx = len(outcomes)-1
        for outcome in reversed(outcomes):
            if outcome.outcome_type.value == 108:
                own_players = env.game.state.home_team.players
                if outcome.player in own_players:
                    if outcomes[outcome_idx - 1].outcome_type.value == 80:  # ACCURATE PASS
                        events[7] = 1
                        reward += 3
                    if outcomes[outcome_idx - 1].outcome_type.value == 57:  # HANDOFF
                        events[8] = 1
                        reward += 2
            outcome_idx -= 1

        # 9: CLOSER TO ENDZONE STEP
        if vars[2] and last_vars[2]:
            dist_endzone_last = last_vars[3].x - vars[4]
            dist_endzone_now = vars[3].x - vars[4]
            if dist_endzone_now < dist_endzone_last:
                events[9] = 1
                reward += 1

        # 10: GETTING THE BALL
        if vars[2] and not last_vars[2]:
            if not outcomes[0].outcome_type.value == 103 if len(outcomes) > 0 else True:  # CHECK IF NOT TOUCHBACK
                events[10] = 1
                reward += 1

        return events, reward

    def check_team_has_ball(obs):
        ball_layer = obs['board']['balls']
        own_player_layer = obs['board']['own players']
        is_carried = env.game.state.pitch.balls[0].is_carried if len(env.game.state.pitch.balls) > 0 else False

        item_index = np.where(ball_layer == 1)
        if own_player_layer[item_index] == 1 and is_carried:
            team_has_ball = True
        else:
            team_has_ball = False
        return team_has_ball

    def check_opponent_has_ball(obs):
        ball_layer = obs['board']['balls']
        opponent_player_layer = obs['board']['opp players']
        is_carried = env.game.state.pitch.balls[0].is_carried if len(env.game.state.pitch.balls) > 0 else False

        item_index = np.where(ball_layer == 1)
        if opponent_player_layer[item_index] == 1 and is_carried:
            team_has_ball = True
        else:
            team_has_ball = False
        return team_has_ball

    while True:
        command, data = remote.recv()

        if command == 'step':
            action = data
            if len(vars) == 0:
                vars = get_variables(env.last_obs)
                last_turn_vars = vars
            last_vars = vars

            reports_one_step = len(env.game.state.reports) if env.game.state.reports is not None else 0

            obs, reward, done, info = env.step(action)

            if info is not None:
                if round_count < info['round']:
                    last_turn_vars = vars
                    first_step_round = True
                round_count = info['round']
                if half_count < info['half']:
                    start_half = True
                half_count = info['half']

            outcomes = env.game.state.reports[reports_one_step:]
            outcomes_turn = env.game.state.reports[last_turn_vars[5]:]

            vars = get_variables(obs)
            events, event_reward = get_events(vars, last_vars, outcomes, outcomes_turn, done, first_step_round, start_half)

            reward = event_reward

            first_step_round = False
            start_half = False
            step_cnt += 1
            episode_step_count += 1

            if done or episode_step_count > 500:
                obs = env.reset()

                vars = get_variables(obs)
                last_turn_vars = vars
                round_count = 0
                episode_step_count = 0

            remote.send((obs, reward, done, info, events))

        elif command == 'reset':
            obs = env.reset()
            remote.send(obs)

        elif command == 'actions':

            mask = torch.zeros(908, dtype=torch.uint8)
            available_actions = env.available_action_types()
            # actions_dictionary = utils.create_action_dictionary_3v3_new_approach()
            actions_dictionary = utils.create_action_dictionary_5v5_pruned()

            for avail_action in available_actions:
                for action_dict in actions_dictionary:

                    if avail_action == action_dict['action_index']:

                        positions = env.available_positions(avail_action)
                        if not positions:
                            to_mask = action_dict['positions_start']
                            mask[to_mask] = 1
                            break
                        else:
                            for pos in positions:
                                try:
                                    if action_dict['position_type'] == 'adjacent':

                                        active_player = env.game.state.active_player
                                        active_x = active_player.position.x
                                        active_y = active_player.position.y

                                        if pos.x < active_x:
                                            if pos.y < active_y:
                                                mask[action_dict['positions_start'] + 0] = 1
                                            elif pos.y == active_y:
                                                mask[action_dict['positions_start'] + 3] = 1
                                            else:
                                                mask[action_dict['positions_start'] + 5] = 1
                                        elif pos.x == active_x:
                                            if pos.y < active_y:
                                                mask[action_dict['positions_start'] + 1] = 1
                                            else:
                                                mask[action_dict['positions_start'] + 6] = 1
                                        else:
                                            if pos.y < active_y:
                                                mask[action_dict['positions_start'] + 2] = 1
                                            elif pos.y == active_y:
                                                mask[action_dict['positions_start'] + 4] = 1
                                            else:
                                                mask[action_dict['positions_start'] + 7] = 1

                                    elif action_dict['position_type'] == 'full_board':
                                        x = pos.x
                                        y = pos.y
                                        # mask_pos = x + 14 * y  # 3v3 full board
                                        mask_pos = x + 18 * y  # 3v3 full board
                                        mask[action_dict['positions_start'] + mask_pos] = 1

                                    else:
                                        player_idx = 0
                                        for player in env.game.state.home_team.players:
                                            if player.position == pos:
                                                mask[action_dict['positions_start'] + player_idx] = 1
                                            player_idx += 1

                                except:
                                    a = 1
                            break

            remote.send(mask)

        elif command == 'active_players':
            active_player = env.game.state.active_player
            if active_player is not None and active_player.position is not None:
                active_x = active_player.position.x
                active_y = active_player.position.y
                position_index = active_x + 18 * active_y
            else:
                position_index = -1
            remote.send(position_index)

        elif command == 'own_players':
            own_players = []
            for player in env.game.state.home_team.players:
                if player.position is not None:
                    x = player.position.x
                    y = player.position.y
                    position_index = x + 18 * y
                    own_players.append(position_index)
                else:
                    position_index = -1
                    own_players.append(position_index)
            remote.send(own_players)

        elif command == 'render':
            env.render()


class VecEnv():
    def __init__(self, envs):
        """
        envs: list of blood bowl game environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(envs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        self.ps = [Process(target=worker, args=(work_remote, remote, env))
            for (work_remote, remote, env) in zip(self.work_remotes, self.remotes, envs)]

        for p in self.ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        cumul_rewards = None
        cumul_dones = None
        cumul_events = None

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]

        obs, rews, dones, infos, events = zip(*results)
        if cumul_rewards is None:
            cumul_rewards = np.stack(rews)
        else:
            cumul_rewards += np.stack(rews)
        if cumul_dones is None:
            cumul_dones = np.stack(dones)
        else:
            cumul_dones |= np.stack(dones)
        if cumul_events is None:
            cumul_events = events
        else:
            cumul_events = np.add(cumul_events, events)
        return np.stack(obs), cumul_rewards, cumul_dones, infos, np.stack(cumul_events)

    def actions(self):
        for remote in self.remotes:
            remote.send(('actions', None))
        results = [remote.recv() for remote in self.remotes]
        try:
            results = torch.stack(results)
        except TypeError:
            print("Error")
            print(results)
        return results

    def active_players(self):
        for remote in self.remotes:
            remote.send(('active_players', None))
        return [remote.recv() for remote in self.remotes]

    def own_players(self):
        for remote in self.remotes:
            remote.send(('own_players', None))
        return [remote.recv() for remote in self.remotes]

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def render(self):
        for remote in self.remotes:
            remote.send(('render', None))
        return

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)
