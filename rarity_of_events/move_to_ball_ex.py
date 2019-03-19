import gym
import numpy as np
import torch
import torchvision

import matplotlib.pyplot as plt

import ffai.ai
from ffai.ai.env import FFAIEnv


def choose_move_action(available_actions):
    # iterate actions and check for MOVE
    chosen_action = rnd.choice(available_actions)
    for action in available_actions:
        if action == 22 or action == 27:  # MOVE(22) or START_MOVE(27)
            chosen_action = action
            return chosen_action
    return chosen_action


def move_towards_ball_or_endzone(availablePositions):
    player = env.game.state.active_player
    player_position = player.position if player is not None else None
    ball_position = env.game.get_ball_position()
    chosen_position = availablePositions[0] if len(availablePositions) > 0 else None
    bestDistance = 100

    if player is None:
        return rnd.choice(available_positions) if len(available_positions) > 0 else None

    if game.has_ball(player):
        point = endzone()

        if point == 1:
            for pos in availablePositions:
                if pos.x < chosen_position.x:
                    chosen_position = pos
            return chosen_position
        else:
            for pos in availablePositions:
                if pos.x > chosen_position.x:
                    chosen_position = pos
            return chosen_position

    else:
        player_with_ball = game.get_player_at(ball_position) if ball_position is not None else None
        if player_with_ball is not None:
            if team_has_ball(game, player_with_ball):
                point = endzone()
                if point == 1:
                    for pos in availablePositions:
                        if pos.x < chosen_position.x:
                            chosen_position = pos
                    return chosen_position
                else:
                    for pos in availablePositions:
                        if pos.x > chosen_position.x:
                            chosen_position = pos
                    return chosen_position
            else:
                for pos in availablePositions:
                    dist = pos.distance(ball_position) if ball_position is not None else None
                    if dist is not None:
                        if dist < bestDistance:
                            chosen_position = pos
                            bestDistance = dist
                    else:
                        return chosen_position
        else:
            for pos in availablePositions:
                dist = pos.distance(ball_position) if ball_position is not None else None
                if dist is not None:
                    if dist < bestDistance:
                        chosen_position = pos
                        bestDistance = dist
                else:
                    return chosen_position

    return chosen_position


def team_has_ball(self, player_with_ball):
    for player in self.state.current_team.players:
        if player.player_id == player_with_ball.player_id:
            return True

def endzone():
    layers = obs.get('board')  # returns another dict
    specific_layer = layers.get('opp touchdown')  # get specific layer {nd-array}
    shape = specific_layer.shape  # Shape of layer (y/x length) (height/length)
    shape_x = int(shape[1] / shape[1])  # Get end zone x-value
    shape_y = int(shape[0] / 2)  # Get the middle of y-value
    point = specific_layer[shape_y, shape_x]  # Get Point in ND Array [y,x] either 1 or 0 if present
    return point


if __name__ == "__main__":

    # Create environment
    # env = gym.make("FFAI-v1")

    # Smaller variants
    # env = gym.make("FFAI-7-v1")
    # env = gym.make("FFAI-5-v1")
    env = gym.make("FFAI-3-v1")

    # Get observations space (layer, height, width)
    obs_space = env.observation_space

    # Get action space
    act_space = env.action_space

    # Set seed for reproducibility
    seed = 1
    env.seed(seed)

    # Create random state for action selection
    rnd = np.random.RandomState(seed)

    # Play n games
    steps = 0
    total_rewards = 0
    goals = 0
    for i in range(100):
        rewards_episode = 0

        # Reset environment
        obs = env.reset()
        done = False

        # Take actions as long as game is not done
        while not done:
            # List of all Actions and Available actions
            actions = env.actions
            avail_actions = env.available_action_types()

            action_type = choose_move_action(avail_actions)   # Get move action if available
            # Looks up the table list of actions from env and returns the definition
            action_number_name = actions[action_type] if action_type is not None else None

            available_positions = env.available_positions(action_type)  # Get positions available
            pos = move_towards_ball_or_endzone(available_positions)

            ball_pos = env.game.get_ball_position()
            game = env.get_game()
            active_player = env.game.state.active_player
            active_player_id = active_player.player_id if active_player is not None else None

            # Create action object
            action = {
                'action-type': action_type,
                'x': pos.x if pos is not None else None,
                'y': pos.y if pos is not None else None
            }

            # Gym step function
            obs, reward, done, info = env.step(action)
            steps += 1
            rewards_episode += reward
            total_rewards += reward

            # Render
            env.render(feature_layers=False)
        print("Reward summed: " + str(rewards_episode))
        print("Game " + str(i))
        print("Home Team Score " + str(game.state.home_team.state.score))

        print("Away Team Score " + str(game.state.away_team.state.score))
        print("-------------")
        goals += game.state.home_team.state.score
        goals += game.state.away_team.state.score

    print("Steps: ", steps)
    print("Goals: ", goals)
    print("Total rewards: ", total_rewards)
