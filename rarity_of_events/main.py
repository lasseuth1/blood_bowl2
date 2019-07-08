import gym
import os
import ffai.ai
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from rarity_of_events.model import CNNPolicy
from rarity_of_events.pruned_hybrid import PrunedHybrid
import torch.optim as optim
from rarity_of_events.memory import Memory
import torch.nn.functional as F
from rarity_of_events.event_buffer import EventBuffer
from rarity_of_events.vec_env import VecEnv
import rarity_of_events.arguments as args
from ffai.ai.renderer import Renderer
import rarity_of_events.utils as utils
import time
import pickle


def main():
    es = [make_env(i, args.board_size) for i in range(args.num_processes)]
    envs = VecEnv([es[i] for i in range(args.num_processes)])

    spatial_obs_space = es[0].observation_space.spaces['board'].shape
    non_spatial_space = (1, 50)
    action_space = len(es[0].actions)

    # MODELS #
    if args.resume:
        ac_agent = torch.load("models/" + args.model_name)   # Load model
    else:
        ac_agent = PrunedHybrid(spatial_obs_space[0], action_space)

    optimizer = optim.RMSprop(ac_agent.parameters(), args.learning_rate)

    # Creating the memory to store the steps taken
    memory = Memory(args.num_steps, args.num_processes, spatial_obs_space, non_spatial_space)

    obs = envs.reset()
    spatial_obs, non_spatial_obs = update_obs(obs)

    memory.spatial_obs[0].copy_(torch.from_numpy(spatial_obs).float())
    memory.non_spatial_obs[0].copy_(torch.from_numpy(non_spatial_obs).float())


    if args.resume & args.log:
        log_file = "logs/" + args.log_filename
        with open(log_file) as log:
            lines = log.readlines()[-1]
            resume_episodes = float(lines.split(", ")[1])
            resume_updates = float(lines.split(", ")[0])
            resume_steps = float(lines.split(", ")[2])
    else:
        resume_episodes = 0
        resume_updates = 0
        resume_steps = 0

    renderer = Renderer()

    for update in range(args.num_updates):

        for step in range(args.num_steps):

            available_actions = envs.actions()
            active_players = envs.active_players()
            own_players = envs.own_players()

            values, actions_policy = ac_agent.act(
                Variable(memory.spatial_obs[step]),
                Variable(memory.non_spatial_obs[step]), available_actions)

            actions, x_positions, y_positions = utils.map_actions_3v3_new_approach(actions_policy, active_players,
                                                                                   own_players)
            action_objects = []

            for action, position_x, position_y in zip(actions, x_positions, y_positions):

                action_object = {
                    'action-type': action,
                    'x': position_x,
                    'y': position_y
                    }
                action_objects.append(action_object)

            obs, reward, done, info = envs.step(action_objects)

            if args.render:
                for i in range(args.num_processes):
                    renderer.render(obs[i], i)

            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            # Update the observations returned by the environment
            spatial_obs, non_spatial_obs = update_obs(obs)

            # insert the step taken into memory
            memory.insert(step, torch.from_numpy(spatial_obs).float(), torch.from_numpy(non_spatial_obs).float(),
                          torch.tensor(actions_policy), torch.tensor(values), reward, masks, available_actions)

        next_value = ac_agent(Variable(memory.spatial_obs[-1]), Variable(memory.non_spatial_obs[-1]))[0].data

        # Compute returns
        memory.compute_returns(next_value, args.gamma)

        spatial = Variable(memory.spatial_obs[:-1])  # shape [20,  4, 26,  7, 14]
        spatial = spatial.view(-1, *spatial_obs_space)  # shape [80, 26,  7, 14]
        non_spatial = Variable(memory.non_spatial_obs[:-1])  # shape [20,  4,  1, 49]
        non_spatial = non_spatial.view(-1, 50)  # shape [80, 49]

        actions = Variable(torch.LongTensor(memory.actions.view(-1, 1)))
        actions_mask = Variable(memory.available_actions[:-1])

        # Evaluate the actions taken
        action_log_probs, values, dist_entropy = ac_agent.evaluate_actions(Variable(spatial),
                                                                           Variable(non_spatial),
                                                                           actions, actions_mask)

        values = values.view(args.num_steps, args.num_processes, 1)
        action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

        advantages = Variable(memory.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()

        # Compute loss
        action_loss = -(Variable(advantages.data) * action_log_probs).mean()

        optimizer.zero_grad()

        total_loss = (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef)
        total_loss.backward()

        nn.utils.clip_grad_norm_(ac_agent.parameters(), args.max_grad_norm)

        optimizer.step()

        memory.non_spatial_obs[0].copy_(memory.non_spatial_obs[-1])
        memory.spatial_obs[0].copy_(memory.spatial_obs[-1])

        # Logging
        if (update + 1) % args.log_interval == 0 and args.log:
            log_file_name = "logs/" + args.log_filename

            episodes, steps, rewards, wins, touchdowns = envs.log()
            updates = resume_updates + update + 1
            this_episodes = episodes.sum()
            resume_episodes += this_episodes
            steps = updates * args.num_processes * args.num_steps
            reward = rewards.sum()
            wins = wins.sum()
            touchdowns = touchdowns.sum()


            log = "Updates {}, Episodes {}, Episodes this update {}, Timesteps {}, Reward {}, Wins {}, " \
                  "Touchdowns {}" \
                .format(updates, resume_episodes, this_episodes, steps, reward, wins, touchdowns
                        )

            log_to_file = "{}, {}, {}, {}, {}, {}, \n" \
                .format(updates, resume_episodes, this_episodes, steps, reward, wins, touchdowns
                        )

            print(log)

            # Save to files
            with open(log_file_name, "a") as myfile:
                myfile.write(log_to_file)

            # Saving the agent
            torch.save(ac_agent, "models/" + args.model_name)


def update_obs(observations):
    """
    Takes the observation returned by the environment and transforms it to an numpy array that contains all of
    the feature layers and non-spatial info
    """
    spatial_obs = []
    non_spatial_obs = []

    for obs in observations:
        feature_layers = np.stack((obs['board']['own players'],
                                   obs['board']['occupied'],
                                   obs['board']['opp players'],
                                   obs['board']['own tackle zones'],
                                   obs['board']['opp tackle zones'],
                                   obs['board']['standing players'],
                                   obs['board']['used players'],
                                   obs['board']['available players'],
                                   obs['board']['available positions'],
                                   obs['board']['roll probabilities'],
                                   obs['board']['block dice'],
                                   obs['board']['active players'],
                                   obs['board']['target player'],
                                   obs['board']['movement allowence'],
                                   obs['board']['strength'],
                                   obs['board']['agility'],
                                   obs['board']['armor value'],
                                   obs['board']['movement left'],
                                   obs['board']['balls'],
                                   obs['board']['own half'],
                                   obs['board']['own touchdown'],
                                   obs['board']['opp touchdown'],
                                   obs['board']['block'],
                                   obs['board']['dodge'],
                                   obs['board']['sure hands'],
                                   obs['board']['pass'],
                                   obs['board']['catch'],
                                   obs['board']['stunned players']
                                   ))

        # Non-spatial info
        non_spatial_info = np.stack((obs['state']['half'],
                                     obs['state']['round'],
                                     obs['state']['is sweltering heat'],
                                     obs['state']['is very sunny'],
                                     obs['state']['is nice'],
                                     obs['state']['is pouring rain'],
                                     obs['state']['is blizzard'],
                                     obs['state']['is own turn'],
                                     obs['state']['is kicking first half'],
                                     obs['state']['is kicking this drive'],
                                     obs['state']['own reserves'],
                                     obs['state']['own kods'],
                                     obs['state']['own casualites'],
                                     obs['state']['opp reserves'],
                                     obs['state']['opp kods'],
                                     obs['state']['opp casualties'],
                                     obs['state']['own score'],
                                     obs['state']['own turns'],
                                     obs['state']['own starting rerolls'],
                                     obs['state']['own rerolls left'],
                                     obs['state']['own ass coaches'],
                                     obs['state']['own cheerleaders'],
                                     obs['state']['own bribes'],
                                     obs['state']['own babes'],
                                     obs['state']['own apothecary available'],
                                     obs['state']['own reroll available'],
                                     obs['state']['own fame'],
                                     obs['state']['opp score'],
                                     obs['state']['opp turns'],
                                     obs['state']['opp starting rerolls'],
                                     obs['state']['opp rerolls left'],
                                     obs['state']['opp ass coaches'],
                                     obs['state']['opp cheerleaders'],
                                     obs['state']['opp bribes'],
                                     obs['state']['opp babes'],
                                     obs['state']['opp apothecary available'],
                                     obs['state']['opp reroll available'],
                                     obs['state']['opp fame'],
                                     obs['state']['is blitz available'],
                                     obs['state']['is pass available'],
                                     obs['state']['is handoff available'],
                                     obs['state']['is foul available'],
                                     obs['state']['is blitz'],
                                     obs['state']['is quick snap'],
                                     obs['state']['is move action'],
                                     obs['state']['is block action'],
                                     obs['state']['is blitz action'],
                                     obs['state']['is pass action'],
                                     obs['state']['is handoff action'],
                                     obs['state']['is foul action']))

        # feature_layers = np.expand_dims(feature_layers, axis=0)
        non_spatial_info = np.expand_dims(non_spatial_info, axis=0)

        spatial_obs.append(feature_layers)
        non_spatial_obs.append(non_spatial_info)

    return np.stack(spatial_obs), np.stack(non_spatial_obs)


def make_env(worker_id, board_size):
    print("Initializing blood bowl environment", worker_id, "...")
    if board_size > 10:
        env_name = "FFAI-v1"
        env = gym.make(env_name)
    else:
        env_name = "FFAI-" + str(board_size) + "-v1"
        env = gym.make(env_name)
    return env


if __name__ == "__main__":
    main()
