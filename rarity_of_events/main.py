import gym
import os
import ffai.ai
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from rarity_of_events.model import CNNPolicy
import torch.optim as optim
from rarity_of_events.memory import Memory
import torch.nn.functional as F
from rarity_of_events.event_buffer import EventBuffer
from rarity_of_events.vec_env import VecEnv
import rarity_of_events.arguments as args


def main():
    # env = gym.make("FFAI-3-v1")

    best_final_rewards = -1000000.0

    es = [make_env(i, board_size=args.board_size) for i in range(args.num_processes)]
    envs = VecEnv([es[i] for i in range(args.num_processes)])

    spatial_obs_space = es[0].observation_space.spaces['board'].shape
    non_spatial_space = (1, 49)
    action_space = len(es[0].actions)

    # MODELS #
    ac_agent = CNNPolicy(spatial_obs_space[0], action_space)  # New model
    # ac_agent = torch.load("models/" + args.model_to_load)         # Load model

    optimizer = optim.RMSprop(ac_agent.parameters(), args.learning_rate)
    # optimizer = optim.Adam(ac_agent.parameters(), learning_rate)

    # Creating the memory to store the steps taken
    memory = Memory(args.num_steps, args.num_processes, spatial_obs_space, non_spatial_space)

    obs = envs.reset()
    spatial_obs, non_spatial_obs = update_obs(obs)

    memory.spatial_obs[0].copy_(torch.from_numpy(spatial_obs).float())
    memory.non_spatial_obs[0].copy_(torch.from_numpy(non_spatial_obs).float())

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])
    episode_intrinsic_rewards = torch.zeros([args.num_processes, 1])
    final_intrinsic_rewards = torch.zeros([args.num_processes, 1])
    episode_events = torch.zeros([args.num_processes, args.num_events])  # [4,10]
    final_events = torch.zeros([args.num_processes, args.num_events])

    # Create event-buffer
    event_buffer = EventBuffer(args.num_events, capacity=args.eb_capacity)

    event_episode_rewards = []

    steps = 0
    dones = 0
    number_of_touchdowns = 0
    log = True
    last_rewards = 0

    for update in range(args.num_updates):

        for step in range(args.num_steps):
            steps += 1
            available_actions = envs.actions()
            value, action, action_objects, positions = ac_agent.act(
                                                         Variable(memory.spatial_obs[step]),
                                                         Variable(memory.non_spatial_obs[step]),
                                                         available_actions,
                                                         envs)

            obs, reward, done, info, events = envs.step(action_objects)

            intrinsic_reward = []

            for e in events:
                if args.rarity_of_events:
                    intrinsic_reward.append(event_buffer.intrinsic_reward(e))
                else:
                    r = reward[len(intrinsic_reward)]
                    intrinsic_reward.append(r)

            # Format variables correctly
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            intrinsic_reward = torch.from_numpy(np.expand_dims(np.stack(intrinsic_reward), 1)).float()
            events = torch.from_numpy(np.expand_dims(np.stack(events), args.num_events)).float().squeeze(2)
            # events = torch.from_numpy(events).float()
            episode_rewards += reward
            episode_intrinsic_rewards += intrinsic_reward
            episode_events += events

            # Event stats: Displaying rewards acquired for each event in a list
            event_rewards = []
            for ei in range(0, args.num_events):
                ev = np.zeros(args.num_events)
                ev[ei] = 1
                er = event_buffer.intrinsic_reward(ev)
                event_rewards.append(er)

            event_episode_rewards.append(event_rewards)

            # If done then clean the history of observations.
            if torch.sum(reward) > 0:
                print()
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_intrinsic_rewards *= masks
            final_events *= masks
            final_rewards += (1 - masks) * episode_rewards
            final_intrinsic_rewards += (1 - masks) * episode_intrinsic_rewards
            final_events += (1 - masks) * episode_events

            for i in range(args.num_processes):
                if done[i]:
                    event_buffer.record_events(np.copy(final_events[i].numpy()))
                    dones += 1
                    log = False

            episode_rewards *= masks
            episode_intrinsic_rewards *= masks
            episode_events *= masks

            # Update the observations returned by the environment
            spatial_obs, non_spatial_obs = update_obs(obs)

            # insert the step taken into memory
            memory.insert(step, torch.from_numpy(spatial_obs).float(), torch.from_numpy(non_spatial_obs).float(),
                          action.data, positions.data, value.data, reward, masks)

        final_episode_reward = np.mean(event_episode_rewards, axis=0)
        event_episode_rewards = []

        next_value = ac_agent(Variable(memory.spatial_obs[-1]), Variable(memory.non_spatial_obs[-1]))[0].data
        memory.compute_returns(next_value, args.gamma)

        spatial = Variable(memory.spatial_obs[:-1])             # [20,  4, 26,  7, 14]
        spatial = spatial.view(-1, *spatial_obs_space)          # [80, 26,  7, 14]
        non_spatial = Variable(memory.non_spatial_obs[:-1])     # [20,  4,  1, 49]
        non_spatial = non_spatial.view(-1, 49)                  # [80, 49]

        actions = Variable(torch.LongTensor(memory.actions.view(-1, 1)))
        positions = Variable(torch.LongTensor(memory.positions.view(-1, 1)))

        # [80, 37]   [80, 1]  [80, 99]
        action_log_probs, values, position_log_probs, dist_entropy = ac_agent.evaluate_actions(Variable(spatial),
                                                                                               Variable(non_spatial),
                                                                                               actions, positions)

        values = values.view(args.num_steps, args.num_processes, 1)
        action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)
        position_log_probs = position_log_probs.view(args.num_steps, args.num_processes, 1)
        advantages = Variable(memory.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(Variable(advantages.data) * action_log_probs).mean()
        position_loss = -(Variable(advantages.data) * position_log_probs).mean()

        optimizer.zero_grad()

        total_loss = (value_loss + action_loss + position_loss - dist_entropy * 0.0001)
        total_loss.backward()

        nn.utils.clip_grad_norm(ac_agent.parameters(), 0.5)

        optimizer.step()

        memory.non_spatial_obs[0].copy_(memory.non_spatial_obs[-1])
        memory.spatial_obs[0].copy_(memory.spatial_obs[-1])

        final_intrinsic_rewards_mean = final_intrinsic_rewards.mean()
        final_rewards_mean = final_rewards.mean()

        #if (update + 1) % log_interval == 0:
        if dones % 4 == 0 and log is False:
            log_file_name = "logs/" + args.log_filename
            total_num_steps = (update + 1) * args.num_processes * args.num_steps
            # update = update + args.updates_when_stop
            # total_num_steps = args.timesteps_when_stop + total_num_steps
            log = "Updates {}, num timesteps {}, mean reward {:.5f}, number of touchdowns {}" \
                .format(update, total_num_steps, final_rewards_mean, number_of_touchdowns)
            log_to_file = "{}, {}, {:.5f}, {}\n" \
                .format(update, total_num_steps, final_rewards_mean, number_of_touchdowns)
            print(log)

            with open(log_file_name, "a") as myfile:
                myfile.write(log_to_file)

            log = True
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
