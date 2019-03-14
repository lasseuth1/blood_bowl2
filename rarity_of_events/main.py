import gym
import ffai.ai
import torch
import torch.nn as nn
import numpy as np
import rarity_of_events
from torch.autograd import Variable
from rarity_of_events.model import CNNPolicy
from rarity_of_events.memory import Memory
import torch.optim as optim
import matplotlib.pyplot as plt
from rarity_of_events.memory import Memory
import torch.nn.functional as F


def main():

    env = gym.make("FFAI-3-v1")
    spatial_obs_space = env.observation_space.spaces['board'].shape
    # spatial_obs_space = (26, 7, 14)
    # non_spatial_space = (1, 7)
    non_spatial_space = (1, 49)
    action_space = len(env.actions)

    # If we want to make a new model
    # ac_agent = CNNPolicy(spatial_obs_space[0], action_space)

    # If we want to load a saved model
    ac_agent = torch.load("5000_trained_model.pt")

    # Parameters
    num_steps = 20
    rnd = np.random.RandomState(0)
    learning_rate = 0.01
    gamma = 0.99

    optimizer = optim.RMSprop(ac_agent.parameters(), learning_rate)
    # optimizer = optim.Adam(ac_agent.parameters(), learning_rate)
    number_of_games = 1000

    # Creating the memory to store the steps taken
    memory = Memory(num_steps, spatial_obs_space, non_spatial_space)

    obs = env.reset()
    spatial_obs, non_spatial_obs = update_obs(obs)

    memory.spatial_obs[0].copy_(torch.from_numpy(spatial_obs).float())
    memory.non_spatial_obs[0].copy_(torch.from_numpy(non_spatial_obs).float())

    total_non_legal_actions = 0
    total_legal_actions = 0

    for game in range(number_of_games):

        legal_actions = 0
        non_legal_actions = 0
        done = False
        non_legal_actions_collected = [0] * 38

        while not done:

            for step in range(num_steps):

                available_actions = env.available_action_types()
                value, action = ac_agent.act(Variable(memory.spatial_obs[step]), Variable(memory.non_spatial_obs[step]))
                # chosen_action = action[0]
                chosen_action = action.data.squeeze(0).numpy()

                if chosen_action in available_actions:

                    available_positions = env.available_positions(chosen_action[0])
                    pos = rnd.choice(available_positions) if len(available_positions) > 0 else None

                    action_object = {
                        'action-type': chosen_action[0],
                        'x': pos.x if pos is not None else None,
                        'y': pos.y if pos is not None else None
                    }

                    obs, reward, done, info = env.step(action_object)
                    env.render()

                    # Update the observations returned by the environment
                    spatial_obs, non_spatial_obs = update_obs(obs)

                    reward = 1.0
                    legal_actions += 1
                    total_legal_actions += 1

                else:
                    non_legal_actions += 1
                    total_non_legal_actions += 1
                    reward = 0
                    non_legal_actions_collected[chosen_action[0]] += 1

                # random_action = rnd.choice(available_actions) if len(available_actions) > 0 else None

                # insert the step taken into memory
                memory.insert(step, torch.from_numpy(spatial_obs).float(), torch.from_numpy(non_spatial_obs).float(),
                              action.data.squeeze(1), value.data.squeeze(1), torch.tensor(reward))

                if done:
                    env.reset()
                    break

            next_value = ac_agent(Variable(memory.spatial_obs[-1]), Variable(memory.non_spatial_obs[-1]))[0].data
            memory.compute_returns(next_value, gamma)

            action_probs, values = ac_agent.evaluate_actions(
                Variable(memory.spatial_obs[:-1].view(-1, *spatial_obs_space)),
                Variable(memory.non_spatial_obs[:-1].view(-1, 49)))

            action_log_probs = F.log_softmax(action_probs)
            # action_log_probs = action_probs.log()

            actions = Variable(torch.LongTensor(memory.actions.view(-1, 1)))
            chosen_action_log_probs = action_log_probs.gather(1, actions)

            advantages = Variable(memory.returns[:-1]) - values
            entropy = (action_probs * action_log_probs).sum(1).mean()
            value_loss = advantages.pow(2).mean()
            action_loss = -(Variable(advantages.data) * chosen_action_log_probs).mean()

            optimizer.zero_grad()
            total_loss = (value_loss * 0.5 + action_loss - entropy * 0.01)
            total_loss.backward()
            nn.utils.clip_grad_norm(ac_agent.parameters(), 0.5)
            optimizer.step()

            memory.non_spatial_obs[0].copy_(memory.non_spatial_obs[-1])
            memory.spatial_obs[0].copy_(memory.spatial_obs[-1])

        print("Game: ", game, "Legal actions: ", legal_actions, "Non legal actions: ", non_legal_actions)
        average = total_non_legal_actions/total_legal_actions
        # print("Running average: ", "%.2f" % average)
        print("Ratio: ", "%.2f" % average)

    print("Total non legal: ", total_non_legal_actions)
    # torch.save(ac_agent, "10000_trained_model.pt")


def update_obs(obs):
    """
    Takes the observation returned by the environment and transforms it to an numpy array that contains all of
    the feature layers and non-spatial info
    """
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
                               obs['board']['pass']
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

    current_obs = (feature_layers, non_spatial_info)

    spatial_obs = np.expand_dims(current_obs[0], axis=0)
    non_spatial_obs = np.expand_dims(current_obs[1], axis=0)

    return spatial_obs, non_spatial_obs


if __name__ == "__main__":
    main()
