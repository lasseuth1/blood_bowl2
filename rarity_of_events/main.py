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
    non_spatial_space = (1,7)
    action_space = len(env.actions)
    ac_agent = CNNPolicy(spatial_obs_space[0], action_space)

    # Parameters
    num_steps = 20
    rnd = np.random.RandomState(0)
    learning_rate = 0.01
    epsilon = 1e-5
    alpha = 0.99
    gamma = 0.99

    optimizer = optim.RMSprop(ac_agent.parameters(), learning_rate, eps=epsilon, alpha=alpha)
    number_of_games = 10000

    # Creating the memory to store the steps taken
    memory = Memory(num_steps, spatial_obs_space, non_spatial_space)

    obs = env.reset()
    obs = update_obs(obs)
    spatial_obs = np.expand_dims(obs[0], axis=0)
    non_spatial_obs = np.expand_dims(obs[1], axis=0)

    memory.spatial_obs[0].copy_(torch.from_numpy(spatial_obs).float())
    memory.non_spatial_obs[0].copy_(torch.from_numpy(non_spatial_obs).float())

    for game in range(number_of_games):

        legal_actions = 0
        non_legal_actions = 0
        reward_this_episode = 0
        number_of_steps = 0
        done = False

        while not done:

            for step in range(num_steps):
                if done:
                    env.reset()
                    break

                available_actions = env.available_action_types()
                value, action = ac_agent.act(Variable(memory.spatial_obs[step]), Variable(memory.non_spatial_obs[step]))
                # chosen_action = action[0]
                chosen_action = action.data.squeeze(0).numpy()
                available_positions = env.available_positions(chosen_action[0])

                if chosen_action in available_actions:
                    legal_actions += 1
                    pos = rnd.choice(available_positions) if len(available_positions) > 0 else None
                    action_object = {
                        'action-type': chosen_action[0],
                        'x': pos.x if pos is not None else None,
                        'y': pos.y if pos is not None else None
                    }

                    next_obs, reward, done, info = env.step(action_object)
                    number_of_steps += 1
                    env.render()
                    reward = 1.0
                    reward_this_episode += 1

                    obs = next_obs
                    obs = update_obs(obs)
                    spatial_obs = np.expand_dims(obs[0], axis=0)
                    non_spatial_obs = np.expand_dims(obs[1], axis=0)

                else:
                    non_legal_actions += 1
                    reward = 0

                # insert the step taken into memory
                memory.insert(step, torch.from_numpy(spatial_obs).float(), torch.from_numpy(non_spatial_obs).float(),
                              action.data.squeeze(1), value.data.squeeze(1), torch.tensor(reward))

            next_value = ac_agent(Variable(memory.spatial_obs[-1]), Variable(memory.non_spatial_obs[-1]))[0].data
            memory.compute_returns(next_value, False, gamma)

            action_probs, values = ac_agent.evaluate_actions(
                Variable(memory.spatial_obs[:-1].view(-1, *spatial_obs_space)),
                Variable(memory.non_spatial_obs[:-1].view(-1, 7)))

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
            print("game: ", game, "Steps taken: ", number_of_steps, "reward: ", reward_this_episode)
            print("ACTION: ", chosen_action)

        print("game: ", game, "Steps taken: ", number_of_steps, "reward: ", reward_this_episode)

    print("game: ", game, "Steps taken: ", number_of_steps, "reward: ", reward_this_episode)


def update_obs(obs):
    """
    Takes the observation returned by the environment and transforms it to an numpy array that contains all of
    the feature layers.
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

    # 'is own turn', 'own score', 'opponent score', 'half', 'round', 'own rerolls left', 'opponent rerolls left, 'procedure'
    non_spatial_info = np.stack((obs['state']['is own turn'],
                                 obs['state']['own score'],
                                 obs['state']['opp score'],
                                 obs['state']['half'],
                                 obs['state']['round'],
                                 obs['state']['own rerolls left'],
                                 obs['state']['opp rerolls left']))

    currentObs = (feature_layers, non_spatial_info)

    return currentObs

if __name__ == "__main__":
    main()
