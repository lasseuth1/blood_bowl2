import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rarity_of_events.arguments as args


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, x, y):
        raise NotImplementedError

    def act(self, spatial_inputs, non_spatial_input, actions_mask):

        # The model returns a value, policy for actions, and positions
        value, policy = self(spatial_inputs, non_spatial_input)

        policy[~actions_mask] = float('-inf')  # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
        policy = F.softmax(policy, dim=1)

        actions = policy.multinomial(1)

        return value, actions

    def evaluate_actions(self, spatial_inputs, non_spatial_input, actions, actions_mask):
        # gets values, actions, and positions
        value, policy = self(spatial_inputs, non_spatial_input)

        # actions_mask = actions_mask.view(-1, 1, 242).squeeze()
        # actions_mask = actions_mask.view(-1, 1, 1078).squeeze()
        # actions_mask = actions_mask.view(-1, 1, 492).squeeze()
        actions_mask = actions_mask.view(-1, 1, 908).squeeze()

        policy[~actions_mask] = float('-inf')

        log_probs = F.log_softmax(policy, dim=1)
        probs = F.softmax(policy, dim=1)
        action_log_probs = log_probs.gather(1, actions)

        log_probs = torch.where(log_probs[None, :] == float('-inf'), torch.tensor(0.), log_probs)

        dist_entropy = -(log_probs * probs).sum(-1).mean()

        return action_log_probs, value, dist_entropy


class PrunedHybrid(FFPolicy):
    def __init__(self, num_inputs, action_space_shape):
        super(PrunedHybrid, self).__init__()

        # num_inputs is 26 (number of feature layers)
        self.conv1 = nn.Conv2d(num_inputs, out_channels=16, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Linear layers
        self.linear1 = nn.Linear(50, 25)

        # # The outputs for 3v3
        # self.critic = nn.Linear(1945, 1)
        # self.actor = nn.Linear(1945, 492)

        # The outputs for 5v5
        self.critic = nn.Linear(4633, 1)
        self.actor = nn.Linear(4633, 908)

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

    def forward(self, spatial_input, non_spatial_input):
        """
        The forward functions defines how the data flows through the graph (layers)
        """
        # Spatial input through two convolutional layers
        x = self.conv1(spatial_input)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        # Non-spatial input through one linear layer (fully-connected)
        y = self.linear1(non_spatial_input)
        y = F.relu(y)

        # Concatenate the outputs
        flatten_x = x.flatten(start_dim=1)
        flatten_y = y.flatten(start_dim=1)
        concatenated = torch.cat((flatten_x, flatten_y), dim=1)

        value = self.critic(concatenated)
        policy = self.actor(concatenated)

        # return value, policy
        return value, policy

    def get_action_probs(self, spatial_input, non_spatial_input):
        x, y = self(spatial_input, non_spatial_input)
        action_probs = F.softmax(y)
        return action_probs
