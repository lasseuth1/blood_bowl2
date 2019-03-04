import torch
import torch.nn as nn
import torch.nn.functional as F
from rarity_of_events.distributions import Categorical


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, x, y):
        raise NotImplementedError

    def act(self, spatial_inputs, non_spatial_input, deterministic=False):
        value, policy = self(spatial_inputs, non_spatial_input)
        # action = self.dist.sample(policy, deterministic=deterministic)
        #action = policy.max(1)
        probs = F.softmax(policy)
        action = probs.multinomial(1)
        return value, action

    # def evaluate_actions(self, spatial_inputs, non_spatial_input, actions):
    #     value, x = self(spatial_inputs, non_spatial_input)
    #     action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
    #     return value, action_log_probs, dist_entropy

    def evaluate_actions(self, spatial_inputs, non_spatial_input):
        value, x = self(spatial_inputs, non_spatial_input)
        return F.softmax(x), value


class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space_shape):
        super(CNNPolicy, self).__init__()
        # num_inputs is 26 (number of feature layers)
        # kernel-size is the size of the filter that runs over the layers
        self.conv1 = nn.Conv2d(num_inputs, out_channels=52, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=52, out_channels=26, kernel_size=2)


        self.linear1 = nn.Linear(7, 28)
        self.linear2 = nn.Linear(28, 7)

        self.critic = nn.Linear(26 * 5 * 12 + 7, 1)
        self.actor = nn.Linear(26 * 5 * 12 + 7, action_space_shape)

        # num_outputs = 1
        # self.dist = Categorical(26 * 5 * 12 + 7, num_outputs)

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

        # Non-spatial input through two linear layers (fully-connected)
        y = self.linear1(non_spatial_input)
        y = F.relu(y)
        y = self.linear2(y)
        y = F.relu(y)

        # Concatenate the outputs
        concatenated = torch.cat((x.flatten(), y.flatten()), dim=0)
        concatenated = concatenated.view(-1, 26 * 5 * 12 + 7)

        # Return value, policy
        return self.critic(concatenated), self.actor(concatenated)

    def get_probs(self, inputs):
        value, x = self(inputs)
        x = self.dist(x)
        probs = F.softmax(x)
        return probs

    def get_action_probs(self, spatial_input, non_spatial_input):
        x, y = self(spatial_input, non_spatial_input)
        action_probs = F.softmax(y)
        return action_probs

