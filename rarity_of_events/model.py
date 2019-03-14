import torch
import torch.nn as nn
import torch.nn.functional as F


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, x, y):
        raise NotImplementedError

    def act(self, spatial_inputs, non_spatial_input):
        value, policy = self(spatial_inputs, non_spatial_input)
        probs = F.softmax(policy)
        action = probs.multinomial(1)
        return value, action

    def evaluate_actions(self, spatial_inputs, non_spatial_input):
        value, x = self(spatial_inputs, non_spatial_input)
        return F.softmax(x), value


class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space_shape):
        super(CNNPolicy, self).__init__()

        # num_inputs is 26 (number of feature layers)
        self.conv1 = nn.Conv2d(num_inputs, out_channels=52, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=52, out_channels=26, kernel_size=2)

        # Linear layers
        self.linear1 = nn.Linear(49, 98)
        self.linear2 = nn.Linear(98, 49)

        # The actor and the critic outputs
        self.critic = nn.Linear(26 * 5 * 12 + 49, 1)
        self.actor = nn.Linear(26 * 5 * 12 + 49, action_space_shape)

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        self.linear2.weight.data.mul_(relu_gain)

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
        y = self.linear2(y) # kan være det ikke er nødvendigt
        y = F.relu(y)

        # Concatenate the outputs
        concatenated = torch.cat((x.flatten(), y.flatten()), dim=0)
        concatenated = concatenated.view(-1, 26 * 5 * 12 + 49)

        # Return value, policy
        return self.critic(concatenated), self.actor(concatenated)

    def get_action_probs(self, spatial_input, non_spatial_input):
        x, y = self(spatial_input, non_spatial_input)
        action_probs = F.softmax(y)
        return action_probs

