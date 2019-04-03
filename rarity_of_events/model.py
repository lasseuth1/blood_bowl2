import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, x, y):
        raise NotImplementedError

    def act(self, spatial_inputs, non_spatial_input, avail_actions, envs):
        value, policy, position = self(spatial_inputs, non_spatial_input)
        num_processes = envs.num_envs

        probs_action = F.softmax(policy, dim=0)
        temp_action = probs_action * avail_actions
        probs_action *= avail_actions
        summed_actions = torch.sum(probs_action, dim=1)
        normalized_actions = torch.div(probs_action, summed_actions.view(num_processes, 1))

        try:
            actions = normalized_actions.multinomial(1)
        except RuntimeError:
            actions = avail_actions.multinomial(1)

        avail_positions = envs.positions(actions)
        probs_position = F.softmax(position, dim=0)
        temp_position = probs_position * avail_positions
        probs_position *= avail_positions
        summed_pos = torch.sum(probs_position, dim=1)
        normalized_pos = torch.div(probs_position, summed_pos.view(num_processes, 1))

        # Make action objects
        action_objects = []
        positions_collected = torch.zeros(num_processes, 1)
        for i in range(num_processes):
            action = actions[i]  # get action
            pos = normalized_pos[i]  # get position tensor
            try:
                pos = pos.multinomial(1)
                pos = pos.item()
            except RuntimeError:
                pos = avail_positions[i].multinomial(1)
                pos = pos.item()

            action_object = {
                'action-type': action.item(),
                'x': int(pos % 14) if pos is not 98 else None,
                'y': int(pos / 14) if pos is not 98 else None,
            }

            positions_collected[i] = pos

            action_objects.append(action_object)

        return value, actions, action_objects, positions_collected
        # return value, actions, position

    def evaluate_actions(self, spatial_inputs, non_spatial_input, actions, positions):
        """
        Calls forward() function
        """
        value, x, pos = self(spatial_inputs, non_spatial_input)

        log_probs = F.log_softmax(x)
        probs = F.softmax(x)
        action_log_probs = log_probs.gather(1, actions)

        pos_log_probs = F.log_softmax(pos)
        pos_probs = F.softmax(pos)
        position_log_probs = pos_log_probs.gather(1, positions)

        dist_entropy = -(log_probs * probs).sum(-1).mean()

        return action_log_probs, value, position_log_probs, dist_entropy


class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space_shape):
        super(CNNPolicy, self).__init__()

        # num_inputs is 26 (number of feature layers)
        self.conv1 = nn.Conv2d(num_inputs, out_channels=52, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=52, out_channels=26, kernel_size=2)

        # Linear layers
        self.linear1 = nn.Linear(49, 24)
        # self.linear2 = nn.Linear(98, 49)

        # The actor and the critic outputs
        self.critic = nn.Linear(26 * 5 * 12 + 24, 1)
        self.actor = nn.Linear(26 * 5 * 12 + 24, action_space_shape)
        # chose a position among the 7 * 14 squares on the board
        self.position = nn.Linear(26 * 5 * 12 + 24, 7 * 14 + 1)

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        # self.linear2.weight.data.mul_(relu_gain)

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
        # y = self.linear2(y) # kan være det ikke er nødvendigt
        # y = F.relu(y)

        # Concatenate the outputs
        concatenated = torch.cat((x.flatten(), y.flatten()), dim=0)
        # Reshaping the tensor to have a dimensions (1, 1609)
        concatenated = concatenated.view(-1, 26 * 5 * 12 + 24)

        # Return value, policy, spatial-position
        return self.critic(concatenated), self.actor(concatenated), self.position(concatenated)

    def get_action_probs(self, spatial_input, non_spatial_input):
        x, y = self(spatial_input, non_spatial_input)
        action_probs = F.softmax(y)
        return action_probs