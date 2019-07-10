import torch


class Memory(object):
    def __init__(self, num_steps, num_processes, spatial_obs_shape, non_spatial_shape):
        self.spatial_obs = torch.zeros(num_steps + 1, num_processes, *spatial_obs_shape)
        self.non_spatial_obs = torch.zeros(num_steps + 1, num_processes, *non_spatial_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_predictions = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        action_shape = 1
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        # action_space = 242 # 1v1
        # action_space = 1078  # 3v3
        # action_space = 530
        # 3v3
        # action_space = 492
        # 5v5
        action_space = 908
        self.available_actions = torch.zeros(num_steps + 1, num_processes, action_space, dtype=torch.uint8)

    def cuda(self):
        self.spatial_obs = self.spatial_obs.cuda()
        self.non_spatial_obs = self.non_spatial_obs.cuda()
        self.rewards = self.rewards.cuda()
        self.value_predictions = self.value_predictions.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, step, spatial_obs, non_spatial_obs, action, value_pred, reward, mask, available_actions):
        self.spatial_obs[step + 1].copy_(spatial_obs)
        self.non_spatial_obs[step + 1].copy_(non_spatial_obs)
        self.actions[step].copy_(action)
        # self.positions[step].copy_(position)
        self.value_predictions[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step].copy_(mask)
        self.available_actions[step].copy_(available_actions)


    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * gamma * self.masks[step] + self.rewards[step]
                # gamma + self.rewards[step]
