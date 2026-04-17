import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class DictRolloutStorage:
    """Rollout buffer that stores dict-based observations."""

    def __init__(self, num_steps, num_processes, obs_shapes, action_shape,
                 recurrent_hidden_state_size):
        if not isinstance(obs_shapes, dict):
            raise TypeError("obs_shapes must be a dict")

        self.obs = {}
        self.obs_keys = []
        for key, shape in obs_shapes.items():
            self.obs[key] = torch.zeros(num_steps + 1, num_processes, *shape)
            self.obs_keys.append(key)

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, *action_shape)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for key in self.obs_keys:
            self.obs[key] = self.obs[key].to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        for key, value in obs.items():
            self.obs[key][self.step + 1].copy_(value)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        for key in self.obs_keys:
            self.obs[key][0].copy_(self.obs[key][-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, gae_lambda,
                        use_proper_time_limits=True):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = (self.rewards[step]
                         + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                         - self.value_preds[step])
                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                if use_proper_time_limits:
                    gae = gae * self.bad_masks[step + 1]
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step])

    def feed_forward_generator(self, advantages, num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size, drop_last=True)

        for indices in sampler:
            obs_batch = {}
            for key in self.obs_keys:
                flat = self.obs[key][:-1].view(-1, *self.obs[key].size()[2:])
                obs_batch[key] = flat[indices]

            rhs_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices] if advantages is not None else None

            yield (obs_batch, rhs_batch, actions_batch, value_preds_batch,
                   return_batch, masks_batch, old_action_log_probs_batch, adv_targ)
