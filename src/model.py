import torch
import torch.nn as nn
from torch.distributions import Normal

HIDDEN_DIM = 128

# a 3-layer MLP
class SensorBranch(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.net(x)


class Policy(nn.Module):
    def __init__(self, obs_dim=4, action_dim=2, action_limit=(2.0, 2.0)):
        super().__init__()
        self.action_dim = action_dim
        self.action_scale = torch.tensor(action_limit, dtype=torch.float32).unsqueeze(0)

        self.sensor_branch = SensorBranch(input_dim=obs_dim)
        self.actor_mean = nn.Linear(HIDDEN_DIM, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(HIDDEN_DIM, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.actor_logstd, -0.5)

    def forward(self, obs):
        return self.sensor_branch(obs["sensor"])

    def act(self, obs, deterministic=False):
        feat = self.forward(obs)
        mean = self.actor_mean(feat)
        std = self.actor_logstd.expand_as(mean).exp()

        dist = Normal(mean, std)
        action = mean if deterministic else dist.sample()
        action_scaled = torch.tanh(action) * self.action_scale.to(action.device)
        value = self.critic(feat)

        return value, action_scaled, dist.log_prob(action).sum(-1, keepdim=True)

    def evaluate_actions(self, obs, _rhs=None, _masks=None, actions=None):
        feat = self.forward(obs)
        mean = self.actor_mean(feat)
        std = self.actor_logstd.expand_as(mean).exp()

        actions_unscaled = torch.clamp(
            actions / self.action_scale.to(actions.device), -0.999999, 0.999999)
        actions_unscaled = torch.atanh(actions_unscaled)

        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions_unscaled).sum(-1, keepdim=True)
        entropy = dist.entropy().mean()
        value = self.critic(feat)

        return value, log_probs, entropy, None

    def get_value(self, obs):
        feat = self.forward(obs)
        return self.critic(feat)

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        return HIDDEN_DIM
