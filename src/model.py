import torch
import torch.nn as nn
from torch.distributions import Normal

HIDDEN_DIM = 128
LATENT_HIDDEN_DIM = 64 # latent encoder hidden dim

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
    

class LatentEncoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=LATENT_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, z):
        return self.net(z)


class FiLMModulator(nn.Module):
    def __init__(self, latent_hidden_dim=LATENT_HIDDEN_DIM, feat_dim=HIDDEN_DIM):
        super().__init__()
        self.to_gamma_beta = nn.Linear(latent_hidden_dim, feat_dim * 2)

    def forward(self, latent_feat):
        gamma_beta = self.to_gamma_beta(latent_feat)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        # soft modulation
        # gamma = 1 + small_value, instead of starting from 0 which would zero out the features
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)
        return gamma, beta
    

class LatentDiscriminator(nn.Module):
    def __init__(self, summary_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, summary):
        return self.net(summary)


class Policy(nn.Module):
    def __init__(self, obs_dim=4, latent_dim=0, action_dim=2, action_limit=(2.0, 2.0)):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.action_scale = torch.tensor(
            action_limit, dtype=torch.float32).unsqueeze(0)

        # shared observation encoder
        self.sensor_branch = SensorBranch(
            input_dim=obs_dim, hidden_dim=HIDDEN_DIM)

        # actor conditional path
        if latent_dim > 0:
            self.latent_encoder = LatentEncoder(
                latent_dim=latent_dim, hidden_dim=LATENT_HIDDEN_DIM)
            self.actor_modulator = FiLMModulator(
                latent_hidden_dim=LATENT_HIDDEN_DIM,
                feat_dim=HIDDEN_DIM,
            )
        else:
            self.latent_encoder = None
            self.actor_modulator = None

        self.actor_mean = nn.Linear(HIDDEN_DIM, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # critic sees only observation feature
        self.critic = nn.Linear(HIDDEN_DIM, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # backbone 稍大一点，head 小一点更合理
                gain = 1.0
                if m is self.actor_mean or m is self.critic:
                    gain = 0.01
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.actor_logstd, -0.5)

    def encode_obs(self, obs):
        sensor = obs["sensor"]
        return self.sensor_branch(sensor)

    def condition_actor_feat(self, obs_feat, obs):
        if self.latent_dim <= 0:
            return obs_feat

        z = obs["latent"]
        latent_feat = self.latent_encoder(z)
        gamma, beta = self.actor_modulator(latent_feat)

        # FiLM:
        # h_a = gamma(z) * h_o + beta(z)
        actor_feat = gamma * obs_feat + beta
        return actor_feat

    def forward(self, obs):
        obs_feat = self.encode_obs(obs)
        actor_feat = self.condition_actor_feat(obs_feat, obs)
        return obs_feat, actor_feat

    def act(self, obs, deterministic=False):
        obs_feat, actor_feat = self.forward(obs)

        mean = self.actor_mean(actor_feat)
        std = self.actor_logstd.expand_as(mean).exp()
        dist = Normal(mean, std)

        action_pre_tanh = mean if deterministic else dist.sample()
        action_scaled = torch.tanh(action_pre_tanh) * \
            self.action_scale.to(action_pre_tanh.device)

        value = self.critic(obs_feat)
        log_prob = dist.log_prob(action_pre_tanh).sum(-1, keepdim=True)

        return value, action_scaled, log_prob

    def evaluate_actions(self, obs, _rhs=None, _masks=None, actions=None):
        obs_feat, actor_feat = self.forward(obs)

        mean = self.actor_mean(actor_feat)
        std = self.actor_logstd.expand_as(mean).exp()

        actions_unscaled = torch.clamp(
            actions / self.action_scale.to(actions.device),
            -0.999999,
            0.999999,
        )
        actions_pre_tanh = torch.atanh(actions_unscaled)

        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions_pre_tanh).sum(-1, keepdim=True)
        entropy = dist.entropy().mean()
        value = self.critic(obs_feat)

        return value, log_probs, entropy, None

    def get_value(self, obs):
        obs_feat = self.encode_obs(obs)
        return self.critic(obs_feat)

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        return HIDDEN_DIM
