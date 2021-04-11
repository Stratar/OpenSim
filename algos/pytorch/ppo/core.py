import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def init_(m):
    if isinstance(m, nn.Linear):
        gain = torch.nn.init.calculate_gain('tanh')
        torch.nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()

        self.act_dim = num_actions
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, num_actions),
            nn.Softmax(dim=-1)
        )

        self.value_head = nn.Linear(hidden_dim, 1)

        log_std = -0.5 * np.ones(self.act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        
        self.apply(init_)

    def forward(self, x, act=None):
        std = torch.exp(self.log_std)

        hidden = self.net(x)
        mu = self.action_head(hidden)
        pi = Normal(mu, std)
        logp = None
        if act is not None:
            logp = pi.log_prob(act).sum(axis=-1)
        return pi, logp, torch.squeeze(self.value_head(hidden), -1)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(init_)

    def forward(self, x):
        return torch.squeeze(self.net(x), -1)

'''
class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def save_checkpoint(self, checkpoint_file):
        checkpoint_file += self.checkpoint
        torch.save({'state_dict': self.state_dict(), 
                'optimizer': self.optimizer.state_dict()}, 
                os.path.normpath(checkpoint_file))

    def load_checkpoint(self, checkpoint_file):
        checkpoint_file += "_actor.model"

        checkpoint = torch.load(os.path.normpath(checkpoint_file))
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

class MixedActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.hid_net = mlp([obs_dim] + list(hidden_sizes), activation)

        self.action_head = nn.Sequential(
            nn.Linear(hidden_sizes[0], act_dim),
            nn.Softmax(dim=-1)
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_sizes[0], 1),
            nn.Identity()
        )


    def pi_log_gen(self, obs, act):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

    def pi_log_val_gen(self, obs, act):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi, val = self._ppg_distribution(obs)     
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a, val

    def _ppg_distribution(self, obs):
        hid = self.hid_net(obs)
        #print("HID in ppg dist: \n", hid)
        mu = self.action_head(hid)
        std = torch.exp(self.log_std)
        val = torch.squeeze(self.value_head(hid), -1)
        return Normal(mu, std), val

    def _distribution(self, obs):
        hid = self.hid_net(obs)
        #print("HID in dist: \n", hid)
        mu = self.action_head(hid)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

    def save_checkpoint(self, checkpoint_file):
        checkpoint_file += self.checkpoint
        torch.save({'state_dict': self.state_dict(), 
                'optimizer': self.optimizer.state_dict()}, 
                os.path.normpath(checkpoint_file))

    def load_checkpoint(self, checkpoint_file):
        checkpoint_file += "_actor.model"

        checkpoint = torch.load(os.path.normpath(checkpoint_file))
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        val = torch.squeeze(self.v_net(obs), -1)
        return val # Critical to ensure v has right shape.


'''
class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(312,312), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        # policy builder depends on action space
        self.pi = Actor(obs_dim, 312, action_space.shape[0])
        '''    elif isinstance(action_space, Discrete):
                        print("--------------------------------Souldn't be here...--------------------------------")
                        self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)'''

        # build value function
        self.v  = Critic(obs_dim, 312)

    def step(self, obs):
        with torch.no_grad():
            pi, _, _ = self.pi(obs)
            act = pi.sample()
            logp_a = pi.log_prob(act).sum(axis=-1)
            val = self.v(obs)
        return act.numpy(), val.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
   
    def save_checkpoint(self, checkpoint_file):
        checkpoint_file += self.checkpoint
        torch.save({'state_dict': self.state_dict(), 
                'optimizer': self.optimizer.state_dict()}, 
                os.path.normpath(checkpoint_file))

    def load_checkpoint(self, checkpoint_file):
        checkpoint_file += "_actor.model"

        checkpoint = torch.load(os.path.normpath(checkpoint_file))
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])