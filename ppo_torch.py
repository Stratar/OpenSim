import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import common.mpi_pytorch as UMPI

from torch.distributions import kl_divergence
from torch.distributions import MultivariateNormal
from baselines import logger
from baselines.common.mpi_adam import MpiAdam

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.dists = []

        self.advantage = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        #logger.log("probs: ", self.probs)

        #logger.log("actions: ", self.actions)

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                np.array(self.dists),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done, dist):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        self.dists.append(dist)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
        self.dists = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=312, fc2_dims=312):
        super(ActorNetwork, self).__init__()
        
        self.checkpoint = '_actor_torch_ppo.model'
        
        self.actor = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Sigmoid()
                #nn.Tanh()
        )

        self.critic = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        #self.optimizer = MpiAdam(self.parameters(), epsilon=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        #Used to be 0.5*0.5
        self.action_var = T.full((n_actions,), 0.5*0.5).to(self.device)

    def forward(self, state):
        '''
        dist = self.actor(state)
        cov_mat = T.diag(self.action_var).to(self.device)

        #dist = MultivariateNormal(dist, cov_mat)#############################################
        dist = MultivariateNormal(dist, cov_mat)

        val = self.critic(state)
        #Somehow normalise between 0 and 1
        return dist, val
        '''
        return NotImplementedError

    def action_generation(self, state):

        dist = self.actor(state)
        cov_mat = T.diag(self.action_var).to(self.device)

        #dist = MultivariateNormal(dist, cov_mat)#############################################
        dist = MultivariateNormal(dist, cov_mat)

        return dist

    def actor_critic_generation(self, state):
        dist = self.actor(state)

        action_var = self.action_var.expand_as(dist)
        cov_mat = T.diag_embed(action_var).to(self.device)

        dist = MultivariateNormal(dist, cov_mat)

        value = self.critic(state)

        return dist, value

    def save_checkpoint(self, checkpoint_file):
        checkpoint_file += self.checkpoint
        T.save({'state_dict': self.state_dict(), 
                'optimizer': self.optimizer.state_dict()}, 
                os.path.normpath(checkpoint_file))

    def load_checkpoint(self, checkpoint_file):
        checkpoint_file += self.checkpoint

        checkpoint = T.load(os.path.normpath(checkpoint_file))
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=312, fc2_dims=312):
        super(CriticNetwork, self).__init__()

        self.checkpoint = '_critic_torch_ppo.model'

        self.critic = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        #self.optimizer = MpiAdam(self.parameters(), epsilon=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self, checkpoint_file):
        checkpoint_file += self.checkpoint
        T.save({'state_dict': self.state_dict(), 
                'optimizer': self.optimizer.state_dict()}, 
                os.path.normpath(checkpoint_file))

    def load_checkpoint(self, checkpoint_file):
        checkpoint_file += self.checkpoint

        checkpoint = T.load(os.path.normpath(checkpoint_file))
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.n_actions = n_actions

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        UMPI.sync_params(self.actor)########################################################################
        self.critic = CriticNetwork(input_dims, alpha)
        UMPI.sync_params(self.critic)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done, dist):
        self.memory.store_memory(state, action, probs, vals, reward, done, dist)

    def save_models(self, checkpoint_file):
        print('... saving models ...')
        self.actor.save_checkpoint(checkpoint_file)
        self.critic.save_checkpoint(checkpoint_file)

    def load_models(self, checkpoint_file):
        print('... loading models ...')
        self.actor.load_checkpoint(checkpoint_file)
        self.critic.load_checkpoint(checkpoint_file)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        #Not very confident about the acontinuous action selection part
        dist = self.actor.action_generation(state)

        #cov_mat = T.diag(self.actor.action_var).to(self.actor.device)

        #dist = MultivariateNormal(action_mean, cov_mat)#############################################
        #dist = Normal(action_mean, cov_mat)
        #logger.log("+++++++++++Dist: ", dist)

        value = self.critic(state)

        action = dist.sample()
        action = action.detach()
        #action = action.detach()

        #action = T.squeeze(action[0]).item()
        
        #logger.log("---------Action", action)

        #logger.log("action detach: ", action.detach())

        #probs just gives me problems, so make it zero, since it doesn't seem to be used anywhere
        probs = T.squeeze(dist.log_prob(action)).item()
        #action = action.cpu().data.numpy().flatten()
        #logger.log(action)
        '''
        probs = dist.sample(sample_shape=T.Size([self.n_actions]))
        logger.log("---------Probs", probs)

        logger.log("probs detach: ", probs.detach())
        '''
        #probs = 0
        #If I don't use action.detach() in return, use this and return action
        #action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value, dist

    def learn_ppo(self):
        #This would be the optim_epochs in line ~247
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, old_dists, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            #Keep it in memory for the PPG 
            self.memory.advantage = advantage

            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            #I guess this is equivallent to the optim batch sizes in the PPO_SGD ~238
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                #logger.log("Act shape ", actions.size())

                dist = self.actor.action_generation(states)

                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                #actions = dist.sample()
                #actions = action.detach()

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                UMPI.mpi_avg_grads(self.actor.actor)
                UMPI.mpi_avg_grads(self.critic)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        #I need to keep the memory for the auxiliary phase as well
        #self.memory.clear_memory()              

    def learn_ppg(self):
        state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, old_dists, batches = \
                    self.memory.generate_batches()

        advantage = self.memory.advantage

        advantage = T.tensor(advantage).to(self.actor.device)

        values = vals_arr
        values = T.tensor(values).to(self.actor.device)

        for batch in batches:
            states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
            dist, critic_value = self.actor.actor_critic_generation(states)
            #actions = T.tensor(action_arr[batch]).to(self.actor.device)
            #old_probs = T.tensor(old_dists[batch]).to(self.actor.device)
            #new_probs = dist.log_prob(actions)

            critic_value = T.squeeze(critic_value)

            #This is a problem in the loss, because of the grad_fn type being: <MulBackward0>
            #prob_distance = kl_divergence(old_dists[batch][0], dist).mean() + 0.01
            prob_distance = nn.KLDivLoss(size_average=False)(old_dists[batch][0].sample().detach(), dist.sample().detach())
            returns = advantage[batch] + values[batch]
            critic_loss = (returns-critic_value)**2
            critic_loss = critic_loss.mean()

            total_loss = 0.5*critic_loss + 1.0*prob_distance
            #total_loss = prob_distance

            T.autograd.set_detect_anomaly(True)

            self.actor.optimizer.zero_grad()
            total_loss.backward()
            UMPI.mpi_avg_grads(self.actor)
            self.actor.optimizer.step()