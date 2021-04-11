import numpy as np

from numpy import genfromtxt
import torch
from torch.optim import Adam
import gym

import os
from osim.env import ProstheticsEnvMulticlip

import time
#I WAS IMPORTING THE WRONG THING
import algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        self.true_buf = np.zeros(size, dtype=np.float32)

    def store(self, obs, act, rew, val, logp, true_reward):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp

        self.true_buf[self.ptr] = true_reward

        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        #true_rews = np.append(self.true_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        #self.true_buf[path_slice] = core.discount_cumsum(true_rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        #self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

    def dump(self):
        self.ptr, self.path_start_idx = 0, 0

    def get_true_reward(self):
        return np.mean(self.true_buf)


def ppo(model_file, load_after_iters, 
        restore_model_from_file=1, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=4, train_v_iters=4, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_after=5, viz=False):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment

    #env = ProstheticsEnvMulticlip(visualize=viz, model_file=model_file, integrator_accuracy=1e-2)
    #env = gym.make(model_file)
    #env = env_fn()
    env = gym.make('Pendulum-v0')
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    #obs_dim = env.observation_space.shape[0]
    #act_dim = env.action_space.shape[0]

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    save_prefix = model_file[10:-5]

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    def load_actor_model(checkpoint_file):
        logger.log('... loading actor model ...')

        checkpoint_file += "_actor.model"

        checkpoint = torch.load(os.path.normpath(checkpoint_file))
        ac.pi.load_state_dict(checkpoint['state_dict'])
        pi_optimizer.load_state_dict(checkpoint['optimizer'])

    def load_critic_model(checkpoint_file):
        logger.log('... loading critic model ...')

        checkpoint_file += "_critic.model"

        checkpoint = torch.load(os.path.normpath(checkpoint_file))
        ac.v.load_state_dict(checkpoint['state_dict'])
        vf_optimizer.load_state_dict(checkpoint['optimizer'])
   
    def save_actor_model(checkpoint_file):
        checkpoint_file += "_actor.model"
        torch.save({'state_dict': ac.pi.state_dict(), 
                'optimizer': pi_optimizer.state_dict()}, 
                os.path.normpath(checkpoint_file))

   
    def save_critic_model(checkpoint_file):
        checkpoint_file += "_critic.model"
        torch.save({'state_dict': ac.v.state_dict(), 
                'optimizer': vf_optimizer.state_dict()}, 
                os.path.normpath(checkpoint_file))

    n_episodes = 0
    n_steps = 0
    epoch = 0
    true_history = []
    logger.log("Loading value is %i" % restore_model_from_file)
    if restore_model_from_file == 1:
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_f = os.path.normpath(
            base_path + "/../../../" + save_prefix + '/models/' + save_prefix + "_afterIter_" + str(
                load_after_iters))
        #Use the model_f as destination location, so change the load_models()
        logger.log("!Trying to load from {}".format(model_f))        
        load_actor_model(model_f)
        load_critic_model(model_f)
        logger.log("Loaded model from {}".format(model_f))
        # Restore the variables from file
        data = genfromtxt(save_prefix + '/test_afterIter_' + str(load_after_iters) + '.csv', delimiter=',')
        num_rows = sum(1 for row in data)
        logger.log(data[0])
        logger.log(num_rows)
        for i in range(len(data)):
            '''n_episodes = int(data[0])
                                n_steps = int(data[1])
                                epoch = int(data[2])
                                    true_history.append(int(data[3]))'''
            data_vector = data[i]
            n_episodes = int(data_vector[0])
            n_steps = int(data_vector[1])
            epoch = int(data_vector[2])
            true_history.append(int(data_vector[3]))

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    #local_steps_per_epoch = int(steps_per_epoch / num_procs())
    local_steps_per_epoch = steps_per_epoch
    logger.log("Will start at: %i epocs" % epoch)
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp, _ = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi.net)    # average grads across MPI processes
            mpi_avg_grads(ac.pi.action_head)    # average grads across MPI processes
            #mpi_avg_grads(ac.pi)
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    def aux_update():

        for i in range(6):
            data = buf.get()

            obs, act, adv, logp_old, ret = data['obs'], data['act'], data['adv'], data['logp'], data['ret']

            # Policy loss
            pi, logp, val = ac.pi(obs)
            prob_distance = torch.nn.KLDivLoss(size_average=False)(logp, logp_old)
            '''logger.log("Logp old and logp:")
                                                logger.log(logp_old)
                                                logger.log(logp)'''

            #actions = T.tensor(action_arr[batch]).to(self.actor.device)
            #old_probs = T.tensor(old_dists[batch]).to(self.actor.device)
            #new_probs = dist.log_prob(actions)

            #Make it clipped????
            '''logger.log("Value and Ret:")
                                                logger.log(val)
                                                logger.log(ret)'''
            value_loss = ((val - ret)**2).mean()
            '''for param in ac.pi.parameters():
                                                    logger.log("Before backward")
                                                    logger.log(param)'''

            pi_optimizer.zero_grad()
            loss_pi = (value_loss + prob_distance).mean()
            '''logger.log("The losses of PPG are:")
                                                logger.log(prob_distance)
                                                logger.log(value_loss)
                                                logger.log(loss_pi)'''

            loss_pi.backward()
            '''  for param in ac.pi.parameters():
                                              logger.log("After backward")
                                              logger.log(param.grad)'''
            mpi_avg_grads(ac.pi) # average grads across MPI processes
            pi_optimizer.step()
            #This is a problem in the loss, because of the grad_fn type being: <MulBackward0>
            #prob_distance = kl_divergence(old_dists[batch][0], dist).mean() + 0.01
            #prob_distance = nn.KLDivLoss(size_average=False)(old_dists[batch][0].sample().detach(), dist.sample().detach())

            vf_optimizer.zero_grad()
            loss_v = ((ac.v(obs) - ret)**2).mean()
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

    # Prepare for interaction with environment
    start_time = time.time()
    #o, ep_ret, ep_len = env.reset(test=False), 0, 0
    o, ep_ret, ep_len = env.reset(), 0, 0
    # Main loop: collect experience in env and update/log each epoch
    while True: 
        if epoch == epochs:
            logger.log("Done!")
            break

        logger.log("********** Iteration %i ************" % epoch)
        ep_true = 0
        loop_t_value = 0
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            #next_o, r, true_reward, d = env.step(a)
            next_o, r, d, true_reward = env.step(a)
            true_reward = 0
            ep_ret += r
            ep_len += 1
            n_steps += 1
            ep_true += true_reward

            loop_t_value = t

            # save and log
            buf.store(o, a, r, v, logp, true_reward)
            logger.store(VVals=v)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    logger.log('Warning: trajectory cut off by epoch at %i steps.' % ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished

                    n_episodes += 1
                    true_history.append(ep_true)
                    logger.store(TrueRew=ep_true)

                    #o = env.reset(test=False)
                    o = env.reset()
                    d = False
                    ep_true = 0


                    logger.store(EpRet=ep_ret, EpLen=ep_len)


                #o, ep_ret, ep_len = env.reset(test=False), 0, 0
                o, ep_ret, ep_len = env.reset(), 0, 0
        logger.store(Episodes=n_episodes)
        true_history.append(ep_true)
        # Save model
        #Save env, or the state dictionary?
        if (epoch % save_after == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)\
        #PPO
        update()

        #Perform Auxiliary update! add: and epoch > 0
        #if(epoch%32==0):
            #logger.log("PPG UPDATE START!")
            #aux_update()

        buf.dump()

        log_true = logger.get_stats("TrueRew")[0]
        log_rew = logger.get_stats("EpRet")[0]

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=False)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=False)
        #logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        '''logger.log_tabular('Entropy', average_only=True)
                                logger.log_tabular('KL', average_only=True)
                                logger.log_tabular('ClipFrac', average_only=True)
                                logger.log_tabular('StopIter', average_only=True)'''
        logger.log_tabular('TrueRew', average_only=True)
        logger.log_tabular('Episodes', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

        if proc_id() == 0:
            l = open(save_prefix + "/iterations.txt", "a+")
            m = open(save_prefix + "/timesteps.txt", "a+")                                                                  
            n = open(save_prefix + "/training_mean_truerewards.txt", "a+")                                                                  
            r = open(save_prefix + "/training_mean_rewards.txt", "a+")
            
            n.write("Episode %d    " % n_episodes)
            n.write("Reward  %d\r\n" % log_true)

            r.write("Episode %d    " % n_episodes)
            r.write("Reward  %d\r\n" % log_rew)

            #ALSO STORE THE TIMESTEPS SO IT CAN STOP AND RESTART PROPERLY
            if epoch % save_after == 0:
                l.write("%d\r\n" % epoch)

            m.write("%d\r\n" % n_steps)
            
            l.close()
            m.close()
            n.close()
            r.close()
            #It has been indented once
            if epoch % save_after == 0:
            #if save_model_with_prefix:
                base_path = os.path.dirname(os.path.abspath(__file__))
                model_f = os.path.normpath(
                    base_path + '/../../../' + save_prefix + '/models/' + save_prefix + "_afterIter_" + str(
                        epoch))
                #Use the model_f as destination location, so change the save_models()
                save_actor_model(model_f)
                save_critic_model(model_f)
                logger.log("Saved model to file :{}".format(model_f))
                if n_episodes < 100:
                    size = n_episodes
                else:
                    size = 100
                asd = np.zeros((size, 4), dtype=np.int32)
                for i in range(size):
                    asd[i] = [n_episodes, n_steps, epoch, true_history[i]]
                    np.savetxt(save_prefix + '/test_afterIter_' + str(epoch) + '.csv', asd, delimiter=",")

        epoch += 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('restore', type=int, default=1)
    parser.add_argument('load_iters', type=int, default=1)
    parser.add_argument('env', type=str, default='HalfCheetah-v2')
    parser.add_argument('cpu', type=int, default=1)
    parser.add_argument('--hid', type=int, default=312)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)#1537#100#4000
    parser.add_argument('--epochs', type=int, default=1500)#825#1500
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    #mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    load_iters = args.load_iters

    if args.load_iters == 1:
        with open(args.env[10:-5] + '/iterations.txt', 'r') as f:
            lines = f.read().splitlines()
            last_iter = int(lines[-1])
            load_iters = last_iter

    ppo(model_file = args.env, load_after_iters=load_iters, 
        restore_model_from_file=args.restore, actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, viz=False)