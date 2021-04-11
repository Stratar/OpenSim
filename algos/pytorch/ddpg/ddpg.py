import os

from copy import deepcopy
import numpy as np
from numpy import genfromtxt
from collections import deque
import torch
from torch.optim import Adam
import gym
import time
import algos.pytorch.ddpg.core as core
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from mpi4py import MPI

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

    def save_buffer(self, checkpointfile):
        np.save(checkpointfile+"_obs.npy", self.obs_buf)
        np.save(checkpointfile+"_obs2.npy", self.obs2_buf)
        np.save(checkpointfile+"_act.npy", self.act_buf)
        np.save(checkpointfile+"_rew.npy", self.rew_buf)
        np.save(checkpointfile+"_done.npy", self.done_buf)
        a = open(checkpointfile + "_vals.txt", "a+")
        a.write("%d\n" % self.ptr)
        a.write("%d\n" % self.size)
        a.write("%d\n" % self.max_size)
        a.close()

    def load_buffer(self, checkpointfile):
        self.obs_buf = np.load(checkpointfile+"_obs.npy")
        self.obs2_buf = np.load(checkpointfile+"_obs2.npy")
        self.act_buf = np.load(checkpointfile+"_act.npy")
        self.rew_buf = np.load(checkpointfile+"_rew.npy")
        self.done_buf = np.load(checkpointfile+"_done.npy")
        with open(checkpointfile+'_vals.txt', 'r') as a:
            lines = a.read().splitlines()
            self.ptr = int(lines[0])
            self.size = int(lines[1])
            self.max_size = int(lines[2])


def ddpg(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
         steps_per_epoch=1000, epochs=100, replay_size=int(1e6), gamma=0.99, 
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=128, start_steps=10000, 
         update_after=250, update_every=64, act_noise=0.1, num_test_episodes=10, 
         max_ep_len=1000, logger_kwargs=dict(), save_freq=1, save_prefix = "art",
         save_after = 5, restore_model_from_file = 1, load_after_iters = 0):
    """
    Deep Deterministic Policy Gradient (DDPG)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of 
            actions as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    #setup_pytorch_for_mpi() 

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    def save_target_actor_model(checkpoint_file):
        logger.log('... Saving target actor model ...')
        checkpoint_file += "_target_actor.model"
        torch.save({'state_dict': ac_targ.pi.state_dict()},
                    os.path.normpath(checkpoint_file))
        logger.log("Saved model to file:{}".format(checkpoint_file))

    def save_target_critic_model(checkpoint_file):
        checkpoint_file += "_target_critic.model"
        torch.save({'state_dict': ac_targ.q.state_dict()},
                    os.path.normpath(checkpoint_file))
        logger.log("Saved model to file:{}".format(checkpoint_file))

    def load_target_actor_model(checkpoint_file):
        checkpoint_file += "_target_actor.model"

        checkpoint = torch.load(os.path.normpath(checkpoint_file))
        ac_targ.pi.load_state_dict(checkpoint['state_dict'])
        logger.log("Loaded file:{}".format(checkpoint_file))


    def load_target_critic_model(checkpoint_file):
        logger.log('... Loading target critic model ...')
        checkpoint_file += "_target_critic.model"

        checkpoint = torch.load(os.path.normpath(checkpoint_file))
        ac_targ.q.load_state_dict(checkpoint['state_dict'])
        logger.log("Loaded file:{}".format(checkpoint_file))


    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    tot_time = 0
    iters_so_far = 0
    ep_so_far = 0
    prev_ep_so_far = 0
    timesteps_so_far = 0

    lenbuffer = deque(maxlen=100)
    rewbuffer = deque(maxlen=100)

    #sync_params(ac)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = ac.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    def save_actor_model(checkpoint_file):
        #opt_file = checkpoint_file + "_actor_opt.model"
        checkpoint_file += "_actor.model"
        
        saves = {
            'state_dict' : ac.pi.state_dict(),
            'optimizer' : pi_optimizer.state_dict(),
        }
        torch.save(saves, os.path.normpath(checkpoint_file))
        #torch.save(pi_optimizer, os.path.normpath(opt_file))
        logger.log("Saved model to file:{}".format(checkpoint_file))

    def save_critic_model(checkpoint_file):
        #logger.log('... Saving critic model ...')
        opt_file = checkpoint_file + "_critic_opt.model"
        checkpoint_file += "_critic.model"
        saves = {
            'state_dict' : ac.q.state_dict(),
            'optimizer' : q_optimizer.state_dict(),
        }
        torch.save(saves, os.path.normpath(checkpoint_file))
        #torch.save(q_optimizer, os.path.normpath(opt_file))
        logger.log("Saved model to file:{}".format(checkpoint_file))


    def save_everything(checkpoint_file):
        #checkpoint_file += "_various_NNs.model"
        checkpoint_file = "Trials/all.tar"
        chk = {
        'actor_network' : ac.pi.state_dict(),
        'critic_network' : ac.q.state_dict(),
        'target_actor_network' : ac_targ.pi.state_dict(),
        'target_critic_network' : ac_targ.q.state_dict(),
        'pi_optimizer' : pi_optimizer.state_dict(),
        'q_optimizer' : q_optimizer.state_dict(),
        }
        torch.save(chk, os.path.normpath(checkpoint_file))
        logger.log("Saved model to file:{}".format(checkpoint_file))
    
    def load_everything(checkpoint_file):
        #checkpoint_file += "_various_NNs.model"
        checkpoint_file = "Trials/all.tar"
        checkpoint = torch.load(os.path.normpath(checkpoint_file))
        
        ac.pi.load_state_dict(checkpoint['actor_network'])
        ac.q.load_state_dict(checkpoint['critic_network'])
        ac_targ.pi.load_state_dict(checkpoint['target_actor_network'])
        ac_targ.q.load_state_dict(checkpoint['target_critic_network'])
        pi_optimizer.load_state_dict(checkpoint['pi_optimizer'])
        q_optimizer.load_state_dict(checkpoint['q_optimizer'])

        logger.log("Loaded file:{}".format(checkpoint_file))


    def load_actor_model(checkpoint_file):
        #logger.log("... loading actor model ...")
        #opt_file = checkpoint_file + "_actor_opt.model"
        checkpoint_file += "_actor.model"


        checkpoint = torch.load(os.path.normpath(checkpoint_file))
        ac.pi.load_state_dict(checkpoint['state_dict'])
        #for p in ac.pi.parameters():
        #    p.requires_grad = True
        #pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
        pi_optimizer.load_state_dict(checkpoint['optimizer'])
        #ac.pi = torch.load(os.path.normpath(checkpointfile))
        #pi_optimizer = torch.load(os.path.normpath(opt_file))

        logger.log("Loaded file:{}".format(checkpoint_file))


    def load_critic_model(checkpoint_file):
        #opt_file = checkpoint_file + "_critic_opt.model"
        checkpoint_file += "_critic.model"

        checkpoint = torch.load(os.path.normpath(checkpoint_file))
        
        ac.q.load_state_dict(checkpoint['state_dict'])
        #for p in ac.q.parameters():
        #    p.requires_grad = True

        #q_optimizer = Adam(ac.q.parameters(), lr = q_lr)
        q_optimizer.load_state_dict(checkpoint['optimizer'])
        #ac.q = torch.load(os.path.normpath(checkpoint_file))
        #q_optimizer = torch.load(os.path.normpath(opt_file))
        logger.log("Loaded file:{}".format(checkpoint_file))




    # Set up model saving
    logger.setup_pytorch_saver(ac)
    if restore_model_from_file == 1:
        #pi_optimizer = None
        #q_optimizer = None
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_f = os.path.normpath(
            base_path + "/../../../" + save_prefix + '/models/' + save_prefix + "_afterIter_" + str(
                load_after_iters))
        model_buff = os.path.normpath(
            base_path + "/../../../" + save_prefix + '/buffers/' + save_prefix + "_afterIter_" + str(
                load_after_iters))
        
        #ac = torch.load('/home/leonidas/Desktop/data/ddpg/ddpg_s0/pyt_save/model.pt')
        #load_actor_model(model_f)
        #load_critic_model(model_f)
        load_everything(model_f)
        #load_target_actor_model(model_f)
        #load_target_critic_model(model_f)
        #ac_targ = deepcopy(ac)

        replay_buffer.load_buffer(model_buff)
        update_after = 0
        logger.log("... Loading Complete ...")
        '''logger.log("---------- Pi optimizer: ----------")
        for var in pi_optimizer.state_dict():
            print(var, "\t", pi_optimizer.state_dict()[var])

        logger.log("---------- Q optimizer: ----------")
        for var in q_optimizer.state_dict():
            print(var, "\t", q_optimizer.state_dict()[var])
        #logger.log(replay_buffer.act_buf)'''
        data = genfromtxt(save_prefix + '/test_after_Iter' + str(load_after_iters) + '.csv', delimiter=',')
        for i in range(len(data)):
            data_vector = data[i]
            ep_so_far = int(data_vector[0])
            timesteps_so_far = int(data_vector[1])
            iters_so_far = int(data_vector[2])
            time_elapsed = int(data_vector[3])
            lenbuffer.append(int(data_vector[4]))
            rewbuffer.append(int(data_vector[5]))
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    logger.log('\nLoaded Number of parameters: \t pi: %d, \t q: %d\n'%var_counts)
    #sync_params(ac)
    #sync_params(ac_targ)


    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        #mpi_avg_grads(ac.q.q)
        q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        #mpi_avg_grads(ac.pi.pi)
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Record things
        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def flatten_lists(listoflists):
        return [el for list_ in listoflists for el in list_]    

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    epoch = iters_so_far
    episodes = 0


    ep_ret_arr = []
    ep_len_arr = []
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        if t % steps_per_epoch == 0:
            logger.log("********** Iteration %i ************" % epoch)
            ep_ret_arr = []
            ep_len_arr = []        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d
        if d:
            episodes += 1

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            ep_ret_arr.append(ep_ret)
            ep_len_arr.append(ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch += 1

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            lrlocal = (ep_len_arr, ep_ret_arr)
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)
            lens, rews = map(flatten_lists, zip(*listoflrpairs))
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)

            prev_ep_so_far = ep_so_far
            ep_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1

            # Log info about epoch
            logger.log_tabular("EpLenMean", np.mean(lenbuffer))
            logger.log_tabular("EpRewMean", np.mean(rewbuffer))
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular("EpSoFar", ep_so_far)
            logger.log_tabular('EpthisIter', len(lens))
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.log_tabular('BufferSize', replay_buffer.size)

            if MPI.COMM_WORLD.Get_rank() == 0:
                f = open(save_prefix + "/training_rewards.txt", "a+")
                g = open(save_prefix + "/training_episode_lengths.txt", "a+")
                h = open(save_prefix + "/training_mean_reward.txt", "a+")
                k = open(save_prefix + "/training_mean_lengths.txt", "a+")
                l = open(save_prefix + "/iterations.txt", "a+")
                m = open(save_prefix + "/timesteps.txt", "a+")
                
                for i in range((ep_so_far - prev_ep_so_far)):
                    f.write("Episode %d    " % (prev_ep_so_far + i))
                    f.write("Reward  %d\r\n" % rews[i])
                    g.write("Episode %d    " % (prev_ep_so_far + i))
                    g.write("Length  %d\r\n" % lens[i])
                
                h.write("Episode %d    " % ep_so_far)
                h.write("Reward  %d\r\n" % np.mean(rews))
                k.write("Episode %d    " % ep_so_far)
                k.write("Length  %d\r\n" % np.mean(lens))

                if iters_so_far % save_after == 0:
                    l.write("%d\r\n" % iters_so_far)
                    m.write("%d\r\n" % t)
                    

                f.close()
                g.close()
                k.close()
                h.close()
                l.close()
                m.close()

            logger.dump_tabular()
            
            if MPI.COMM_WORLD.Get_rank() == 0 and iters_so_far % save_after == 0:

                '''logger.log("---------- Pi optimizer: ----------")
                for var in pi_optimizer.state_dict():
                    print(var, "\t", pi_optimizer.state_dict()[var])

                logger.log("---------- Q optimizer: ----------")
                for var in q_optimizer.state_dict():
                    print(var, "\t", q_optimizer.state_dict()[var])'''
                base_path = os.path.dirname(os.path.abspath(__file__))
                model_f = os.path.normpath(
                    base_path + '/../../../' + save_prefix + '/models/' + save_prefix + "_afterIter_" + str(
                        iters_so_far))
                model_buff = os.path.normpath(
                    base_path + "/../../../" + save_prefix + '/buffers/' + save_prefix + "_afterIter_" + str(
                        iters_so_far))
                #save_actor_model(model_f)
                #save_critic_model(model_f)
                #save_target_actor_model(model_f)
                #save_target_critic_model(model_f)
                save_everything(model_f)
                replay_buffer.save_buffer(model_buff)
                logger.log("... Saving Complete ...")
                #logger.log(replay_buffer.act_buf)
                if episodes < 100:
                    size = episodes
                else:
                    size = 100
                asd = np.zeros((size, 6), dtype = np.int32)
                for i in range(size):
                    asd[i] = [ep_so_far, timesteps_so_far, iters_so_far, tot_time, lenbuffer[i], rewbuffer[i]]
                np.savetxt(save_prefix + '/test_after_Iter' + str(iters_so_far) + '.csv', asd, delimiter = ",")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('restore', type=int, default=1)
    parser.add_argument('load_iters', type=int, default=1)
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--hid', type=int, default=312)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=825)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    #mpi_fork(args.cpu)

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    load_iters = args.load_iters


    if args.load_iters == 1:
    	with open(args.env + '/iterations.txt', 'r') as f:
    		lines = f.read().splitlines()
    		last_iter = int(lines[-1])
    		load_iters = last_iter
	
    ddpg(lambda : gym.make('Pendulum-v0'), actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs, save_prefix = args.env, 
         restore_model_from_file = args.restore, load_after_iters = load_iters)


