import numpy as np
import sys
import os

import time
import common.mpi_pytorch as UMPI

from numpy import genfromtxt

from mpi4py import MPI
from ppo_torch import Agent
from osim.env import ProstheticsEnvMulticlip
from baselines import logger
from baselines.common import set_global_seeds
#from utils import plot_learning_curve

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#if __name__ == '__main__':
def train(num_timesteps, seed, model_file, save_model_with_prefix, restore_model_from_file, save_after,
          load_after_iters, viz=False, stochastic=True):

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])


    env = ProstheticsEnvMulticlip(visualize=viz, model_file=model_file, integrator_accuracy=1e-2)
    save_prefix = model_file[10:-5]

    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    N = 1536
    batch_size = 512
    n_epochs = 4
    n_aux_epochs = 6
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.shape[0], batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape[0])
    n_games = 300

    env.seed(workerseed)

    best_score = env.reward_range[0]
    score_history = []
    true_rew_history = []                        

    #The iterations used in pposgd_simple.py
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    curr_n_steps = 0
    avg_true_rew = 0
    n_episodes = 0
    initial_time = start_time = time.time()
    learn_time = 0
    

    #These are used for keeping track of the data that we want to maintain across the learning process of the algorithm
    if restore_model_from_file == 1:
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_f = os.path.normpath(
            base_path + "/" + save_prefix + '/models/' + save_prefix + "_afterIter_" + str(
                load_after_iters))
        #Use the model_f as destination location, so change the load_models()
        logger.log("-Trying to load from {}".format(model_f))        
        agent.load_models(model_f)
        logger.log("Loaded model from {}".format(model_f))
        # Restore the variables from file
        data = genfromtxt(save_prefix + '/test_afterIter_' + str(load_after_iters) + '.csv', delimiter=',')
        for i in range(len(data)):
            data_vector = data[i]
            n_episodes = int(data_vector[0])
            n_steps = int(data_vector[1])
            learn_iters = int(data_vector[2])
            score_history.append(int(data_vector[3]))
            true_rew_history.append(int(data_vector[4]))

    UMPI.setup_pytorch_for_mpi()

    #max_terations or max_timesteps reflect the while TRUE in the learning
    #for i in range(n_games):
    while True:
        #Add all the constraints from the pposgd_simple
        if n_steps >= (num_timesteps*1.1):
            logger.log("Max Timesteps reached")
            break
        #logger.log("********** Steps %i ************" % n_steps)
        observation = env.reset(test=False)
        done = False
        score = 0
        cur_ep_true_ret = 0
        #This depends on whether the trajectory we got has been done, or is
        #new. In both cases we reset the environment

        while not done:
            action, prob, val, dist = agent.choose_action(observation)
            #logger.log("Prob: ", prob)
            #I guess the done is like the new in pos_sgd
            #logger.log("OUTSIDE ACTION: ", action)
            #logger.log("Action list", action)
            observation_, reward, true_rew, done= env.step(action.cpu().data.numpy().flatten())
            n_steps += 1
            score += reward
            cur_ep_true_ret += true_rew
            agent.remember(observation, action.cpu().data.numpy().flatten(), prob, val, reward, done, dist)
            #Learn every % N, or horizon (timesteps_per_actorbatch) times
            if n_steps % N == 0:
                complete = time.time()-initial_time
                logger.log("Completed trajectories after: %i" % complete)
                logger.log("********** Iteration %i ************" % learn_iters)
                learn_time = time.time()
                agent.learn_ppo()

                agent.learn_ppg()

                agent.memory.clear_memory() 
                '''  
                lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"])  # local values
                listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
                lens, rews, truerews = map(flatten_lists, zip(*listoflrpairs))
                '''
                #Could move this in the training loop
                if MPI.COMM_WORLD.Get_rank() == 0:
                    l = open(save_prefix + "/iterations.txt", "a+")
                    m = open(save_prefix + "/timesteps.txt", "a+")
                    n = open(save_prefix + "/training_mean_truerewards.txt", "a+")
                    
                    n.write("Episode %d    " % curr_n_steps)
                    n.write("Reward  %d\r\n" % np.mean(true_rew_history))

                    if learn_iters % save_after == 0:
                        l.write("%d\r\n" % learn_iters)
                    m.write("%d\r\n" % n_steps)
                    l.close()
                    m.close()
                    n.close()
                    #It has been indented once
                    if learn_iters % save_after == 0:
                        if save_model_with_prefix:
                            base_path = os.path.dirname(os.path.abspath(__file__))
                            model_f = os.path.normpath(
                                base_path + '/' + save_prefix + '/models/' + save_prefix + "_afterIter_" + str(
                                    learn_iters))
                            print(model_f)
                            logger.log("Saved model to file :{}".format(model_f))
                            #Use the model_f as destination location, so change the save_models()
                            agent.save_models(model_f)
                            if n_episodes < 100:
                                size = n_episodes
                            else:
                                size = 100
                            asd = np.zeros((size, 5), dtype=np.int32)
                            for i in range(size):
                                asd[i] = [n_episodes, n_steps, learn_iters,
                                          score_history[i], true_rew_history[i]]
                                np.savetxt(save_prefix + '/test_afterIter_' + str(learn_iters) + '.csv', asd, delimiter=",")

                learn_complete = time.time()- learn_time
                total_time = time.time() - start_time
                #Change to logger.log()
                print('Process rank: ', MPI.COMM_WORLD.Get_rank())
                print('\n#####################\n Episodes', n_episodes, '\nAvg score %.1f' % avg_score,
                                '\nAvg true reward', avg_true_rew, '\nTime steps', 
                                n_steps, '\nAfter:', learn_complete, 
                                '\nTotal time:', total_time, ' \n#####################\n')
                initial_time = time.time()
                learn_iters += 1
            observation = observation

        curr_n_steps = n_steps - curr_n_steps
        #Update the numbers of episodes
        n_episodes += 1
        score_history.append(score)
        true_rew_history.append(cur_ep_true_ret)
        #avg_score = np.mean(score_history[-100:])
        #avg_true_rew = np.mean(true_rew_history[-100:])
        avg_score = np.mean(score_history)
        avg_true_rew = np.mean(true_rew_history)

        

        #agent.save_models()
        #figure_file = 'plots/scores.png'
        #Can substitute for true_rew_history too
        #x = [i+1 for i in range(len(score_history))]
        #plot_learning_curve(x, score_history, figure_file)
    logger.log("Ended the big loop\n")

# args = ["mpirun", "-np", "4", "python", "main.py", "0", "0", "model_file_name.osim"]
restore = int(sys.argv[1])

load_iters = int(sys.argv[2])
model_file = sys.argv[3]

if load_iters == 1:
    with open(model_file[10:-5] + '/iterations.txt', 'r') as f:
        lines = f.read().splitlines()
        # Get the last line as the last stored iteration
        #MAYBE GET THE SECOND LAST BECAUSE IT MESSES UP WHEN LOADING THE LAST 1 or 2
        last_iter = int(lines[-1])
        load_iters = last_iter


train(5000000, 999, model_file, save_model_with_prefix=True, restore_model_from_file=restore, save_after=1,
      load_after_iters=load_iters, viz=False, stochastic=True)
