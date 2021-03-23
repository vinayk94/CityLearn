""" 
Implementation of a centralized SAC agent.

core and buffer part of neural networks are adopted from spinninup repository.
agent is a centralized SAC algorithm from the same repository and is slightly customized with additional functionalities.
source: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac

@author: Vinay Kanth Reddy Bheemreddy(vinay.bheemreddy@rwth-aachen.de)

"""


import torch
import random
from copy import deepcopy
from torch.optim import Adam
import core as core
from typing import Optional, Any, Tuple, Union, Dict,List
#from optuna.samplers import TPESampler 
import pickle
import os
#from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import StepLR
import sys
from torch.utils.tensorboard import SummaryWriter
import sac_old.core as core
from utils import ReplayBuffer
from pathlib import Path
import itertools
import time
from utils import ReplayBuffer, set_seeds
import numpy as np


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)
def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def set_seeds(seed: int, env = None) -> None:
    """
    Sets seeds for reproducibility
    :param seed: Seed Value
    :param env: Optionally pass gym environment to set its seed
    :type seed: int
    :type env: Gym Environment
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)
        

# Increase state space by previous 2 step actions,energy consumption signal and SOC's.         
        
class State_Memory:
    def __init__(self, size):
        self.buffer = [0] * capacity

    def push(self, last_step_sample):
        self.buffer.append(last_step_sample)

    def __len__(self):
        return len(self.buffer)
    



class SAC:

    def __init__(self,env,test_env,actor_critic=core.MLPActorCritic,ac_kwargs=dict(), seed=0, 
         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
         polyak=0.995,entropy_tuning: bool = False, lr=1e-3,alpha=0.2, batch_size=100, start_steps=10000, 
         update_after=1000, update_every=50, act_noise=0.01, 
         max_ep_len=1000,device='cpu',num_test_episodes=1,save_freq=2, log_mode: List[str] = ["stdout"],
         log_key: str = "timestep",save_model: str = "checkpoints", checkpoint_path: str = None,log_interval: int = 10,load_model=False,
                dir_prefix:str =None):

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed =seed
        self.env = env
        self.test_env = test_env
        self.obs_dim = env.observation_space.shape
        self.act_dim = env.action_space.shape[0]
        self.act_limit = env.action_space.high[0]
        self.replay_size = replay_size
        self.batch_size =batch_size
        #self.noise_scale = act_noise
        
        self.load_model = load_model
        self.log_key = log_key
        #self.logdir = logdir
        self.save_model = save_model
        self.checkpoint_path = checkpoint_path
        #self.log_interval = log_interval
        #self.logger = Logger(logdir=logdir, formats=[*log_mode])
        

        #self.pi_lr = pi_lr
        #self.q_lr = q_lr
        self.lr =lr
        self.ac_kwargs =ac_kwargs

        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.max_ep_len= max_ep_len
        
        self.gamma =gamma
        self.polyak=polyak
        self.alpha = alpha
        self.entropy_tuning = entropy_tuning

        self.start_steps= start_steps
        self.update_after= update_after
        self.update_every= update_every
        self.save_freq = save_freq

        self.action_time_step=0 #no. of updates
        self.current_timestep= 0
        self.current_epoch=0
        self.dir_prefix= dir_prefix
        
        # Store the weights and scores in a new directory
        self.directory = "logs/sac_single_Agent_{}{}/".format(self.dir_prefix,time.strftime("%Y%m%d-%H%M%S")) # appends the timedate
        os.makedirs(self.directory, exist_ok=True)
        self.model_dir = os.path.join(self.directory,'model_param/')
        os.makedirs(self.model_dir)

        # Tensorboard writer object
        self.writer = SummaryWriter(log_dir=self.directory+'tensorboard/')
        print("Logging to {}\n".format(self.directory+'tensorboard/'))
        

        #self.test_env = env
        self.num_test_episodes = num_test_episodes  

        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(self.device)
        self.ac_targ = deepcopy(self.ac).to(self.device)
        #actually no need of saving the policy parameters as target above, since we do not need any target Actor in SAC.
        
        if self.load_model:
            if os.path.exists(self.checkpoint_path):
                self.ac.load_state_dict(torch.load(os.path.abspath(self.checkpoint_path))) 
                self.ac_targ = deepcopy(self.ac).to(self.device)
            

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.pi_scheduler= StepLR(self.pi_optimizer,step_size=1, gamma=0.96)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        self.q_scheduler= StepLR(self.q_optimizer,step_size=1, gamma=0.96)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)


        # from https://github.com/SforAiDl/genrl/blob/master/genrl/deep/agents/sac/sac.py
        if self.entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.lr)
            
        #else:
        #    self.alpha=self.alpha

        # no need of action scales setting
        # action_limit is directly obtained within the MLPActorCritic class 
        # action_bias is not need as for the city learn environment, actions are bounded with -1/3 to +1/3 
        # and the bias sums to 0



        # Assign device
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # Assign seed
        if seed is not None:
            set_seeds(seed, self.env)

        #initialize logs
        self.empty_logs()


        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts  = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        print(var_counts)
        self.logs["var_counts"]= var_counts  
        print(colorize('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts, 'green', bold=True))
        self.writer.add_scalar('Number of parameters/pi',var_counts[0])
        self.writer.add_scalar('Number of parameters/q1',var_counts[1])
        self.writer.add_scalar('Number of parameters/q2',var_counts[2])
        #print(colorize(msg, color, bold=True))
                  

    def load_weights(self, weights) -> None:
        """
        Load weights for the agent from pretrained model
        """
        self.q1.load_state_dict(weights["q1_weights"])
        self.q2.load_state_dict(weights["q2_weights"])
        self.policy.load_state_dict(weights["policy_weights"])        
        
    def empty_logs(self):
        """
        Empties logs
        """
        self.logs = {}
        self.logs["q1_loss"] = []
        self.logs["q2_loss"] = []
        self.logs["policy_loss"] = []
        self.logs["alpha_loss"] = []        
        self.logs["var_counts"] = ()

    def safe_mean(log: List[int]):
        """
        Returns 0 if there are no elements in logs
        """
        return np.mean(log) if len(log) > 0 else 0    

    def get_logging_params(self) -> Dict[str, Any]:
        """
        :returns: Logging parameters for monitoring training
        :rtype: dict
        """
        logs = {
            "policy_loss": safe_mean(self.logs["policy_loss"]),
            "q1_loss": safe_mean(self.logs["q1_loss"]),
            "q2_loss": safe_mean(self.logs["q2_loss"]),
            "alpha_loss": safe_mean(self.logs["alpha_loss"]),
        }

        self.empty_logs()
        return logs



    # Set up function for computing SAC Q-losses
    def compute_loss_q(self,data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        
        #logging into tensorboard
        self.writer.add_scalar('loss/Critic1_loss', loss_q1,self.current_timestep)
        self.writer.add_scalar('loss/Critic2_loss', loss_q2,self.current_timestep)

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),Q2Vals=q2.detach().numpy())

        self.logs["q1_loss"].append(loss_q1.item())
        self.logs["q2_loss"].append(loss_q2.item())

        return loss_q,q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self,data):
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        # alpha loss
        alpha_loss = torch.tensor(0.0).to(self.device)

        if self.entropy_tuning:
            alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
            self.writer.add_scalar('loss/entropy_tuning_loss', alpha_loss,self.current_timestep)
            self.logs["alpha_loss"].append(alpha_loss.item())
        else:
            alpha_loss=0
            
        #logging into tensorboard
        self.writer.add_scalar('loss/Actor_loss', loss_pi,self.current_timestep)


        self.logs["policy_loss"].append(loss_pi.item())



        return loss_pi,alpha_loss,pi_info

    def update(self,data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False


        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, alpha_loss,pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        
        if self.entropy_tuning:
         # Next run one gradient descent step for alpha.
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()       
        
            self.writer.add_scalar('entropy_tuning_param/alpha', self.alpha,self.current_timestep)

        # Unfreeze Q-network so you can optimize it at the next SAC step.
        for p in self.q_params:
            p.requires_grad = True


        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
                
    def reset_action_tracker(self):
        self.action_tracker = []

    def reset_reward_tracker(self):
        self.reward_tracker = []
        
    def get_action(self,o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), deterministic)

    def eval_agent(self,test=True):
        if test == True:
            eval_env= self.test_env
            t_env='testing environment'
        else:
            eval_env=deepcopy(self.env)
            t_env='training environment'
        ep_rews =[]
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = eval_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                # o = (o-self.replay_buffer.obs_buf_min) /(self.replay_buffer.obs_buf_max - self.replay_buffer.obs_buf_min)
                nom = o-self.replay_buffer.obs_buf_min
                denom = self.replay_buffer.obs_buf_max - self.replay_buffer.obs_buf_min
                denom[denom==0] = 1
                o = nom/denom
                o, r, d, _ = eval_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            ep_rews.append(ep_ret)

        print("Evaluating on the {} for {} episode, Mean Reward: {}".format(t_env,self.num_test_episodes, np.mean(ep_rews)))
        #print('Final cost',eval_env.cost())
        
        self.writer.add_scalar("Scores/ramping", eval_env.cost()['ramping'], self.current_epoch)
        self.writer.add_scalar("Scores/1-load_factor", eval_env.cost()['1-load_factor'], self.current_epoch)
        self.writer.add_scalar("Scores/average_daily_peak", eval_env.cost()['average_daily_peak'], self.current_epoch)
        self.writer.add_scalar("Scores/peak_demand", eval_env.cost()['peak_demand'], self.current_epoch)
        self.writer.add_scalar("Scores/net_electricity_consumption", eval_env.cost()['net_electricity_consumption'], self.current_epoch)
        self.writer.add_scalar("Scores/total", eval_env.cost()['total'], self.current_epoch)
        self.writer.add_scalar("Scores/test_episode_reward",np.mean(ep_rews),self.current_epoch)
        
        return np.mean(ep_rews), eval_env.cost()['total']


    def learn(self) -> None:

        ep_num = 0
        best_score=1.5
        return_per_episode = []

        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        epoch_start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        #self.current_epoch=1    
        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):

            self.current_timestep = t #for logging
            
            # if t > 8759 update minmax of buffer and use it to normalize
            # so,we collect data for 1 year and calculate min-max of obs and rewards
            
            if t == self.start_steps:
                self.replay_buffer.collect_minmax()


            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > self.start_steps:
                #print(t)
                a = self.get_action(o)
            else:
                a = self.env.action_space.sample()
            
            # Step the env
            o2, r, d, _ = self.env.step(a)
            self.writer.add_scalar('Rewards/single_Agent_reward',r,self.current_timestep)
            
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==self.max_ep_len else d

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                #print('End of trajectory: Episode return is', ep_ret )
                #print('Cost function is', self.env.cost())
                ep_num +=1
                return_per_episode.append(ep_ret)
                self.writer.add_scalar('Rewards/return_per_episode',ep_ret,ep_num)
                
                o, ep_ret, ep_len = self.env.reset(), 0, 0
                  
            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
            #if t >= self.update_after: #instead of updating for some fixed steps, update for every step
                #print('updating')
                for _ in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    #print(batch)
                    #print(batch.size)
                    #sys.exit()
                    self.update(data=batch)

            #End of epoch handling
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch
                
                self.current_epoch+=1
                self.pi_scheduler.step()
                self.q_scheduler.step()
                
                print('Epoch:', epoch,'Policy_LR:', self.pi_scheduler.get_lr(), 'Critic_LR:', self.q_scheduler.get_lr())

                print('time step: {} , epoch: {} ,time elapsed: {} '.format(t+1,epoch,time.time()-epoch_start_time))
                train_mean_return, test_score = self.eval_agent(test=False)
                
                #test_mean_return=self.eval_agent(test=True)
                #print('time_per_epoch',time.time()-epoch_start_time)  
                epoch_start_time=time.time()
                print('\n')
                
                
                # Save model
                if (epoch % self.save_freq == 0):
                    if test_score < best_score:
                        best_score= test_score
                        print('Better evaluation score and hence saving model to {}'.format(os.path.join(self.directory,'model_param/')))
                        torch.save(self.ac.state_dict(), os.path.join(self.directory,'model_param/')+'checkpoint.pt')

            if (t+1) % self.steps_per_epoch == 0:
                self.action_time_step =0

            else:
                self.action_time_step +=1
                
            

        return epoch, train_mean_return*(self.batch_size)       