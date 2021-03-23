""" 
Implementation of a centralized DDPG agent.

core and buffer part of neural networks are adopted from spinninup repository.
agent is a centralized ddpg agent from the same repository and is slightly customized with additional functionalities.
source: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg

@author: Vinay Kanth Reddy Bheemreddy(vinay.bheemreddy@rwth-aachen.de)

"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import scipy.signal
import time
import random
from copy import deepcopy
from torch.optim import Adam
#import pickle
import os
#from torch.optim.lr_scheduler import OneCycleLR


# CORE

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

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(128,128),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

# REPLAY BUFFER

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
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
         
# CENTRAL AGENT    
class RL_Agents:
    def __init__(self, env):
        
        torch.manual_seed(0)
        np.random.seed(0)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed =0
        self.env = env
        self.test_env = env #could be other climate environment
        
        # Assign seed
        random.seed(self.seed)
        if self.env is not None:
            self.env.seed(self.seed)
            
        self.obs_dim = env.observation_space.shape
        self.act_dim = env.action_space.shape[0]
        self.act_limit = env.action_space.high[0]
        self.replay_size = int(1e6)
        self.batch_size =64
        self.noise_scale = 0.1
        self.pi_lr = 0.0007740099932059868
        self.q_lr = 0.0004450729217896341
        self.ac_kwargs =dict(hidden_sizes=[96,96])
        
        self.steps_per_epoch = 8759
        self.epochs = 100
        self.max_ep_len= 8760
        
        self.gamma =0.99
        self.polyak=0.995
        
        self.start_steps= 8759
        self.update_after= 8760
        self.update_every= 40
        self.save_freq = 2
        
        self.expl_noise_init = 0.75 # Exploration noise at time-step 0
        self.expl_noise_final = 0.01 # Magnitude of the minimum exploration noise
        self.expl_noise_decay_rate = 1/(290*8760)  # Decay rate of the exploration noise in 1/h
        
        self.action_time_step=0 #no. of updates
        
        #self.test_env = env
        self.num_test_episodes = 1  
                             
        # Create actor-critic module and target networks
        self.ac = MLPActorCritic(self.env.observation_space, self.env.action_space, **self.ac_kwargs).to(self.device)
        if os.path.exists("checkpoint.pt"):
            self.ac.load_state_dict(torch.load(os.path.abspath("checkpoint.pt")))                 
        self.ac_targ = deepcopy(self.ac).to(self.device)
                             
        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)
        
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        #self.pi_scheduler= OneCycleLR(self.pi_optimizer,max_lr=0.001,epochs=self.epochs,steps_per_epoch=self.steps_per_epoch )
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=self.q_lr)
        #self.q_scheduler= OneCycleLR(self.q_optimizer,max_lr=0.001,epochs=self.epochs,steps_per_epoch=self.steps_per_epoch)
        

                             
    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self,data):
        o, a, r, o2, d = data['obs'].to(self.device), data['act'].to(self.device), data['rew'].to(self.device),data['obs2'].to(self.device), data['done'].to(self.device)

        q = self.ac.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self,data):#
        o = data['obs'].to(self.device)
        q_pi = self.ac.q(o, self.ac.pi(o))

        return -q_pi.mean()
    def add_to_buffer(self,state, action, reward, next_state, done):
        self.replay_buffer.store(state, action, reward, next_state, done)
        

    def update(self,data):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
        #self.q_scheduler.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()
        #self.pi_scheduler.step()
        

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self,o):
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device))
        upd_noise_scale = max(self.expl_noise_final, self.expl_noise_init * (1 - self.action_time_step * self.expl_noise_decay_rate))
        a += upd_noise_scale* np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)
    
    #use deterministic action while testing
    def select_action(self,o):
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device))
        return np.clip(a, -self.act_limit, self.act_limit)
    
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
                o, r, d, _ = eval_env.step(self.select_action(o))
                ep_ret += r
                ep_len += 1
            ep_rews.append(ep_ret)
            
        #print("Evaluating on {} episodes, Mean Reward: {} and Std Deviation for the reward: {}".format(
        #                    self.num_test_episodes, np.mean(ep_rews), np.std(ep_rews) ))
        print("Evaluating on the {} for {} episode, Mean Reward: {}".format(t_env,self.num_test_episodes, np.mean(ep_rews)))
        print('Final cost',eval_env.cost())
        return np.mean(ep_rews)     
    

    
    def learn(self) -> None:
        
       # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        #start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        
        epoch_start_time = time.time()
        #ep_reward = np.zeros(epochs)
        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):
            
            
            #start_time = time.time()
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy (with some noise, via act_noise). 
            if t > self.start_steps:
                a = self.get_action(o)
            else:
                a = self.env.action_space.sample()
            
            #print(a)
            # Step the env
            #if epoch>0:
                #print(a.shape)
                #print(self.env.time_step)
            
            o2, r, d, _ = self.env.step(a)
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
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
            #if t >= self.update_after: #instead of updating for some fixed steps, update for every step
                #print('updating')
                for _ in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(data=batch)
                    
            #End of epoch handling
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch
                
                print('time step: {} , epoch: {} ,time elapsed: {} '.format(t+1,epoch,time.time()-epoch_start_time))
                train_mean_return= self.eval_agent(test=False)
                test_mean_return=self.eval_agent(test=True)
                #print('time_per_epoch',time.time()-epoch_start_time)  
                epoch_start_time=time.time()
                print('\n')
                
                # Save model
                if (epoch % self.save_freq == 0):
                    torch.save(self.ac.state_dict(), os.path.join(folder,'model_param/')+'checkpoint{}.pt'.format(epoch))
                                    
                        
            if (t+1) % self.steps_per_epoch == 0:
                self.action_time_step =0
                
            else:
                self.action_time_step +=1
                
        #print(time.time()-start_time)            
        return epoch, test_mean_return*(self.batch_size)   
    
class RBC_Agent:
    def __init__(self, actions_spaces):
        self.actions_spaces = actions_spaces
        self.reset_action_tracker()

    def reset_action_tracker(self):
        self.action_tracker = []

    def select_action(self, states):
        hour_day = states[0][0]

        # Daytime: release stored energy
        a = [[0.0 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        if hour_day >= 9 and hour_day <= 21:
            a = [[-0.08 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]

        # Early nightime: store DHW and/or cooling energy
        if (hour_day >= 1 and hour_day <= 8) or (hour_day >= 22 and hour_day <= 24):
            a = []
            for i in range(len(self.actions_spaces)):
                if len(self.actions_spaces[i].sample()) == 2:
                    a.append([0.091, 0.091])
                else:
                    a.append([0.091])

        self.action_tracker.append(a)
        return np.array(a)

    
                                    
                             
                             
        
        
        
