import os
import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam
import numpy as np
import random
import copy
import gym
import json
from sklearn.decomposition import PCA
# torch.autograd.set_detect_anomaly(True)
from torch.optim.lr_scheduler import StepLR
import time
import math
import sys

#if torch.cuda.is_available():
#    torch.cuda.set_device(2)

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class RBC_Agent:
    def __init__(self, actions_spaces):
        self.actions_spaces = actions_spaces
        self.reset_action_tracker()

    def reset_action_tracker(self):
        self.action_tracker = []

    def select_action(self, states):
        hour_day = states[2][2]

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


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, action_space, action_scaling_coef, hidden_dim=[400, 300],
                 init_w=3e-3, log_std_min=-20, log_std_max=2, epsilon=1e-6):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon
        
        #print(num_inputs)
        #sys.exit()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])

        self.mean_linear = nn.Linear(hidden_dim[1], num_actions)
        self.log_std_linear = nn.Linear(hidden_dim[1], num_actions)

        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_scale = torch.FloatTensor(
            action_scaling_coef * (action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            action_scaling_coef * (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        #print('input to netwrork : ',state.shape)
        #sys.exit()
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(PolicyNetwork, self).to(device)


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=[400, 300], init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], 1)
        self.ln1 = nn.LayerNorm(hidden_size[0])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.ln1(F.relu(self.linear1(x)))
        x = self.ln2(F.relu(self.linear2(x)))
        x = self.linear3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class Net_consumption_Buffer:
    def __init__(self,size, dim):
        self.buffer = np.zeros((size,dim),dtype=np.float32)
        # note that the size refers to the no.of previous time steps stored.
        # dimensions refers to the number of features of the states that are stored.
        self.max_size=size
        self.dim = dim
        self.size = 0
        self.ptr= 0

    def push(self, last_step_sample):
        #need to be improved as the no. of dimensions stored increases.
        # as of now, it works for single dimension and single storage.
        self.buffer[self.ptr] = last_step_sample
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def __len__(self):
        return len(self.buffer)


class SAC_single_agent:
    def __init__(self, building_ids, buildings_states_actions, building_info, observation_spaces=None,
                 action_spaces=None, hidden_dim=[400, 300], discount=0.99, tau=5e-3, lr=3e-4, batch_size=100,
                 replay_buffer_capacity=1e5, start_training = None,exploration_period = None,safe_exploration= False,
                 action_scaling_coef=1., reward_scaling=1., update_per_step=1,seed=0,pca_compression = 1.,add_previous_consumption=False):
        with open(buildings_states_actions) as json_file:
            self.buildings_states_actions = json.load(json_file)

        self.building_ids = building_ids
        self.start_training = start_training
        self.discount = discount
        self.batch_size = batch_size
        self.tau = tau
        self.action_scaling_coef = action_scaling_coef
        self.reward_scaling = reward_scaling

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.deterministic = False

        self.update_per_step = update_per_step
        #self.iterations_as = iterations_as
        self.safe_exploration = safe_exploration
        self.exploration_period = exploration_period
        self.add_previous_consumption = add_previous_consumption

        self.action_list_ = []
        self.action_list2_ = []

        self.time_step = 0
        self.pca_flag = 0
        self.norm_flag = 0

        self.action_spaces = action_spaces
        self.observation_spaces =  observation_spaces
        print(self.observation_spaces)

        # Optimizers/Loss using the Huber loss
        self.soft_q_criterion = nn.SmoothL1Loss()

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critic1_loss_, self.critic2_loss_, self.actor_loss_, self.alpha_loss_, self.alpha_, self.q_tracker = {}, {}, {}, {}, {}, {}




        self.critic1_loss_, self.critic2_loss_, self.actor_loss_, self.alpha_loss_, self.alpha_, self.q_tracker, self.log_pi_tracker = [], [], [], [], [], [], []

        self.state_memory = Net_consumption_Buffer(2, 1)
        #state_dim = len(self.observation_spaces.low) + 1  # +1 for previous step consumption
        #state_dim = len(self.observation_spaces[0].low) + 1
        
        state_dim = len(self.observation_spaces.low)
        #print(state_dim)
        
        if self.add_previous_consumption:
            state_dim = int((pca_compression) *  (1+state_dim))
        else:
            state_dim = int((pca_compression) * state_dim)
        
        #print(state_dim)
        #sys.exit()
        action_dim = self.action_spaces.shape[0]
        #action_dim = self.action_spaces[0].shape[0]  
        self.alpha = 0.2
        self.pca = PCA(n_components=state_dim)
        self.replay_buffer = ReplayBuffer(int(replay_buffer_capacity))

        # init networks
        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        for target_param, param in zip(self.target_soft_q_net1.parameters(),
                                       self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_soft_q_net2.parameters(),
                                       self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        # Policy
        self.policy_net = PolicyNetwork(state_dim, action_dim, self.action_spaces, self.action_scaling_coef,
                                             hidden_dim).to(self.device)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=lr)
        self.soft_q1_scheduler= StepLR(self.soft_q_optimizer1,step_size=1, gamma=0.98)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=lr)
        self.soft_q2_scheduler= StepLR(self.soft_q_optimizer2,step_size=1, gamma=0.98)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.soft_pi_scheduler= StepLR(self.policy_optimizer,step_size=1, gamma=0.98)
        self.target_entropy = -np.prod(self.action_spaces.shape).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, states, deterministic=False):

        self.time_step += 1
        explore = self.time_step <= self.exploration_period

        k = 0
        if explore:

            if self.safe_exploration:
                hour_day = states[2]
                a_dim = len(self.action_spaces.sample())

                # Daytime: release stored energy
                act = [0.0 for _ in range(a_dim)]
                if hour_day >= 9 and hour_day <= 21:
                    act = [-0.08 for _ in range(a_dim)]

                # Early nightime: store DHW and/or cooling energy
                if (hour_day >= 1 and hour_day <= 8) or (hour_day >= 22 and hour_day <= 24):
                    act = [0.091 for _ in range(a_dim)]

            else:
                act = self.action_scaling_coef * self.action_spaces.sample()

            k += 1
        else:
            
            state_ = states
            # Adding previous consumption information information to the state
            if self.add_previous_consumption:
                state_ = np.hstack((states, self.state_memory.buffer[-1]))
            #ok the thing that you are doing below is good, maintaining a running mean and std.
            state_ = (state_ - self.norm_mean) / self.norm_std
            state_ = self.pca.transform(state_.reshape(1, -1))[0]
            state_ = torch.FloatTensor(state_).unsqueeze(0).to(self.device)

            if deterministic is False:
                act, _, _ = self.policy_net.sample(state_)
            else:
                _, _, act = self.policy_net.sample(state_)

            act = act.detach().cpu().numpy()[0]
            k += 1

        return act

    def add_to_buffer(self, states, actions, rewards, next_states, done):

        #first you need to collect the reward and push it to the consumption memory for that agent.
        #next take the previous step consumption from memory and concatenate it to the current state.
        # and take the current reward/consumption and concatenate it to the the next state.

        # if you have collected enough samples for the buffer, then calculate statistics and normalize them
        #normalize when you are about to start training the networks, until then add the previous electricity consumption and push it to buffer
        if self.time_step >= self.start_training and self.batch_size <= len(self.replay_buffer):



                # This code only runs once. Once the random exploration phase is over, we normalize all the states and rewards to make them have mean=0 and std=1, and apply PCA. We push the normalized compressed values back into the buffer, replacing the old buffer.
            
            if self.pca_flag == 0:
            #if self.norm_flag == 0:
                X = np.array([j[0] for j in self.replay_buffer.buffer])
                self.norm_mean = np.mean(X, axis=0)
                self.norm_std = np.std(X, axis=0) + 1e-5
                X = (X - self.norm_mean) / self.norm_std

                R = np.array([j[2] for j in self.replay_buffer.buffer])
                self.r_norm_mean = np.mean(R)
                self.r_norm_std = np.std(R) / self.reward_scaling + 1e-5

                self.pca.fit(X)
                new_buffer = []
                for s, a, r, s2, dones in self.replay_buffer.buffer:
                    s_buffer = np.hstack(self.pca.transform(((s - self.norm_mean)/self.norm_std).reshape(1,-1))[0])
                    s2_buffer = np.hstack(self.pca.transform(((s2 - self.norm_mean)/self.norm_std).reshape(1,-1))[0])
                    new_buffer.append((s_buffer, a, (r - self.r_norm_mean) / self.r_norm_std, s2_buffer, dones))
                    
                #print(new_buffer[0][0])
                #print(new_buffer[0][2])
                #sys.exit()

                self.replay_buffer.buffer = new_buffer
                self.pca_flag = 1
                self.norm_flag = 1

            

        # if you are just collecting the samples, push the samples normally to respective buffers and normalize them on the go.
        # you need to concatenate here once also, the current consumption to the next state and the previous step consumption to the current state.
        # mind that you maintain raw electricity consumption in the state buffer but normalized ones in the main buffer.



        # Push inputs and targets to the regression buffer. The targets are the net electricity consumption.
        #self.state_memory.push(rewards)

        #print(self.state_memory.buffer[0])
        #print(np.hstack((states, self.state_memory.buffer[0])))
        #print(np.hstack((states, self.state_memory.buffer[0])).shape)

        #sys.exit()
        if self.add_previous_consumption:
            states = np.hstack((states, self.state_memory.buffer[0]))
            next_states = np.hstack((next_states, self.state_memory.buffer[-1]))
        #if self.norm_flag == 1 :
        if self.pca_flag == 1:
            states = (states - self.norm_mean) / self.norm_std
            states = self.pca.transform(states.reshape(1, -1))[0]
            next_states = (next_states - self.norm_mean) / self.norm_std
            next_states = self.pca.transform(next_states.reshape(1, -1))[0]
            #print('next_states shape is ',next_states.shape)
            #sys.exit()
            rewards = (rewards - self.r_norm_mean) / self.r_norm_std

           
        self.replay_buffer.push(states, actions, rewards, next_states, done)

    def update(self):
        # for _ in range(1 + max(0, self.time_step - 8760)//5000):
        #if self.time_step >= self.start_training and self.batch_size <= len(self.replay_buffer):
            #for _ in range(self.update_per_step):

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        #print('size of state is ', state.shape)
        #print('size of reward is ', reward.shape)
        

        if self.device.type == "cuda":
            state = torch.cuda.FloatTensor(state).to(self.device)
            next_state = torch.cuda.FloatTensor(next_state).to(self.device)
            action = torch.cuda.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            #reward = torch.FloatTensor(reward).to(self.device)
            done = torch.cuda.FloatTensor(done).unsqueeze(1).to(self.device)
        else:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            #reward = torch.FloatTensor(reward).to(self.device)
            done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            # Update Q-values. First, sample an action from the Gaussian policy/distribution for the current (next) state and its associated log probability of occurrence.
            new_next_actions, new_log_pi, _ = self.policy_net.sample(next_state)

            # The updated Q-value is found by subtracting the logprob of the sampled action (proportional to the entropy) to the Q-values estimated by the target networks.
            target_q_values = torch.min(
                self.target_soft_q_net1(next_state, new_next_actions),
                self.target_soft_q_net2(next_state, new_next_actions),
            ) - self.alpha * new_log_pi
            #print('target q value shape is',target_q_values.shape)
            #print('reward shape is',reward.shape)
            

            q_target = reward + (1 - done) * self.discount * target_q_values
            self.q_tracker.append(q_target.mean())

        # Update Soft Q-Networks
        q1_pred = self.soft_q_net1(state, action)
        q2_pred = self.soft_q_net2(state, action)

        q1_loss = self.soft_q_criterion(q1_pred, q_target)
        q2_loss = self.soft_q_criterion(q2_pred, q_target)

        self.soft_q_optimizer1.zero_grad()
        q1_loss.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q2_loss.backward()
        self.soft_q_optimizer2.step()

        # Update Policy
        new_actions, log_pi, _ = self.policy_net.sample(state)

        q_new_actions = torch.min(
            self.soft_q_net1(state, new_actions),
            self.soft_q_net2(state, new_actions)
        )

        policy_loss = (self.alpha * log_pi - q_new_actions).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Optimize the temperature parameter alpha, used for exploration through entropy maximization
        #alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        #self.alpha_optimizer.zero_grad()
        #alpha_loss.backward()
        #self.alpha_optimizer.step()
        self.alpha = 0.2  # self.log_alpha[uid].exp()
        #self.alpha =self.log_alpha.exp()    
        self.critic1_loss_.append(q1_loss.item())
        self.critic2_loss_.append(q2_loss.item())
        self.actor_loss_.append(policy_loss.item())
        #self.alpha_loss_.append(alpha_loss.item())
        #self.alpha_.append(self.alpha.item())
        self.log_pi_tracker.append(log_pi.mean())

        # Soft Updates
        for target_param, param in zip(self.target_soft_q_net1.parameters(),
                                       self.soft_q_net1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(self.target_soft_q_net2.parameters(),
                                       self.soft_q_net2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        #return  q1_loss.item(),q2_loss.item(),policy_loss.item(),alpha_loss.item(),self.alpha.item()
        return  q1_loss.item(),q2_loss.item(),policy_loss.item()

            
                    
    def save_model(self,path):
        """Saves a checkpoint of all the models
        """
        save_path =path+'/monitor/checkpoints/'
        #if not os.path.exists(path+'/monitor/checkpoints/iter_{self.time_step}'):
            #save_path = os.makedirs(path+'/monitor/checkpoints/iter_{self.time_step}')
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #print(save_path)
        actor_path = save_path+"/sac_actor"
        critic1_path = save_path+"/sac_critic1"
        critic2_path = save_path+"/sac_critic2"
        
        #print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy_net.state_dict(), actor_path)
        torch.save(self.soft_q_net1.state_dict(), critic1_path)
        torch.save(self.soft_q_net2.state_dict(), critic2_path)
        
       # Load model parameters
    def load_model(self, actor_path, critic1_path,critic2_path):
        print('Loading models from {} and {}'.format(actor_path, critic1_path,critic2_path))
        if actor_path is not None:
            self.policy_net.load_state_dict(torch.load(actor_path))
        if critic1_path is not None:
            self.soft_q_net1.load_state_dict(torch.load(critic1_path))
        if critic2_path is not None:
            self.soft_q_net2.load_state_dict(torch.load(critic2_path))
            
        
 