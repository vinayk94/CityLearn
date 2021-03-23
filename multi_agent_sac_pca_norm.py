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

        # print(num_inputs)
        # sys.exit()

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
        # print('input to netwrork : ',state.shape)
        # sys.exit()
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


class SAC_multi_agent:
    def __init__(self, env,building_ids, buildings_states_actions, building_info, observation_spaces=None,
                 action_spaces=None, hidden_dim=[400, 300], discount=0.99, tau=5e-3, lr=3e-4, batch_size=100,
                 replay_buffer_capacity=1e5, start_training=None, exploration_period=None, safe_exploration=False,
                 action_scaling_coef=1., reward_scaling=1., update_per_step=1, seed=0, pca_compression=1.,
                 add_previous_consumption=False):

        with open(buildings_states_actions) as json_file:
            self.buildings_states_actions = json.load(json_file)

        self.env = env
        
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
        # self.iterations_as = iterations_as
        self.safe_exploration = safe_exploration
        self.exploration_period = exploration_period
        self.add_previous_consumption = add_previous_consumption

        self.time_step = 0
        self.pca_flag = {uid: 0 for uid in building_ids}

        self.action_spaces = {uid: a_space for uid, a_space in zip(building_ids, action_spaces)}
        self.observation_spaces = {uid: o_space for uid, o_space in zip(building_ids, observation_spaces)}

        # Optimizers/Loss using the Huber loss
        self.soft_q_criterion = nn.SmoothL1Loss()

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critic1_loss_, self.critic2_loss_, self.actor_loss_, self.alpha_loss_, self.alpha_, self.q_tracker = {}, {}, {}, {}, {}, {}

        self.replay_buffer, self.reg_buffer, self.soft_q_net1, self.soft_q_net2, self.target_soft_q_net1, self.target_soft_q_net2, self.policy_net, self.soft_q_optimizer1, self.soft_q_optimizer2, self.policy_optimizer, self.target_entropy, self.alpha, self.log_alpha, self.alpha_optimizer, self.pca, self.norm_mean, self.norm_std, self.r_norm_mean, self.r_norm_std, self.log_pi_tracker = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

        for uid in building_ids:
            self.critic1_loss_[uid], self.critic2_loss_[uid], self.actor_loss_[uid], self.alpha_loss_[uid], self.alpha_[
                uid], self.q_tracker[uid], self.log_pi_tracker[uid] = [], [], [], [], [], [], []

            #state_dim = int((pca_compression) * (1 + len(self.observation_spaces[uid].low)))
            state_dim = int((pca_compression) * (len(self.observation_spaces[uid].low)))
            action_dim = self.action_spaces[uid].shape[0]
            self.alpha[uid] = 0.2

            self.pca[uid] = PCA(n_components=state_dim)

            self.replay_buffer[uid] = ReplayBuffer(int(replay_buffer_capacity))

            # init networks
            self.soft_q_net1[uid] = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.soft_q_net2[uid] = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)

            self.target_soft_q_net1[uid] = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.target_soft_q_net2[uid] = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)

            for target_param, param in zip(self.target_soft_q_net1[uid].parameters(),
                                           self.soft_q_net1[uid].parameters()):
                target_param.data.copy_(param.data)

            for target_param, param in zip(self.target_soft_q_net2[uid].parameters(),
                                           self.soft_q_net2[uid].parameters()):
                target_param.data.copy_(param.data)

            # Policy
            self.policy_net[uid] = PolicyNetwork(state_dim, action_dim, self.action_spaces[uid],self.action_scaling_coef, hidden_dim).to(self.device)
            self.soft_q_optimizer1[uid] = optim.Adam(self.soft_q_net1[uid].parameters(), lr=lr)
            self.soft_q_optimizer2[uid] = optim.Adam(self.soft_q_net2[uid].parameters(), lr=lr)
            self.policy_optimizer[uid] = optim.Adam(self.policy_net[uid].parameters(), lr=lr)
            self.target_entropy[uid] = -np.prod(self.action_spaces[uid].shape).item()
            self.log_alpha[uid] = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer[uid] = optim.Adam([self.log_alpha[uid]], lr=lr)

    def select_action(self, states, deterministic=False):
        self.time_step += 1
        explore = self.time_step <= self.exploration_period
        action_order = np.array(range(len(self.building_ids)))
        _building_ids = [self.building_ids[i] for i in action_order]
        _states = [states[i] for i in action_order]
        actions = [None for _ in range(len(self.building_ids))]
        k = 0
        if explore:
            for uid, state in zip(_building_ids, _states):
                if self.safe_exploration:
                    hour_day = state[2]
                    a_dim = len(self.action_spaces[uid].sample())

                    # Daytime: release stored energy
                    act = [0.0 for _ in range(a_dim)]
                    if hour_day >= 9 and hour_day <= 21:
                        act = [-0.08 for _ in range(a_dim)]

                    # Early nightime: store DHW and/or cooling energy
                    if (hour_day >= 1 and hour_day <= 8) or (hour_day >= 22 and hour_day <= 24):
                        act = [0.091 for _ in range(a_dim)]

                else:
                    act = self.action_scaling_coef * self.action_spaces[uid].sample()

                actions[action_order[k]] = act
                k += 1 #k is the building count number


        else:
            for uid, state in zip(_building_ids, _states):
                state_ =  state
                
                #if self.add_dist_cons:
                 #   state_ = np.hstack((state_,

                state_ = (state_ - self.norm_mean[uid]) / self.norm_std[uid]
                state_ = self.pca[uid].transform(state_.reshape(1, -1))[0]
                state_ = torch.FloatTensor(state_).unsqueeze(0).to(self.device)

                if deterministic is False:
                    act, _, _ = self.policy_net[uid].sample(state_)
                else:
                    _, _, act = self.policy_net[uid].sample(state_)

                actions[action_order[k]] = act.detach().cpu().numpy()[0]
                k += 1

        return actions

    def add_to_buffer(self, states, actions, rewards, next_states, done):

        if self.time_step >= self.start_training and self.batch_size <= len(self.replay_buffer[self.building_ids[0]]):
            for uid in self.building_ids:
                # This code only runs once. Once the random exploration phase is over, we normalize all the states and rewards to make them have mean=0 and std=1, and apply PCA. We push the normalized compressed values back into the buffer, replacing the old buffer.
                if self.pca_flag[uid] == 0:
                    X = np.array([j[0] for j in self.replay_buffer[uid].buffer])
                    self.norm_mean[uid] = np.mean(X, axis=0)
                    self.norm_std[uid] = np.std(X, axis=0) + 1e-5
                    X = (X - self.norm_mean[uid]) / self.norm_std[uid]

                    R = np.array([j[2] for j in self.replay_buffer[uid].buffer])
                    self.r_norm_mean[uid] = np.mean(R)
                    self.r_norm_std[uid] = np.std(R) / self.reward_scaling + 1e-5

                    self.pca[uid].fit(X)
                    new_buffer = []
                    for s, a, r, s2, dones in self.replay_buffer[uid].buffer:
                        s_buffer = np.hstack(
                            self.pca[uid].transform(((s - self.norm_mean[uid]) / self.norm_std[uid]).reshape(1, -1))[0])
                        s2_buffer = np.hstack(
                            self.pca[uid].transform(((s2 - self.norm_mean[uid]) / self.norm_std[uid]).reshape(1, -1))[
                                0])
                        new_buffer.append(
                            (s_buffer, a, (r - self.r_norm_mean[uid]) / self.r_norm_std[uid], s2_buffer, dones))

                    self.replay_buffer[uid].buffer = new_buffer
                    self.pca_flag[uid] = 1

        for (uid, o, a, r, o2) in zip(self.building_ids, states, actions, rewards,next_states):

            if self.pca_flag[uid] == 1:
                o = (o - self.norm_mean[uid]) / self.norm_std[uid]
                o = self.pca[uid].transform(o.reshape(1, -1))[0]
                o2 = (o2 - self.norm_mean[uid]) / self.norm_std[uid]
                o2 = self.pca[uid].transform(o2.reshape(1, -1))[0]
                r = (r - self.r_norm_mean[uid]) / self.r_norm_std[uid]

            self.replay_buffer[uid].push(o, a, r, o2, done)


    def update(self):

        for uid in self.building_ids:
            state, action, reward, next_state, done = self.replay_buffer[uid].sample(self.batch_size)
            #print(uid)
            if self.device.type == "cuda":
                state = torch.cuda.FloatTensor(state).to(self.device)
                next_state = torch.cuda.FloatTensor(next_state).to(self.device)
                action = torch.cuda.FloatTensor(action).to(self.device)
                reward = torch.cuda.FloatTensor(reward).unsqueeze(1).to(self.device)
                done = torch.cuda.FloatTensor(done).unsqueeze(1).to(self.device)
            else:
                state = torch.FloatTensor(state).to(self.device)
                next_state = torch.FloatTensor(next_state).to(self.device)
                action = torch.FloatTensor(action).to(self.device)
                reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
                done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

            with torch.no_grad():
                # Update Q-values. First, sample an action from the Gaussian policy/distribution for the current (next) state and its associated log probability of occurrence.
                new_next_actions, new_log_pi, _ = self.policy_net[uid].sample(next_state)

                # The updated Q-value is found by subtracting the logprob of the sampled action (proportional to the entropy) to the Q-values estimated by the target networks.
                target_q_values = torch.min(
                    self.target_soft_q_net1[uid](next_state, new_next_actions),
                    self.target_soft_q_net2[uid](next_state, new_next_actions),
                ) - self.alpha[uid] * new_log_pi

                q_target = reward + (1 - done) * self.discount * target_q_values
                self.q_tracker[uid].append(q_target.mean())

            # Update Soft Q-Networks
            q1_pred = self.soft_q_net1[uid](state, action)
            q2_pred = self.soft_q_net2[uid](state, action)

            q1_loss = self.soft_q_criterion(q1_pred, q_target)
            q2_loss = self.soft_q_criterion(q2_pred, q_target)

            self.soft_q_optimizer1[uid].zero_grad()
            q1_loss.backward()
            self.soft_q_optimizer1[uid].step()

            self.soft_q_optimizer2[uid].zero_grad()
            q2_loss.backward()
            self.soft_q_optimizer2[uid].step()

            # Update Policy
            new_actions, log_pi, _ = self.policy_net[uid].sample(state)

            q_new_actions = torch.min(
                self.soft_q_net1[uid](state, new_actions),
                self.soft_q_net2[uid](state, new_actions)
            )

            policy_loss = (self.alpha[uid] * log_pi - q_new_actions).mean()

            self.policy_optimizer[uid].zero_grad()
            policy_loss.backward()
            
            self.policy_optimizer[uid].step()
            
            

            # Optimize the temperature parameter alpha, used for exploration through entropy maximization
            # alpha_loss = -(self.log_alpha[uid] * (log_pi + self.target_entropy[uid]).detach()).mean()
            # self.alpha_optimizer[uid].zero_grad()
            # alpha_loss.backward()
            # self.alpha_optimizer[uid].step()
            self.alpha[uid] = 0.2  # self.log_alpha[uid].exp()

            self.critic1_loss_[uid].append(q1_loss.item())
            self.critic2_loss_[uid].append(q2_loss.item())
            self.actor_loss_[uid].append(policy_loss.item())
            #self.alpha_loss_[uid].append(alpha_loss.item())
            #self.alpha_[uid].append(self.alpha[uid].item())
            #self.log_pi_tracker[uid].append(log_pi.mean())

            # Soft Updates
            for target_param, param in zip(self.target_soft_q_net1[uid].parameters(),
                                           self.soft_q_net1[uid].parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )

            for target_param, param in zip(self.target_soft_q_net2[uid].parameters(),
                                           self.soft_q_net2[uid].parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )
        #print(uid)    
        return self.critic1_loss_[uid] , self.critic2_loss_[uid] , self.actor_loss_[uid]
    
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
        for uid in self.building_ids:
            torch.save(self.policy_net[uid].state_dict(), actor_path+str(uid[-1]))
            torch.save(self.soft_q_net1[uid].state_dict(), critic1_path+str(uid[-1]))
            torch.save(self.soft_q_net2[uid].state_dict(), critic2_path+str(uid[-1]))
        
       # Load model parameters
    def load_model(self, actor_path, critic1_path,critic2_path):
        print('Loading models from {} and {}'.format(actor_path, critic1_path,critic2_path))
        
        for uid in self.building_ids:
        
            if actor_path is not None:
                self.policy_net[uid].load_state_dict(torch.load(actor_path+str(uid[-1])))
            if critic1_path is not None:
                self.soft_q_net1[uid].load_state_dict(torch.load(critic1_path+str(uid[-1])))
            if critic2_path is not None:
                self.soft_q_net2[uid].load_state_dict(torch.load(critic2_path+str(uid[-1])))


