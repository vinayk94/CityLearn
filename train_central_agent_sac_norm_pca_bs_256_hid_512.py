from citylearn import CityLearn
import matplotlib.pyplot as plt
from pathlib import Path
from single_agent_sac_norm_pca import SAC_single_agent
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import csv
import time
import re
import pandas as pd
import torch
from joblib import dump, load
import os
import sys

import argparse

if torch.cuda.is_available():
    torch.cuda.set_device(1)
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 0)')
parser.add_argument('--n_episodes', type=int, default=70, metavar='N',
                    help='Number of episodes to train for (default: 100)')
parser.add_argument('--start_steps', type=int, default=8760 * 3 + 1, metavar='N',
                    help='Steps sampling random actions (default: 8760)')
parser.add_argument('--checkpoint_interval', type=int, default=10, metavar='N',
                    help='Saves a checkpoint with actor/critic weights every n episodes')
parser.add_argument('--log', type=bool, default=False,
                    help='log to tensorboard and csv files')
args = parser.parse_args()

# Load environment
climate_zone = 1
data_path = Path("data/Climate_Zone_" + str(climate_zone))
building_attributes = data_path / 'building_attributes.json'
weather_file = data_path / 'weather_data.csv'
solar_profile = data_path / 'solar_generation_1kW.csv'
#building_state_actions = 'buildings_state_action_space_full.json'
building_state_actions = 'buildings_state_action_space_central_full.json'
building_id = ["Building_1","Building_2","Building_3","Building_4","Building_5","Building_6","Building_7","Building_8","Building_9"]
#building_id = ["Building_1","Building_2"]
objective_function = ['ramping', '1-load_factor', 'average_daily_peak', 'peak_demand', 'net_electricity_consumption']

# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
# Can be obtained using observations_spaces[i].low or .high
env = CityLearn(data_path, building_attributes, weather_file, solar_profile, building_id,
                buildings_states_actions=building_state_actions, cost_function=objective_function, verbose=0,
                central_agent=True, simulation_period=(0, 8760 - 1))
#observations_spaces, actions_spaces = env.get_state_action_spaces()

observations_spaces = env.observation_space
actions_spaces = env.action_space

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()
#print(building_info)

# Hyperparameters
bs = 256
tau = 0.005
gamma = 0.99
lr = 0.0003
hid = [512, 512]

# Store the weights and scores in a new directory
if args.log:
    parent_dir = "central_agent/central_sac_all_build_only_std_pca_hid{}_bs_{}_{}_rew3/".format(hid[0],bs,time.strftime("%Y%m%d-%H%M%S"))  # apprends the timedate
    os.makedirs(parent_dir, exist_ok=True)

    # Create the final dir
    final_dir = parent_dir + "final/"
    os.makedirs(final_dir, exist_ok=True)
    # Tensorboard writer object
    writer = SummaryWriter(log_dir=parent_dir + 'tensorboard/')
    print("Logging to {}\n".format(parent_dir + 'tensorboard/'))

# start_training is redundant for us

# RL CONTROLLER
# Instantiating the control agent(s)
agents = SAC_single_agent(building_id, building_state_actions, building_info, observation_spaces=observations_spaces,
                          action_spaces=actions_spaces, hidden_dim=hid, start_training=8760 * 3, discount=gamma,
                          tau=tau, lr=lr, batch_size=bs,
                          replay_buffer_capacity=1e6, exploration_period=8760 * 3 + 1,
                          pca_compression = 1,action_scaling_coef=0.5, reward_scaling=5., update_per_step=1, seed=0)

# Select many episodes for training. In the final run we will set this value to 1 (the buildings run for one year)
# n_episodes = 250
total_ep = 0
updates = 0

# The list of scores and rewards
score_list = []
reward_list = []

best_reward = 1.2

k, c = 0, 0
cost, cum_reward = {}, {}

# The number of episodes can be replaces by a stopping criterion (i.e. convergence of the average reward)
#if args.log:
 #   file_name = "{}/single_pca_norm.csv".format(parent_dir)

start = time.time()
for e in range(args.n_episodes):
    cum_reward[e] = 0
    rewards = []
    episode_reward = 0

    is_evaluating = False  # Evaluate deterministic policy after 7 epochs
    rewards = []
    state = env.reset()
    done = False

    j = 0

    each_episode_time = time.time()
    while not done:
        action = agents.select_action(state, deterministic=is_evaluating)
        next_state, reward, done, _ = env.step(action)
        agents.add_to_buffer(state, action, reward, next_state, done)

        if agents.time_step >= agents.start_training and agents.batch_size <= len(agents.replay_buffer):
            # for _ in range(agents.update_per_step):

            # critic_1_loss,critic_2_loss,policy_loss,alpha_loss,alpha= agents.update()
            critic_1_loss, critic_2_loss, policy_loss = agents.update()
            if args.log:
                writer.add_scalar('loss/critic_1', critic_1_loss, total_ep)
                writer.add_scalar('loss/critic_2', critic_2_loss, total_ep)
                writer.add_scalar('loss/policy', policy_loss, total_ep)
            # writer.add_scalar('loss/entropy_loss', alpha_loss, total_ep)
            # writer.add_scalar('entropy_temprature/alpha', alpha, total_ep)

        state = next_state
        # cum_reward[e] += reward[0]
        # rewards.append(reward)
        k += 1
        episode_reward += reward

        total_ep += 1
    if args.log:
        # Tensorboard log citylearn cost function
        writer.add_scalar("Scores/ramping", env.cost()['ramping'], e)
        writer.add_scalar("Scores/1-load_factor", env.cost()['1-load_factor'], e)
        writer.add_scalar("Scores/average_daily_peak", env.cost()['average_daily_peak'], e)
        writer.add_scalar("Scores/peak_demand", env.cost()['peak_demand'], e)
        writer.add_scalar("Scores/net_electricity_consumption", env.cost()['net_electricity_consumption'], e)
        writer.add_scalar("Scores/total", env.cost()['total'], e)
        writer.add_scalar("Scores/episode_reward", episode_reward, e)

        # Append the total score/reward to the list
    score_list.append(env.cost()['total'])
    reward_list.append(episode_reward)

    # Save trained Actor and Critic network periodically as a checkpoint if it's the best model achieved
    # if e % args.checkpoint_interval == 0:
    if env.cost()['total'] < best_reward:
        best_reward = env.cost()['total']
        if args.log:
            print("Saving new best model to {}".format(parent_dir))
            agents.save_model(parent_dir)

    #else:
    #    agents.soft_q1_scheduler.step()
    #    agents.soft_q2_scheduler.step()
    #    agents.soft_pi_scheduler.step()

    print('Episode',e,'Loss -', env.cost(),'ep_reward',episode_reward, 'Simulation time (min) -', (time.time() - start) / 60.0, 'episode time(min) -',
          (time.time() - each_episode_time) / 60.0)



