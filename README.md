# CityLearn challenge

This repository contains the work submitted to the ([citylearn challenge 2020](https://sites.google.com/view/citylearnchallenge/previous-edition-2020?authuser=0)) and the extensions on it.

The updated description of the environment and the general utility functions can be found at the ([challenge page](https://sites.google.com/view/citylearnchallenge/environment))

#### Additions
    * A central [DDPG](https://arxiv.org/abs/1509.02971) controller which is submitted to the 2020 edition is added.
    * And the [SAC](https://arxiv.org/abs/1801.01290) extensions for the central and decentralized settings.

CityLearn is an open source OpenAI Gym environment for the implementation of Multi-Agent Reinforcement Learning (RL) for building energy coordination and demand response in cities. Its objective is to facilitiate and standardize the evaluation of RL agents such that different algorithms can be easily compared with each other.
![Demand-response](https://github.com/intelligent-environments-lab/CityLearn/blob/master/images/dr.jpg)
## Description
Districts and cities have periods of high demand for electricity, which raise electricity prices and the overall cost of the power distribution networks. Flattening, smoothening, and reducing the overall curve of electrical demand helps reduce operational and capital costs of electricity generation, transmission, and distribution networks. Demand response is the coordination of electricity consuming agents (i.e. buildings) in order to reshape the overall curve of electrical demand.
CityLearn allows the easy implementation of reinforcement learning agents in a multi-agent setting to reshape their aggregated curve of electrical demand by controlling the storage of energy by every agent. Currently, CityLearn allows controlling the storage of domestic hot water (DHW), and chilled water (for sensible cooling and dehumidification). CityLearn also includes models of air-to-water heat pumps, electric heaters, solar photovoltaic arrays, and the pre-computed energy loads of the buildings, which include space cooling, dehumidification, appliances, DHW, and solar generation.

## Requirements
CityLearn requires the installation of the following Python libraries:
- Pandas 0.24.2 or older
- Numpy 1.16.4 or older
- Gym 0.14.0
- Json 2.0.9

In order to run the main files with the sample agent provided you will need:
- PyTorch 1.1.0


## Files

- [buildings_state_action_space.json](/buildings_state_action_space.json): json file containing the possible states and actions for every building, from which users can choose.
- [building_attributes.json](/data/building_attributes.json): json file containing the attributes of the buildings and which users can modify.
- [citylearn.py](/citylearn.py): Contains the ```CityLearn``` environment and the functions ```building_loader()``` and ```autosize()```
- [energy_models.py](/energy_models.py): Contains the classes ```Building```, ```HeatPump``` and ```EnergyStorage```, which are called by the ```CityLearn``` class.
- [reward_function.py](/reward_function.py): Contains the reward functions that wrap and modifiy the rewards obtained from ```CityLearn```. This function can be modified by the user in order to minimize the cost function of ```CityLearn```. There are two reward functions, one works for multi-agent systems (decentralized RL agents), and the other works for single-agent systems (centralized RL agent). Setting the attribute central_agent=True in CityLearn will make the environment return the output from sa_reward_function, while central_agent=False (default mode) will make the environment return the output from ma_reward_function.
- [example_rbc.ipynb](/example_rbc.ipynb): jupyter lab file. Example of the implementation of a manually optimized Rule-based controller (RBC) that can be used for comparison
- [example_central_agent.ipynb](/example_central_agent.ipynb): jupyter lab file. Example of the implementation of a SAC centralized RL algorithm from Open AI stable baselines, for 1 and 9 buildings.

