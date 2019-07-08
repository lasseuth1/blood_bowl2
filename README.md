# Reinforcement Learning & Blood Bowl

The code provided in the repository is an attempt of applying Reinforcement Learning to Blood Bowl. 

The code is structured in a way, that runs parallel environments of Blood Bowl using the code in vec_env.py, where the trajectories collected is used for updating the models parameters after a fixed number of steps. A lot of the configurations for the code can be altered in arguments.py.

## FFAI
The folder named ffai is a copy of the repository from https://github.com/njustesen/ffai - containing all that is needed for running the FFAI Game Engine. 

## Rarity Of Events
The folder named rarity_of_events contains all the source code for this project, where the two files main_roe.py and vec_env_roe.py is the ones needed for the implementation of the RoE-technique described in this paper: 
```
https://arxiv.org/abs/1803.07131
```
