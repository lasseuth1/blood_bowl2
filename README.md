# Reinforcement Learning & Blood Bowl

The code provided in the repository is an attempt of applying Reinforcement Learning to Blood Bowl. 

The code is structured in a way, that runs parallel environments of Blood Bowl using the code in ``vec_env.py``, where the trajectories collected is used for updating the models parameters after a fixed number of steps. A lot of the configurations for the code can be altered in arguments.py.

## FFAI
The folder named ffai is a copy of the repository from https://github.com/njustesen/ffai - containing all that is needed for running the FFAI Game Engine. 

## Rarity Of Events
The folder named rarity_of_events contains all the source code for this project, where the two files ``main_roe.py`` and ``vec_env_roe.py`` is the ones needed for the implementation of the RoE-technique described in this paper: 
```
https://arxiv.org/abs/1803.07131
``` 

## Results

Currently, preliminary results have been acquired on the three smallest/easiest Gym environments in FFAI, which has a board of 4 × 3, 12 × 5, and 16 × 9 squares with 1, 3, and 5 fielded players on each team, respectively. A more comprehensive description of the experiments can be found in the paper: Blood Bowl: A New Board Game Challenge and Competition for AI (https://njustesen.files.wordpress.com/2019/07/justesen2019blood.pdf)


## Rewards
Be aware that FFAI provides a reward at each timestep, if your team is leading. This can be altered in ``env.py`` in the ``_step()`` function. 
The results mentioned above, uses a different reward function and can be altered from ``vec_env.py`` in the ``get_events()`` function. For rarity of events, the reward is calculated from the events defined.


## Reproducibility 

**Before getting started:**
* We recommend to use Anaconda (or other similar software) for package management.
* The project is implemented using python3.6
* Install Pytorch installed (https://pytorch.org/ )
* Install matplotlib

**The code:**

In order to switch between board sizes, change the variable board_size in ``arguments.py``

As of know,  when switching between board sizes, there are several places in the code where changes must be made manually in order for the program to run. Now, the code is set-up for 5v5.

The following files contains commented lines such as: # 1v1, # 3v3 and # 5v5, which is where the lines following must be either commented, or commented out, depending on the variant of the board:

**main.py** and **main_roe**:
* These are quite similar, but ``main_roe.py`` is used for Rarity of Events. 
Search for # 1v1 or # 3v3 or # 5v5.

**memory.py:**
* This is where observations of each timestep are stored. The action_space must be changed to fit the board size.
Search for # 1v1 or # 3v3 or # 5v5.

**pruned_hybrid.py:**
* The model/architecture file.
The sizes of the layers changes when the board size is changed.  The change should be made in the ``__init__`` function of the PrunhedHybrid class (line 49-69).
Search for # 1v1 or # 3v3 or # 5v5.

**vec_env.py** and **vec_env_roe.py:**
* The files we use to run parallel FFAI environments. Again, the files are similar, but ``vec_env_roe.py`` is used for implemented Rarity of Events for each environment.
Search for # 1v1 or # 3v3 or # 5v5.



