
"""
Arguments for running main
"""

# Board size: 3, 5, 7, or 11
board_size = 3

if board_size == 3:
    board_width = 14
    board_height = 7
# elif board_size = 5:
# elif board_size = 7:
else:
    board_width = 14
    board_height = 7

# Basic arguments
num_updates = 200000
num_processes = 6
num_events = 5
num_steps = 10

# For logging and saving
debugging = True
log_interval = 50
model_to_save = "ac_agent_spp_1.pt" if not debugging else "(debugging).pt"
model_to_load = "ac_agent_spp_1.pt" if not debugging else "(debugging).pt"
log_filename = "ac_agent_spp_2.log" if not debugging else "(debugging).log"

# Learning rate and discount factor
learning_rate = 0.001
gamma = 0.99

# Rarity of Events
rarity_of_events = False
eb_capacity = 100


# Used when training is resumed after stop
resume = False
updates_when_stop = 6076 if resume else 0
timesteps_when_stop = 486480 if resume else 0

# Rendering
render = True
