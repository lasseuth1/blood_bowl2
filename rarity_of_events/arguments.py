
"""
Arguments for running main
"""

# Board size: 3, 5, 7, or 11
board_size = 3

# Basic arguments
num_updates = 200000
num_processes = 8
num_events = 5
num_steps = 10

# For logging and saving
log_interval = 50
model_name = "ac_model_1_noevents.pt"
model_to_load = "ac_model_1_noevents.pt"
log_filename = "testing_touchdowns.log"

# Learning rate and discount factor
learning_rate = 0.001
gamma = 0.99

# Rarity of Events
rarity_of_events = False
eb_capacity = 100

# Used when training is resumed after stop
updates_when_stop = 5019
timesteps_when_stop = 401600
