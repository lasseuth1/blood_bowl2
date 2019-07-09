
"""
Arguments for running main
"""

# Board size: 3, 5, 7, or 11
board_size = 5

if board_size == 3:
    board_width = 14
    board_height = 7
elif board_size == 1:
    board_width = 4
    board_height = 3
# elif board_size = 7:
else:
    board_width = 14
    board_height = 7

# Basic arguments
num_updates = 200000
num_processes = 8
num_events = 11
num_steps = 20

# For logging and saving
debugging = False
log_interval = 50
save_interval = 100

# Filenames
model_name = "5v5_ph_a2c_20_events.pt" if not debugging else "debugging.pt"
log_filename = "5v5_ph_a2c_20_events.log" if not debugging else "debugging.log"
log_event_file_name = "5v5_events_ph_a2c_20_events.log" if not debugging else "debugging.pt"
log_event_reward_file_name = "5v5_rewards_ph_a2c_20_events.log" if not debugging else "debugging.pt"


# Learning rate and discount factor
learning_rate = 0.001
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.05

# Rarity of events
rarity_of_events = False
eb_capacity = 100

# Used when training is resumed after stop
resume = False
log = True

# Rendering
render = False
# print_game = False
