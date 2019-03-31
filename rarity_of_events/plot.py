import matplotlib.pyplot as plt
import numpy as np


def plot_training():
    with open('training_new_noevents.log') as log:
        lines = log.readlines()
        x = [float(line.split(", ")[1]) for line in lines]
        y = [float(line.split()[2]) for line in lines]

    plt.plot(x, y)
    plt.title = "Testing"
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.show()


if __name__ == "__main__":
    plot_training()
