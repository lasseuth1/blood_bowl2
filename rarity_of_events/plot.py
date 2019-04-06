import matplotlib.pyplot as plt
import numpy as np


def plot_training():
    # files_to_plot = ["logs/passes_log", "logs/casualty_log", "logs/(debugging).log"]
    files_to_plot = ["logs/test.log"]

    for file in files_to_plot:
        with open(file) as log:
            lines = log.readlines()
            x = [float(line.split(", ")[0]) for line in lines]
            y = [float(line.split(", ")[1]) for line in lines]
            z = moving_average(y, window_size=4)  # Smoothened data

        plt.plot(x, z)
        plt.title = file + " plot"
        plt.xlabel('Number of Episodes')
        plt.ylabel('Mean Reward for ' + file)
        plt.show()


def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


if __name__ == "__main__":
    plot_training()
