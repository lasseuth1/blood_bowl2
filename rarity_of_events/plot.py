import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training():
    # files_to_plot = ["logs/passes_log", "logs/casualty_log", "logs/100kepisodes_spp1.log"]
    files_to_plot = ["logs/1v1_models/1v1_hybrid_a2c_10_masking_touchdown.log"]

    for file in files_to_plot:
        with open(file) as log:
            lines = log.readlines()
            x = [float(line.split(",")[3]) for line in lines]  # time steps
            z = [float(line.split(",")[5]) for line in lines]  # reward
            # h = [float(line.split(",")[5]) for line in lines]  # mean reward
            # w = [float(line.split(",")[2]) for line in lines]  # episodes this update
            # z = [float(line.split(",")[6]) for line in lines]  # touchdown
            # d = [0 for i in range(1288)] # mean reward step
            #
            # for i in np.arange(len(y)):
            #     reward_td = float(z[i]*10)
            #     z[i] = reward_td
            #     if w[i] != 0:
            #         d[i] = (y[i] - reward_td) / w[i]
            #     if z[i] != 0:
            #         pr_episode = float(z[i]/w[i])
            #         z[i] = pr_episode


            xs = []
            zs = []
            # ds = []
            # hs = []
            for i in np.arange(0, len(x)):
                mean_x = []
                mean_z = []
                # mean_d = []
                # mean_h = []
                if i == 0:
                    mean_x.append(x[i])
                    mean_z.append(z[i])
                    # mean_d.append(d[i])
                    # mean_h.append(h[i])
                for j in np.arange(max(0, i - round(10 / 2)), min(i + round(10 / 2), len(x))):
                    mean_x.append(x[j])
                    mean_z.append(z[j])

                xs.append(np.mean(mean_x))
                zs.append(np.mean(mean_z))
                # ds.append(np.mean(mean_d))
                # hs.append(np.mean(mean_h))

        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.rcParams.update({'font.size': 14})
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_facecolor('#F2F2F2')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)

        # colors used : #990000, #734d26, #3333ff

        plt.plot(xs, zs, color='blue', label='A2C', linestyle="-")
       # plt.plot(x, z, color='blue', linewidth=1, alpha=0.2)  # noise


        #plt.plot(x, y, color='orange', linewidth=1, alpha=0.3)  # noise

        plt.title("1v1", fontsize=18)

        # plt.title('Performance on 3v3')
        plt.xlabel('Steps', fontsize=16)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 5))
        plt.ylabel('Touchdowns / Episode', fontsize=16)

        plt.axhline(y=1.8, color='#006600', linestyle='--', label='EndzoneBot') # for 1v1
        plt.axhline(y=0.215, color='#990000', linestyle=':', label='RandomBot')
        # plt.axhline(y=0.87, color='green', linestyle='--', label='Endzone baseline') # for 3v3
        #plt.legend(loc='upper left')

        plt.ylim(bottom=0)
        plt.xlim(left=0, right=1400000)
        plt.style.use('ggplot')
        plt.grid(alpha=0.3)
        plt.show()
        fig.savefig("FFAI-v1-1_A2C.pdf")


if __name__ == "__main__":
    plot_training()
