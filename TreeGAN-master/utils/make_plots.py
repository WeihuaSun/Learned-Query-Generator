import pickle
from matplotlib import pyplot as plt
import numpy as np


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[n:] = ret[n:] / n
    ret[:n] = ret[:n] / np.arange(1, n + 1)
    return ret


def make_losses_plot(losses, title):
    flat_losses = [item for sublist in losses for item in sublist]

    plt.plot(flat_losses, label='loss')
    plt.plot(moving_average(np.array(flat_losses), 100), label='moving avg')
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    x_ids = []
    last = 0
    for l in losses:
        last += len(l)
        x_ids.append(last)
    plt.xticks(x_ids, [i + 1 for i in range(len(losses))])
    plt.legend()
    plt.show()


def make_loss_plot(loss, title):
    plt.plot(loss, label='loss')
    plt.plot(moving_average(np.array(loss), 100), label='moving avg')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title(title)
    plt.legend()
    plt.show()


def make_rewards_plot(rewards):
    plt.plot(rewards, label='avg episode reward')
    plt.plot(moving_average(np.array(gen_rewards), 50), label='moving avg')
    plt.xlabel('episode')
    plt.ylabel('avg reward')
    plt.title("generator rewards")
    plt.legend()
    plt.show()


with open('./../data/java/stats/train4/pre_train_generator_losses', 'rb') as f:
    pre_train_gen_losses = pickle.load(f)

with open('./../data/java/stats/train4/discriminator_losses', 'rb') as f:
    disc_losses = pickle.load(f)

with open('./../data/java/stats/train4/generator_losses', 'rb') as f:
    gen_losses = pickle.load(f)

with open('./../data/java/stats/train4/generator_rewards', 'rb') as f:
    gen_rewards = pickle.load(f)


make_losses_plot(pre_train_gen_losses, "pre-training generator loss")
make_loss_plot(disc_losses[0], "discriminator loss")
make_losses_plot(gen_losses, "generator loss")
make_rewards_plot(gen_rewards)







