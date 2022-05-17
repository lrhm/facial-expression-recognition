import matplotlib.pyplot as plt
import numpy as np
import ipdb


def plot_train_loss(train, val, save_path):
    plt.clf()

    # detach items in train and val

    detatcher = lambda x: (x[0], x[1].cpu())
    # train = list(map(detatcher, train))
    val = list(map(detatcher, val))

    plt.plot(*zip(*train), label="train")
    plt.plot(*zip(*val), label="val")
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.savefig(save_path)
