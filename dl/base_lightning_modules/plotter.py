import matplotlib.pyplot as plt
#

def plot_train_loss(data, save_path):
    plt.clf()
    plt.plot(*zip(*data))
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.savefig(save_path)

