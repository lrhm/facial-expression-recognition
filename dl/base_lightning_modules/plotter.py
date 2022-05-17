import matplotlib.pyplot as plt
#

def plot_train_loss(train,val,save_path):
    plt.clf()
    plt.plot(*zip(*train), label = "train")
    plt.plot(*zip(*train), label = "val")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Training loss")
    plt.savefig(save_path)

