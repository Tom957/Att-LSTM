import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def save(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)
def load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    plot_data = "pmnist"
    batch_num_per_epoch = 600
    plot_num = 200
    plot_idx = 0
    colors = ['b','r']
    models = ["lstm", "arnn_k8"]
    models_name = ["LSTM", "Att-LSTM"]
    for i in range(len(models)):
        file_name = "./%s_%s.pkl" % (plot_data, models[i])
        if os.path.exists(file_name):
            accs = load(file_name)[plot_idx]
            acs = []
            for a, ac in enumerate(accs):
                if (a+1) % 6 == 0:
                   acs.append(ac)
            x = range(len(acs))

            x = [l * batch_num_per_epoch for l in x]
            for j, y_ in enumerate(acs):
                print("%s: %3d, %.4f" % (models[i], j, y_))
            y = [ac for ac in acs]
            plt.plot(x, y, colors[i], label=models_name[i], linewidth=2)
            print()
    plt.ylim((0, 1))
    plt.xlim((0,  plot_num*batch_num_per_epoch))

    plt.xlabel("Trianing step")
    plt.ylabel("Accuracy ")

    plt.title("Pixel-by-pixel permuted MNIST")
    plt.legend(loc='lower right')
    plt.show()



