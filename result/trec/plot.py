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
    plot_data = "trec"
    plot_num = 100
    test_each = 1
    plot_idx = 1
    colors = ['r','b','g','m','c','k','m']
    cell_name = ["lstm","arnn_k2","lstm.old"]
    model_name = ["Hierarchical LSTM","Hierarchical Att-LSTM",""]
    col_idx = 0
    for model, cell in zip(model_name,cell_name):
        file_name = "./%s_%s.pkl" % (plot_data, cell)
        if os.path.exists(file_name):
            loss_accs = load(file_name)[:plot_num]
            acs = []
            for i, (ls, ac) in enumerate(loss_accs):
                if i % test_each == 0:
                    acs.append(ac)
                    print("%s_%s: loss=%.4f, accuracy=%.4f" %(plot_data, cell, ls, ac))
            x = range(len(acs))
            y = acs
            plt.plot(x, y, colors[col_idx], label=model , linewidth=1.5)
            col_idx += 1
            print()
    if plot_idx == 1 and plot_data == 'trec':
        plt.ylim((0.7, 1.0))
        plt.xlim((0, plot_num/test_each))
        plt.legend(loc='lower right')
    elif plot_idx == 1 and plot_data == 'msqc':
        plt.ylim((0.85, .95))
        plt.xlim((0, plot_num/test_each))
        plt.legend(loc='lower right')
    else:
        plt.ylim((0, 0.015))
        plt.xlim((0, 100))
        plt.legend(loc='upper right')

    plt.xlabel("Trianing step")
    plt.title("TREC question classification")
    plt.ylabel("Accuracy")
    plt.show()
