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
    plot_data = "msqc"
    plot_num = 100
    plot_idx = 1
    colors = ['r','b','g','m','c','k','m']
    cells = ["lstm","arnn_k2","arnn_k4"]
    models = ["Hierarchical LSTM","Hierarchical Att-LSTM",""]
    col_idx = 0
    for model, cell in zip(models,cells):
        file_name = "./%s_%s.pkl" % (plot_data, cell)
        if os.path.exists(file_name):
            loss_accs = load(file_name)[:plot_num]
            for ls, ac in loss_accs:
                print("%s_%s: %.4f" %(plot_data, cell, ac))
            x = range(len(loss_accs))
            y = [ls_ac[plot_idx] for ls_ac in loss_accs]
            plt.plot(x, y, colors[col_idx]+"", label=model , linewidth=1.5)
            col_idx += 1
            print()
        else:
            print("not data file")
    if plot_idx == 1 and plot_data == 'trec':
        plt.ylim((0.75, 1.0))
        plt.xlim((0, plot_num))
        plt.legend(loc='lower right')
    elif plot_idx == 1 and plot_data == 'msqc':
        plt.ylim((0.7, .95))
        plt.xlim((0, plot_num))
        plt.legend(loc='lower right')
    else:
        plt.ylim((0, 0.015))
        plt.xlim((0, plot_num))
        plt.legend(loc='upper right')

    plt.xlabel("Trianing step")
    plt.title("MSQC question classification")
    plt.ylabel("Accuracy")
    plt.show()
