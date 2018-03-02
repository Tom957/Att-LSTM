import pickle
import matplotlib.pyplot as plt
import os

def save(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    plot_start = 00

    idx = 0

    plot_nums = [300,400,800,1000]
    step_sizes = [100,200,400,600]

    plot_num = plot_nums[idx]
    step_size = step_sizes[idx]

    plot_idx = 1
    linewidth = 1.1

    mses = [0.176 for _ in range(plot_num)]
    x = range(len(mses))
    plt.plot(x, mses, "k--", label="Baseline", linewidth=linewidth)

    file_name = "./add_rnn_%d.pkl"%step_size
    if os.path.exists(file_name):
        mses = load(file_name)[plot_idx][:plot_num]
        for i,x in enumerate(mses):
            print("[%d] rnn: %f" % (i,x))
        x = range(len(mses))
        plt.plot(x, mses, "g", label="tanh RNN", linewidth=1)
        print("")

    file_name = "./add_lstm_%d.pkl" % step_size
    if os.path.exists(file_name):
        mses = load(file_name)[plot_idx][:plot_num]
        for i, x in enumerate(mses):
            print("[%d] lstm: %f" % (i,x))
        x = range(len(mses))
        plt.plot(x, mses, "b", label="LSTM", linewidth=linewidth)
        print("")

    file_name = "./add_arnn_%d.pkl"%step_size
    if os.path.exists(file_name):
        mses = load(file_name)[plot_idx][:plot_num]
        for i in range(plot_num):
            if i > len(mses):
                mses.append(mses[-1])
        for i, y in enumerate(mses):
            print("[%d] arnn: %f" % (i, y))
        x = range(len(mses))
        plt.plot(x, mses, "r", label="Att-LSTM", linewidth=linewidth)
        print("")



    plt.ylim((0, 1))
    plt.xlim((plot_start, plot_num))
    plt.xlabel("Traning steps")
    plt.ylabel("Test MSE")
    plt.title("Sequence length = %d" % step_sizes[idx])
    plt.legend(loc='upper right')
    plt.show()
