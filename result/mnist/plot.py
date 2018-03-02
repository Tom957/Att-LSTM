import pickle
import numpy as np
import matplotlib.pyplot as plt

def save(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    idx = 0
    plot_start = 00
    batch_num_per_epoch = 600
    plot_num = 250

    lstm_accs_test = load("./mnist_arnn_k8.pkl")[idx]
    lstm_accs = lstm_accs_test[:plot_num]
    lstm_accs = [acc + 0.00 for acc in lstm_accs]
    for x in lstm_accs_test:
        print("arnn: " + str(x))
    lstm_x = range(len(lstm_accs))
    lstm_x = [x * 600 for x in lstm_x]
    plt.plot(lstm_x, lstm_accs, "r", label="Att-LSTM", linewidth=2)
    print()

    lstm_accs_test = load("./mnist_lstm.pkl")[idx]
    lstm_accs = lstm_accs_test[:plot_num]
    lstm_accs = [acc + 0.00 for acc in lstm_accs]
    for x in lstm_accs_test:
        print("lstm: " + str(x))
    lstm_x = range(len(lstm_accs))
    lstm_x = [x * 600 for x in lstm_x]
    plt.plot(lstm_x, lstm_accs, "b", label="LSTM", linewidth=2)
    print()
    plt.xlabel("Trianing step")
    plt.ylabel("Accuracy")
    plt.title("Pixel-by-pixel MNIST")
    plt.xlim((plot_start, plot_num*batch_num_per_epoch))
    plt.legend(loc='lower right')
    plt.show()

