import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import argparse
import os
from models.arnn import AttnRNN


def save(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)
def load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class LSTMCM(object):
    def __init__(self, config, cell, reuse=None):
        self.batch_size = config.batch_size
        self.step_size = config.step_size
        self.input_size = config.input_size
        self.class_num = config.class_num

        self.unit_size = config.unit_size
        self.learning_rate = config.learning_rate
        self.device = config.device

        self.K = cell.K
        self.cell_name = cell.name

        with tf.device(self.device), tf.name_scope(self.cell_name), \
             tf.variable_scope("LSTMCM", reuse=reuse):
            with tf.variable_scope("inputs"):
                self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.step_size])
                inputs = tf.reshape(self.inputs,
                                    [self.batch_size, self.step_size,  self.input_size])

            with tf.variable_scope("rnn_unit") as scope:
                lstm_outputs, self.final_state = cell.dynamic_rnn(inputs, scope=scope)
                final_hidden = lstm_outputs[-1]
                lstm_outputs = tf.reshape(final_hidden, [self.batch_size, self.unit_size])

            with tf.variable_scope("full_conn"):
                w_shape = [self.unit_size, self.class_num]
                truc_init = tf.truncated_normal_initializer()
                softmax_W = tf.get_variable('softmax_W', w_shape, initializer=truc_init)
                softmax_b = tf.get_variable('softmax_b', [self.class_num])
                outputs = tf.nn.softmax(tf.matmul(lstm_outputs, softmax_W) + softmax_b)
                self.targets = tf.placeholder(tf.float32, [self.batch_size, self.class_num])

            with tf.variable_scope("train"):
                self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.targets * tf.log(outputs),
                                                              reduction_indices=[1]))
                correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(self.targets, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            with tf.variable_scope("optimize"):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                gvs = optimizer.compute_gradients(self.cross_entropy)
                capped_gvs = [(None if grad is None else tf.clip_by_value(grad, -1., 1.), var)
                              for grad, var in gvs]
                self.train_op = optimizer.apply_gradients(capped_gvs)


    def run(self, sess, data, accuracys):
        (valid_accs, test_accs) = accuracys
        batch_size = self.batch_size
        num_examples = data.train.num_examples
        num_batch = int(num_examples / batch_size)
        for batch_id in range(num_batch):
            batch_xs, batch_ys = data.train.next_batch(batch_size)
            feed = {self.inputs: batch_xs, self.targets: batch_ys}
            loss, state, _ = sess.run([self.cross_entropy, self.final_state, self.train_op], feed)


        def valid_test():
            accuracy = 0.0
            num_valid_examp = data.validation.num_examples
            num_valid_batch = int(num_valid_examp / batch_size)
            for i in range(num_valid_batch):
                batch_xs, batch_ys = data.validation.next_batch(batch_size)
                feed = {self.inputs: batch_xs, self.targets: batch_ys}
                accuracy += sess.run(self.accuracy, feed)
            valid_accs.append(accuracy / num_valid_batch)
            print('[%s vald] acc: %f' % (self.cell_name, accuracy / num_valid_batch))

            accuracy = 0.0
            num_test_examp = data.test.num_examples
            num_test_batch = int(num_test_examp / batch_size)
            for i in range(num_test_batch):
                batch_xs, batch_ys = data.test.next_batch(batch_size)
                feed = {self.inputs: batch_xs, self.targets: batch_ys}
                accuracy += sess.run(self.accuracy, feed)
            test_accs.append(accuracy / num_test_batch)
            print('[%s test] acc: %f' % (self.cell_name, accuracy / num_test_batch))
            print('')
        valid_test()

        return (valid_accs, test_accs)

def parse_input():
    parser = argparse.ArgumentParser(description="Parse network configuretion")
    parser.add_argument("--batch_size",type=int, default=100)
    parser.add_argument("--step_size",type=int, default=784)
    parser.add_argument("--input_size",type=int, default=1)
    parser.add_argument("--class_num",type=int, default=10)
    parser.add_argument("--unit_size",type=int, default=100)
    parser.add_argument("--epoch_num",type=int, default=800)
    parser.add_argument("--test_iter",type=int, default=100)
    parser.add_argument("--learning_rate",type=float, default=1.0e-3)
    parser.add_argument("--clip_value",type=float, default=1.0)
    parser.add_argument("--device",type=str, default='/gpu:0')
    parser.add_argument("--cell_name",type=str, default="arnn")
    parser.add_argument("--K",type=int, default=8)
    return parser.parse_args()

if __name__ == "__main__":
    config = parse_input()
    mnist = input_data.read_data_sets("./data/mnist", one_hot=True)
    graph = tf.Graph()
    with graph.as_default():
        cell = AttnRNN(config.unit_size, config.K, config.cell_name)
        model = LSTMCM(config, cell)

    sess_cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_cfg.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=sess_cfg) as sess:
        valid_accs = []
        test_accs = []
        accuracys = (valid_accs, test_accs)
        if config.cell_name == 'arnn':
            fname = "mnist_arnn_k%d.pkl" % (config.K)
        else:
            fname = "mnist_%s.pkl" % (config.cell_name)
        file_name = os.path.join("result", "mnist", fname)
        sess.run(tf.global_variables_initializer())
        for epoch in range(config.epoch_num):
            accuracys = model.run(sess, mnist, accuracys)
            save(file_name, accuracys)
            print("[ep: %d] save secuess..%s\n\n" % (epoch+1, file_name))

