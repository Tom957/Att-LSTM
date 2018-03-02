import tensorflow as tf
import argparse
import os
import pickle

from models.arnn import AttnRNN
from data.add.batch_loader import BatchLoader


def save(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class LSTMADD(object):
    def __init__(self, config, cell, reuse=None):
        self.config = config
        self.batch_size = config.batch_size
        step_size = config.step_size
        output_size = config.output_size
        input_size = config.input_size

        unit_size = config.unit_size
        learning_rate = config.learning_rate
        device = config.device
        batch_size = self.batch_size

        self.cell_name = cell.name
        self.itor_time = 0

        with tf.device(device), tf.name_scope(self.cell_name), tf.variable_scope("LSTMADD", reuse=reuse):
            with tf.variable_scope("inputs"):
                self.inputs = tf.placeholder(tf.float32, [batch_size, step_size, input_size])

            with tf.variable_scope("rnn_unit") as scope:
                cell_outputs, self.final_state = cell.dynamic_rnn(self.inputs, scope=scope)
                final_hidden = cell_outputs[-1]
                cell_outputs = tf.reshape(final_hidden, [batch_size, unit_size])

            with tf.variable_scope("full_conn"):
                w_shape = [unit_size, output_size]
                truc_init = tf.truncated_normal_initializer()
                fully_W = tf.get_variable('fully_W', w_shape, initializer=truc_init)
                fully_b = tf.get_variable('fully_b', [output_size])
                outputs = tf.matmul(cell_outputs, fully_W) + fully_b
                self.targets = tf.placeholder(tf.float32, [batch_size, output_size])

            with tf.variable_scope("train"):
                square_err = (self.targets - outputs) * (self.targets - outputs)
                self.mean_square_err = tf.reduce_mean(tf.reduce_sum(square_err, reduction_indices=[1]))

            with tf.variable_scope("optimize"):
                optimizer = tf.train.RMSPropOptimizer(learning_rate)
                gvs = optimizer.compute_gradients(self.mean_square_err)
                capped_gvs = [(None if grad is None else tf.clip_by_value(grad, -1.0, 1.0), var)
                              for grad, var in gvs]
                self.train_op = optimizer.apply_gradients(capped_gvs)

    def test(self, data, test_mses):
        self.itor_time += 1
        test_mse = 0.0
        num_test_batch = data.test_num
        for i in range(num_test_batch):
            batch_xs, batch_ys = data.next_test_batch()
            feed = {self.inputs: batch_xs, self.targets: batch_ys}
            test_mse += sess.run(self.mean_square_err, feed)
        test_mses.append(test_mse / num_test_batch)
        print('[iter_%d] %s test mse: %f' % (self.itor_time, self.cell_name,
                                             test_mse / num_test_batch))
        return test_mses

    def run(self, sess, data, mses):
        (train_mses, test_mses) = mses
        num_batch = data.train_num
        trainn_mse = 0.0
        for batch_id in range(num_batch):
            batch_xs, batch_ys = data.next_train_batch()
            feed = {self.inputs: batch_xs, self.targets: batch_ys}
            mse, _ = sess.run([self.mean_square_err, self.train_op], feed)
            trainn_mse += mse

            if (batch_id % 50 == 0):
                test_mses = self.test(data, test_mses)

        train_mses.append(trainn_mse / num_batch)

        return (train_mses, test_mses)


def parse_input():
    parser = argparse.ArgumentParser(description="Parse network configuretion")
    parser.add_argument("--batch_size",type=int, default=20)
    parser.add_argument("--step_size",type=int, default=600)
    parser.add_argument("--input_size",type=int, default=2)
    parser.add_argument("--output_size",type=int, default=1)
    parser.add_argument("--unit_size",type=int, default=100)
    parser.add_argument("--epoch_num",type=int, default=400)
    parser.add_argument("--learning_rate",type=float, default=1.0e-3)
    parser.add_argument("--device",type=str, default='/gpu:0')
    parser.add_argument("--cell_name",type=str, default="arnn")
    parser.add_argument("--K",type=int, default=8)
    return parser.parse_args()

if __name__ == "__main__":
    config = parse_input()
    dataset = BatchLoader(config.batch_size, config.step_size)
    graph = tf.Graph()
    with graph.as_default():
        name = config.cell_name
        cell = AttnRNN(config.unit_size, config.K, config.cell_name)
        model = LSTMADD(config, cell)

    sess_cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_cfg.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=sess_cfg) as sess:
        mses = ([], [])
        sess.run(tf.global_variables_initializer())
        path = os.path.join("result", "add")
        file_name =os.path.join(path, "add_"+model.cell_name + "_" + str(config.step_size)  + ".pkl")
        for epoch in range(config.epoch_num):
            mses = model.run(sess, dataset, mses)
            save(file_name, mses)
            print("[ep: %d] save secuess..%s\n\n" % (epoch + 1, file_name))


