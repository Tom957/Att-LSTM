import tensorflow as tf
from tensorflow.contrib import layers
import sys
import os


from models.model import Model
from models.arnn import AttnRNN

class HierRNN(Model):
    """
    Hierarchical Recurrent Neural Network
    """

    def __init__(self, sess, loader, config):
        self.sess = sess

        # Data Information
        self.loader = loader

        self.batch_size = loader.batch_size
        self.max_word_length = loader.max_word_length
        self.max_sent_length = loader.max_sent_length
        self.char_vocab_size = loader.char_vocab_size
        self.word_vocab_size = loader.word_vocab_size

        self.ckp_dir = config.ckp_dir
        self.dataset_name = loader.data_name
        self.ckp_name = config.ckp_name
        self.result_dir = config.result_dir

        # Char Embedding
        self.char_embed_dim = config.char_embed_dim

        # Highway
        self.highway_layers = config.highway_layers

        # RNN
        self.first_unit_size = config.first_unit_size
        self.secod_unit_size = config.secod_unit_size

        # FC
        self.num_classes = loader.class_num

        # Training
        self.is_test = config.is_test
        self.test_per_batch = config.test_per_batch
        self.cell_name = config.cell_name
        self.clip_norm = config.clip_norm
        self.learning_rate = config.learning_rate
        self.learning_rate_decay = config.learning_rate_decay
        self.reg_lambda = config.reg_lambda
        self.K = config.K
        self.accuracys = []

        # Build Hierarchical LSTM model
        print("building model...")
        self.create_model()
        print("builded model\n")

    def highway(self, input_, layer_size=1, bias=-2, g=tf.nn.relu, name="highway"):
        """Highway Network (cf. http://arxiv.org/abs/1505.00387).
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
        where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
        """
        output = input_
        size = input_.get_shape()[1]

        with tf.variable_scope(name):
            for idx in range(layer_size):
                with tf.variable_scope('output_lin_%d' % idx):
                    output = g(tf.nn.rnn_cell._linear(output, size, 1))
                with tf.variable_scope('transform_lin_%d' % idx):
                    transform_gate = tf.sigmoid(tf.nn.rnn_cell._linear(input_, size, 1) + bias)
                carry_gate = 1. - transform_gate
                output = transform_gate * output + carry_gate * input_
        return output

    def mlp(self, input_, layer_size=1, bias=-2, g=tf.nn.sigmoid, name="mlp"):
        """MLP
        output = g(Wy + b)
        where g is nonlinearity,
        """
        output = input_
        size = input_.get_shape()[1]
        with tf.variable_scope(name):
            output = g(tf.nn.rnn_cell._linear(output, size, 1))
        return output

    def batch_norm(self, input_, epsilon=1e-3, name="batch_norm"):
        shape = input_.get_shape()
        with tf.variable_scope(name):
            gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))
            beta = tf.get_variable("beta", [shape[-1]],
                                   initializer=tf.constant_initializer(0.))
            mean, variance = tf.nn.moments(input_, [0, 1, 2])
            return tf.nn.batch_norm_with_global_normalization(input_, mean,
                                                              variance, beta,
                                                              gamma, epsilon,
                                                              scale_after_normalization=True)

    def batch_norm_v2(self, x, epsilon=1e-5, name="batch_norm"):
        """Code modification of http://stackoverflow.com/a/33950177"""
        shape = x.get_shape()
        with tf.variable_scope(name):
            size = shape[-1]
            randm_init = tf.random_normal_initializer(1., 0.02)
            const_init = tf.constant_initializer(0.)
            gamma = tf.get_variable("gamma", [size], initializer=randm_init)
            beta = tf.get_variable("beta", [size], initializer=const_init)
            mean, variance = tf.nn.moments(x, [0])
            return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)

    def create_model(self):
        with tf.variable_scope("hier_stc"):
            with tf.variable_scope("input"):
                shape = [self.char_vocab_size, self.char_embed_dim]
                char_embedding = tf.get_variable("char_embed", shape,dtype=tf.float32)
                shape = [self.batch_size, self.max_sent_length, self.max_word_length]
                self.char_inputs = tf.placeholder(tf.int32, shape)

            with tf.variable_scope("create_lstm"):
                first_cell = AttnRNN(self.first_unit_size, self.K, self.cell_name)
                second_cell = AttnRNN(self.secod_unit_size, self.K, self.cell_name)

            with tf.variable_scope("first_level") as scope:
                char_outputs = []

                char_idxs = tf.unstack(self.char_inputs, self.max_sent_length, 1)
                for idx, char_idx in enumerate(char_idxs):
                    if idx != 0: scope.reuse_variables()
                    char_embed = tf.nn.embedding_lookup(char_embedding, char_idx)
                    char_embed_slices = tf.unstack(char_embed, self.max_word_length, 1)
                    char_output_list, _ = first_cell.dynamic_rnn_v2(char_embed_slices,
                                                                   scope="first_cell")
                    char_output = tf.concat_v2(char_output_list,1)
                    char_output = self.batch_norm_v2(char_output)
                    if self.highway_layers >= 1:
                        char_output = self.highway(char_output, self.highway_layers, 0)
                    elif  self.highway_layers == -1:
                        char_output = self.mlp(char_output, 1, 0)
                    char_outputs.append(char_output)

            with tf.variable_scope("second_level") as scope:
                word_output_list, _ = second_cell.dynamic_rnn_v2(char_outputs,
                                                                 scope="second_cell")
                word_outputs = word_output_list

            rnn_outs = word_outputs
            with tf.variable_scope("fully_connect"):
                rnn_outs = tf.concat_v2(rnn_outs, 1)
                # rnn_outs = rnn_outs[-1]
                out_len = rnn_outs.get_shape()[1]

                regularizer = layers.l2_regularizer(self.reg_lambda)
                softmax_W = tf.get_variable(name="softmax_W",
                                            shape=[out_len, self.num_classes],
                                            dtype=tf.float32,
                                            regularizer=regularizer)
                softmax_b = tf.get_variable(name="softmax_b",shape=[self.num_classes])
                fc_output = tf.nn.xw_plus_b(rnn_outs, softmax_W, softmax_b)

            with tf.variable_scope("train_ops"):
                self.target_outputs = tf.placeholder(tf.float32,
                                                     [self.batch_size, self.num_classes])
                with tf.variable_scope('loss'):
                    loss = tf.nn.softmax_cross_entropy_with_logits(fc_output,
                                                                   self.target_outputs)
                    self.loss = tf.reduce_mean(loss) / self.batch_size
                # Accuracy
                with tf.name_scope("accuracy"):
                    predictions = tf.argmax(fc_output, 1, name="predictions")
                    correct_predictions = tf.equal(predictions, tf.argmax(self.target_outputs, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"),
                                                   name="accuracy")

    def train(self,ep):
        cost = 0
        times = 0
        loss_sum = 0.0
        acc_sum = 0.0
        t_sum = 0
        while True:
            target_output_arr, \
            words_index_arr, \
            chars_index_arr, \
            isOneEpoch = self.loader.next_batch()
            feed_dict = {
                self.char_inputs: chars_index_arr,
                self.target_outputs: target_output_arr
            }
            ops = [self.optimz_op, self.loss, self.accuracy, self.global_step]
            _, loss, accuracy, step = self.sess.run(ops, feed_dict=feed_dict)

            t_sum += 1
            acc_sum += accuracy
            loss_sum += loss
            cost += loss
            times += 1
            if self.test_per_batch >=0 and times % self.test_per_batch == 0:
                self.test(ep)
            if isOneEpoch is True: break
        return cost / times

    def test(self,ep):
        target_output_arr, \
        words_index_arr, \
        chars_index_arr = self.loader.get_test_data()
        test_len = len(target_output_arr)
        start_idx = 0
        losses = 0.0
        accuracys = 0.0
        while True:
            if start_idx + self.batch_size > test_len: break;
            chars_index_batch = chars_index_arr[start_idx: start_idx + self.batch_size]
            target_output_batch = target_output_arr[start_idx: start_idx + self.batch_size]
            feed_dict = {
                self.char_inputs: chars_index_batch,
                self.target_outputs: target_output_batch
            }
            loss, accuracy = self.sess.run([self.loss, self.accuracy],feed_dict=feed_dict)
            losses += loss
            accuracys += accuracy
            start_idx += self.batch_size
        losses = losses / (start_idx / self.batch_size)
        accuracys = accuracys / (start_idx / self.batch_size)
        self.accuracys.append((losses, accuracys))
        if self.cell_name == 'arnn':
            fname = "%s_arnn_k%d.pkl" % (self.dataset_name,self.K)
        else:
            fname = "%s_%s.pkl" % (self.dataset_name,self.cell_name)
        path = os.path.join(self.result_dir,self.dataset_name,fname)
        self.save_obj(path, self.accuracys)
        print("Test [%d]: losses %f, accuracys %f" % (ep, losses, accuracys))
        return loss

    def run(self, epoch=300):
        # Define Training procedure
        self.lr = tf.Variable(self.learning_rate, trainable=False)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.optimz_op = optimizer.apply_gradients(grads_and_vars,
                                                   global_step=self.global_step)

        # ready for train
        print("initializing variables...")
        tf.global_variables_initializer().run()
        print("initialized variables\n")

        if not self.is_test:
            for ep in range(epoch):
                self.train(ep+1)
                self.test(ep+1)
                lr = self.learning_rate * (self.learning_rate_decay**ep)
                self.sess.run(tf.assign(self.lr, lr) )
                if (ep+1) % 100 == 0:
                    is_save = input("Is save checkpoint? yes or no: ")
                    if is_save == "yes":
                        if self.cell_name == 'arnn':
                            fname = "arnn_k%d_ep%d.pkl" % (self.K, ep+1)
                        else:
                            fname = "%s_ep%d.pkl" % (self.cell_name, ep+1)
                        self.save(self.ckp_dir, self.dataset_name, fname)
        else:
            if self.load(self.ckp_dir, self.dataset_name,self.ckp_name):
                self.test(1)
            else:
                sys.exit(1)
