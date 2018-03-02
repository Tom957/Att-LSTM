import numpy as np
import tensorflow as tf

class AttnRNN(object):
    '''AttnRNN implemented with same initializations'''
    def __init__(self, num_units, K=1, mode="lstm"):
        """
        :param num_units: a scalar for outputs vector size
        :param K: a scalar for previous size
        """
        self._num_units = num_units
        self.K = K
        self.is_tree = True
        self.mode = mode
        self.type = "rnn"
        self.name = mode
        print("creat a "+mode)

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def orthogonal(self,shape):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return q.reshape(shape)

    def identity_initializer(self, scale):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            '''Ugly cause LSTM params calculated in one matrix multiply'''
            size = shape[0]
            t = np.identity(size) * scale
            return tf.constant(t, dtype)
        return _initializer

    def orthogonal_initializer(self):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            return tf.constant(self.orthogonal(shape), dtype)
        return _initializer

    def __call__(self, inputs, states, scope=None):
        """
        :param inputs: Tensor of shape [batch_size, input_size]
                `batch_size` will be preserved (known)
                `input_size` must be static (known)
        :param states: include a list of cells and a list of hiddens, list size is K
                `cells` a list of tensor, shape is [batch_size, outputs_size]
                `hiddens`  a list of tensor, shape is [batch_size, outputs_size]
        :param scope: variable scope
        :return: outputs: include hidden_ and state, state include cell_ and hidden_
                 hidden_:Tensor of shape [batch_size, outputs_size].
                 cell_:Tensor of shape [batch_size, outputs_size].
        """
        with tf.variable_scope(scope or type(self).__name__):
            (cells, hiddens) = states

            inputs_size = inputs.get_shape().as_list()[1]
            x_shape = [inputs_size, self._num_units]
            h_shape = [self._num_units, self._num_units]
            b_shape = [self._num_units]
            orth_init = self.orthogonal_initializer()
            iden_init = self.identity_initializer(0.95)

            i_x_weight = tf.get_variable('weight_x_i',x_shape, initializer=orth_init)
            u_x_weight = tf.get_variable('weight_x_u',x_shape, initializer=orth_init)
            f_x_weight = tf.get_variable('weight_x_f',x_shape, initializer=orth_init)
            o_x_weight = tf.get_variable('weight_x_o',x_shape, initializer=orth_init)

            i_h_weight = tf.get_variable('weight_h_i',h_shape, initializer=orth_init)
            u_h_weight = tf.get_variable('weight_h_u',h_shape, initializer=iden_init)
            f_h_weight = tf.get_variable('weight_h_f',h_shape, initializer=orth_init)
            o_h_weight = tf.get_variable('weight_h_o',h_shape, initializer=orth_init)

            # variable tensor, shape is [outputs_size]
            zero_init = tf.zeros_initializer()
            i_bias = tf.get_variable('bias_xh_i', b_shape, initializer=zero_init)
            u_bias = tf.get_variable('bias_xh_u', b_shape, initializer=zero_init)
            f_bias = tf.get_variable('bias_xh_f', b_shape, initializer=zero_init)
            o_bias = tf.get_variable('bias_xh_o', b_shape, initializer=zero_init)

            if self.mode == "rnn":
                hidden_next = tf.matmul(inputs, i_x_weight) + tf.matmul(hiddens[-1], i_h_weight) + i_bias
                hidden_next = tf.tanh(hidden_next)
                outputs = hidden_next, hidden_next

            elif self.mode == "lstm":
                i = tf.matmul(inputs, i_x_weight) + tf.matmul(hiddens[-1], i_h_weight) + i_bias
                u = tf.matmul(inputs, u_x_weight) + tf.matmul(hiddens[-1], u_h_weight) + u_bias
                o = tf.matmul(inputs, o_x_weight) + tf.matmul(hiddens[-1], o_h_weight) + o_bias
                f = tf.matmul(inputs, f_x_weight) + tf.matmul(hiddens[-1], f_h_weight) + f_bias
                cell_next = cells[-1] * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(u)
                hidden_next = tf.tanh(cell_next) * tf.sigmoid(o)
                outputs = hidden_next, cell_next
            else:
                f = tf.matmul(inputs, f_x_weight) + tf.matmul(hiddens[-1], f_h_weight) + f_bias
                o = tf.matmul(inputs, o_x_weight) + tf.matmul(hiddens[-1], o_h_weight) + o_bias
                i = tf.matmul(inputs, i_x_weight) + tf.matmul(hiddens[-1], i_h_weight) + i_bias
                i_gt = tf.sigmoid(i)
                hs = list()
                for k, (cell, hid) in enumerate(zip(cells, hiddens)):
                    h_k_weight = tf.get_variable('weight_hu_%d' % k, h_shape, initializer=iden_init)
                    h_k_bias = tf.get_variable('bias_xh_uk_%d' % k, b_shape, initializer=zero_init)
                    h_k = tf.matmul(hid * i_gt, h_k_weight) + h_k_bias
                    hs.append(h_k)
                u_h = self.attention(hs, 1)
                u =tf.matmul(inputs, u_x_weight) + u_h + u_bias
                cell_next = cells[-1] * tf.sigmoid(f) + tf.tanh(u) * (1 - tf.sigmoid(f))
                hidden_next = tf.tanh(cell_next) * tf.sigmoid(o)
                outputs = hidden_next, cell_next

            return outputs

    def attention(self, inputs, attn_size=0):
        """Attention mechanism layer.
        :param inputs: outputs of RNN layer (not final state)
        :param attention_size: linear size of attention weights
        :return: outputs of the passed RNN reduced with attention vector
        """
        if isinstance(inputs, tuple):
            inputs = tf.concat(2, inputs)
        elif isinstance(inputs, list):
            inputs = tf.stack(inputs,1)

        batch_size = inputs.get_shape()[0].value
        seq_len = inputs.get_shape()[1].value
        hid_size = inputs.get_shape()[2].value

        if attn_size <= 0:
            attn_size = self.K #300 1

        # Attention mechanism
        W = tf.Variable(tf.random_normal([hid_size, attn_size], stddev=0.1))
        b = tf.Variable(tf.random_normal([1, attn_size], stddev=0.1))
        W_u = tf.Variable(tf.random_normal([attn_size, 1], stddev=0.1))

        inp_rshp = tf.reshape(inputs, [batch_size*seq_len, hid_size])
        u = tf.tanh(tf.matmul(inp_rshp, W) + b)
        uv = tf.matmul(u, W_u)
        exps = tf.reshape(tf.exp(uv), [batch_size, seq_len])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [batch_size, 1])

        # Output of RNN is reduced with attention vector
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, seq_len, 1]), 1)
        return output

    def dynamic_rnn(self, inputs, states=None, scope=None):
        """
        :param inputs: Tensor of shape [batch_size, times, input_size]
                        `batch_size` will be preserved (known)
                        `times` sequence length
                        `input_size` must be static (known)
        :param states: None create zero, else use input
                        states contain cells and hiddens
                        cells and hiddens is a list, K element,
                        element shape is [batch_size, outputs_size]
        :param scope: a String
        :return: Tensor of shape [batch_size, times, outputs_size]
        """
        batch_size = inputs.get_shape().as_list()[0]
        times = inputs.get_shape().as_list()[1]
        input_size = inputs.get_shape().as_list()[2]
        outputs_size = self._num_units
        outputs = []
        with tf.variable_scope(scope or type(self).__name__):
            if states == None:
                cells = []
                hiddens = []
                for i in range(self.K):
                    cell = tf.zeros([batch_size, outputs_size])
                    hidden = tf.zeros([batch_size, outputs_size])
                    cells.append(cell)
                    hiddens.append(hidden)
                states = (cells, hiddens)
            else:
                cells, hiddens = states
            inps = tf.unstack(inputs,times,1)
            for idx, inp in enumerate(inps):
                if idx > 0: tf.get_variable_scope().reuse_variables()
                output, cell = self.__call__(inp, states)
                cells.append(cell)
                hiddens.append(output)
                cells = cells[1:self.K + 1]
                hiddens = hiddens[1:self.K + 1]
                states = (cells, hiddens)
                outputs.append(output)
        return outputs, states


    def dynamic_rnn_v2(self, inputs, states=None, scope=None):
        """
        :param inputs:Tensor List. shape is times*[batch_size, input_size]
                    `batch_size` will be preserved (known)
                    `times` sequence length
                    `input_size` must be static (known)
        :param states: None create zero, else use input
                        states contain cells and hiddens
                        cells and hiddens is a list, child_size element,
                        element shape is [batch_size, outputs_size]
        :param scope: a String
        :return: outputs: Tensor List, shape is times*[batch_size, outputs_size].
        """
        batch_size = inputs[0].get_shape().as_list()[0]
        input_size = inputs[0].get_shape().as_list()[1]
        outputs_size = self._num_units
        outputs = []
        with tf.variable_scope(scope or type(self).__name__):
            if states == None:
                cells = []
                hiddens = []
                for i in range(self.K):
                    cell = tf.zeros([batch_size, outputs_size])
                    hidden = tf.zeros([batch_size, outputs_size])
                    cells.append(cell)
                    hiddens.append(hidden)
                states = (cells, hiddens)
            else:
                cells, hiddens = states
            for idx, inp in enumerate(inputs):
                if idx > 0: tf.get_variable_scope().reuse_variables()
                output, cell = self.__call__(inp, states)
                cells.append(cell)
                hiddens.append(output)
                cells = cells[1:self.K + 1]
                hiddens = hiddens[1:self.K + 1]
                states = (cells, hiddens)
                outputs.append(output)
        return outputs, states
