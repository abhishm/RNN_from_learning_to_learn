import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

class BasicRNN(object):
    def __init__(self, state_size, num_steps, num_classes, learning_rate, summary_every=100):
        """Create a Basic RNN classfier with the given STATE_SIZE,
        NUM_STEPS, and NUM_CLASSES
        """
        self.state_size = state_size
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # tensorflow machinery
        self.session = tf.Session()
        self.summary_writer = tf.summary.FileWriter(os.path.join(os.getcwd(), "tensorboard/"))
        self.no_op = tf.no_op()

        # counters
        self.train_itr = 0

        # create and initialize variables
        self.create_graph()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.session.run(tf.variables_initializer(var_lists))

        # make sure all variables are initialized
        self.session.run(tf.assert_variables_initialized())

        # add the graph
        self.summary_writer.add_graph(self.session.graph)
        self.summary_every = summary_every

    def create_placeholders(self):
        self.input = tf.placeholder(tf.int32, shape=(None, self.num_steps), name="input")
        self.target = tf.placeholder(tf.int32, shape=(None, self.num_steps), name="target")
        self.init_state = tf.placeholder(tf.float32, shape=(None, self.state_size), name="init_state")
        self.input_one_hot = tf.one_hot(self.input, self.num_classes)  # one-hot encoding of the input
        self.rnn_inputs = tf.unpack(self.input_one_hot, axis=1)  # unpacking inuts to make a list

    def create_variables(self):
        """Create variables for one layer RNN and the softmax
        """
        with tf.variable_scope("rnn"):
            W = tf.get_variable("W", [self.num_classes + self.state_size, self.state_size])
            b = tf.get_variable("b", [self.state_size], initializer=tf.constant_initializer(0))

        with tf.variable_scope("softmax"):
            W_softmax = tf.get_variable("W_softmax", [self.state_size, self.num_classes])
            b_softmax = tf.get_variable("b_softmax", [self.num_classes],
                                       initializer=tf.constant_initializer(0))

    def rnn(self):
        """ multi step RNN using tensorflow api
        """
        with tf.name_scope("rnn"):
            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.state_size)
            self.outputs, self.final_state = tf.nn.rnn(rnn_cell, self.rnn_inputs, initial_state=self.init_state)

    def softmax_loss(self):
        """A softmax operations on the output of the RNN
        OUTPUTS: is a list of tensors representing the outut from each rnn step
        """
        with tf.variable_scope("softmax", reuse=True):
            W_softmax = tf.get_variable("W_softmax", [self.state_size, self.num_classes])
            b_softmax = tf.get_variable("b_softmax", [self.num_classes],
                                       initializer=tf.constant_initializer(0))
            logits = [tf.matmul(output, W_softmax) + b_softmax for output in self.outputs]
            self.probs = [tf.nn.softmax(logit) for logit in logits]
            losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logit, self.target[:, i]) for i, logit in
                        enumerate(logits)]
            self.loss = tf.reduce_mean(losses)

    def create_variables_for_optimizations(self):
        """create variables for optimizing
        """
        with tf.name_scope("optimization"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.gradients = self.optimizer.compute_gradients(self.loss, var_list=self.trainable_variables)
            self.train_op = self.optimizer.apply_gradients(self.gradients)

    def create_summaries(self):
        """create summary variables
        """
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.gradient_summaries = []
        for grad, var in self.gradients:
            if grad is not None:
                gradient_summary = tf.summary.histogram(var.name + "/gradient", grad)
                self.gradient_summaries.append(gradient_summary)
        self.weight_summaries = []
        weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for w in weights:
            weight_summary = tf.summary.histogram(w.name, w)
            self.weight_summaries.append(weight_summary)

    def merge_summaries(self):
        """Merge all sumaries
        """
        self.summarize = tf.summary.merge([self.loss_summary]
                                            + self.weight_summaries
                                            + self.gradient_summaries)

    def create_graph(self):
        self.create_placeholders()
        self.create_variables()
        self.rnn()
        self.softmax_loss()
        self.create_variables_for_optimizations()
        #self.create_summaries()
        #self.merge_summaries()

    def update_params(self, batch):
        """Given a batch of data, update the network to minimize the loss
        """
        write_summay = self.train_itr % self.summary_every == 0
        if not hasattr(self, "init_state_value"):
            self.init_state_value = np.zeros((len(batch[0]), self.state_size))
        _, self.init_state_value, summary = self.session.run([self.train_op,
                                        self.final_state,
                                        self.summarize if write_summay else self.no_op],
                                        feed_dict={self.input: batch[0],
                                                   self.target:batch[1],
                                                   self.init_state:self.init_state_value})
        if write_summay:
            self.summary_writer.add_summary(summary, self.train_itr)

        self.train_itr += 1
