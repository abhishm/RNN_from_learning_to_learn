import numpy as np

class SequentialData(object):
    def __init__(self, data_size, num_classes, batch_size, num_steps):
        """Set the parameters for generating the data using a predefined rule
        """
        self.batch_size = batch_size
        self.data_size = data_size
        self.num_classes = num_classes
        self.num_steps = num_steps

    def gen_data(self):
        """Return the X and Y sequence generated according a predefined rule
        """
        x = np.random.choice(2, size=(self.data_size,))
        y = []
        for i in xrange(self.data_size):
            threshold = 0.5
            if x[i - 3] == 1:
                threshold += 0.5
            if x[i - 4] == 1:
                threshold -= 0.25
            if np.random.rand() > threshold:
                y.append(0)
            else:
                y.append(1)
        return x, np.array(y)

    def gen_batch(self, data):
        """Return a generator for producing the input sequence and
        target sequence for rnn
        """
        x, y = data
        column_size = self.data_size // self.batch_size

        data_x = np.zeros((self.batch_size, column_size), dtype = np.int32)
        data_y = np.zeros((self.batch_size, column_size), dtype = np.int32)

        for i in xrange(self.batch_size):
            data_x[i] = x[i * column_size:(i + 1) * column_size]
            data_y[i] = y[i * column_size:(i + 1) * column_size]

        num_epochs = column_size // self.num_steps

        for i in xrange(num_epochs):
            x = data_x[:, i * self.num_steps:(i + 1) * self.num_steps]
            y = data_y[:, i * self.num_steps:(i + 1) * self.num_steps]
            yield (x, y)

    def gen_epoch(self, n):
        """Return a generator that gives data for N epochs
        """
        data = self.gen_data()
        for _ in xrange(n):
            yield self.gen_batch(data)
