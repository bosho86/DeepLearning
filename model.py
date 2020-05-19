import tensorflow as tf
import numpy as np

class DNNModel(object):
    """
    Creates training and validation computational graphs.
    Note that tf.variable_scope enables parameter sharing so that both graphs are identical.
    """
    def __init__(self, config, placeholders, mode):
        """
        Basic setup.
        :param config: configuration dictionary
        :param placeholders: dictionary of input placeholders
        :param mode: training, validation or inference
        """
        assert mode in ['training', 'validation', 'inference']
        self.delay = config['delay']
        self.config = config
        self.input_ = placeholders['input_pl']
        self.target = placeholders['target_pl']
        self.out_last_last =  self.input_
        self.out_last = self.input_
        self.out = self.input_
        self.mode = mode
        self.is_training = self.mode == 'training'
        self.reuse = ((self.mode == 'validation') or (self.mode == 'inference'))
        self.batch_size = tf.shape(self.input_)[0]  # dynamic size
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.length_split = config['length_split']
        self.summary_collection = 'training_summaries' if mode == 'training' else 'validation_summaries'

    def build_graph(self):
        self.build_model()
        self.build_loss()
        self.count_parameters()

    def build_model(self):
        """
        Builds the actual model.
        """

        with tf.variable_scope('rnn_model', reuse=self.reuse): 
            x = tf.layers.dense(self.input_, units=int(self.input_dim), reuse=self.reuse, name='layer1')
            x = tf.nn.relu(x)
            x = tf.layers.dense(x, units=int(self.input_dim/4), reuse=self.reuse, name='layer2')
            x = tf.nn.relu(x)
            x = tf.layers.dense(x, units=int(self.input_dim/16), reuse=self.reuse, name='layer3')
            x = tf.nn.relu(x)
            x = tf.layers.dropout(x, rate=0.1, training=self.is_training)
            x = tf.layers.dense(x, units=self.output_dim, reuse=self.reuse, name='layer4')
            x = tf.nn.relu(x)
            outputs = tf.nn.softmax(x, name='softmaxLayer')
            self.prediction = outputs
            

    def build_loss(self):
        """
        Builds the loss function.
        """
        # only need loss if we are not in inference mode
        if self.mode != 'inference':
            with tf.name_scope('loss'):
                self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.target, logits=self.prediction)
               
                
                tf.summary.scalar('loss', self.loss, collections=[self.summary_collection])
        
        if self.mode == 'validation':
            with tf.name_scope('loss'):
  
                self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.target, logits=self.prediction)
                

                

    def count_parameters(self):
        """
        Counts the number of trainable parameters in this model
        """
        self.n_parameters = 0
        for v in tf.trainable_variables():
            params = 1
            for s in v.get_shape():
                params *= s.value
            self.n_parameters += params



class CNNModel(object):
    """
    Creates training and validation computational graphs.
    Note that tf.variable_scope enables parameter sharing so that both graphs are identical.
    """
    def __init__(self, config, placeholders, mode):
        """
        Basic setup.
        :param config: configuration dictionary
        :param placeholders: dictionary of input placeholders
        :param mode: training, validation or inference
        """
        assert mode in ['training', 'validation', 'inference']
        self.delay = config['delay']
        self.config = config
        self.input_ = placeholders['input_pl']
        self.target = placeholders['target_pl']
        self.out_last_last =  self.input_
        self.out_last = self.input_
        self.out = self.input_
        self.mode = mode
        self.is_training = self.mode == 'training'
        self.reuse = ((self.mode == 'validation') or (self.mode == 'inference'))
        self.batch_size = tf.shape(self.input_)[0]  # dynamic size
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.length_split = config['length_split']
        self.summary_collection = 'training_summaries' if mode == 'training' else 'validation_summaries'

    def build_graph(self):
        self.build_model()
        self.build_loss()
        self.count_parameters()

    def build_model(self):
        """
        Builds the actual model.
        """

        with tf.variable_scope('cnn_model', reuse=self.reuse):
            x = tf.expand_dims(self.input_, -1)
            x = tf.layers.batch_normalization(x, training=self.is_training, momentum=0.99)
            x = tf.layers.conv1d(x, filters=3, kernel_size=5, strides=1, \
                                  data_format='channels_last', reuse=self.reuse, name='layer1')
            x = tf.nn.relu(x)
            x = tf.layers.max_pooling1d(x, pool_size=3, strides=2, data_format='channels_last')
            x = tf.layers.dropout(x, rate=0.1, training=self.is_training)
            x = tf.layers.batch_normalization(x, training=self.is_training, momentum=0.99)
            x = tf.layers.conv1d(x, filters=5, kernel_size=5, strides=1, \
                                  data_format='channels_last', reuse=self.reuse, name='layer2')
            x = tf.nn.relu(x)
            x = tf.layers.max_pooling1d(x, pool_size=5, strides=2, data_format='channels_last') 
        
            x = tf.layers.batch_normalization(x, training=self.is_training, momentum=0.99)
            x = tf.layers.conv1d(x, filters=10, kernel_size=5, strides=1, \
                                  data_format='channels_last', reuse=self.reuse, name='layer3')
            x = tf.nn.relu(x)
            x = tf.layers.max_pooling1d(x, pool_size=10, strides=2, data_format='channels_last') 
            x = tf.layers.dropout(x, rate=0.1, training=self.is_training)
            x = tf.layers.batch_normalization(x, training=self.is_training, momentum=0.99)
            x = tf.layers.conv1d(x, filters=10, kernel_size=4, strides=1, \
                                  data_format='channels_last', reuse=self.reuse, name='layer4')
            x = tf.nn.relu(x)
            x = tf.layers.max_pooling1d(x, pool_size=10, strides=2, data_format='channels_last')
   
            x = tf.contrib.layers.flatten(x)
            # dense layers
            x = tf.layers.dense(x, units=20, reuse=self.reuse, name='layer7')
            x = tf.nn.relu(x)
            x = tf.layers.dense(x, units=10, reuse=self.reuse, name='layer8')
            x = tf.nn.relu(x)
            x = tf.layers.dense(x, units=self.output_dim, reuse=self.reuse, name='layer9')
            x = tf.nn.relu(x)
            outputs = tf.nn.softmax(x, name='softmaxLayer')
            self.prediction = outputs
            

    def build_loss(self):
        """
        Builds the loss function.
        """
        # only need loss if we are not in inference mode
        if self.mode != 'inference':
            with tf.name_scope('loss'):
                # squared difference
                self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.target, logits=self.prediction)
                # L1 loss
                
                tf.summary.scalar('loss', self.loss, collections=[self.summary_collection])
        
        if self.mode == 'validation':
            with tf.name_scope('loss'):
                self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.target, logits=self.prediction)

                
                

    def count_parameters(self):
        """
        Counts the number of trainable parameters in this model
        """
        self.n_parameters = 0
        for v in tf.trainable_variables():
            params = 1
            for s in v.get_shape():
                params *= s.value
            self.n_parameters += params
