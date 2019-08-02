import tensorflow as tf


import numpy as np
import argparse

import sys,os
sys.path.insert(1, os.path.join(sys.path[0], 'helpers'))
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from train_DAE import decode

from scipy.signal import savgol_filter


class DAE:
    """ Denoising Autoendoder (DAE)

    More details about the network in the original paper:
    http://www.jmlr.org/papers/v11/vincent10a.html

    The user specifies the structure of this network
    by specifying number of inputs, the number of hidden
    units for each layer and the number of final outputs.
    All this information is set in the utils/flags.py file.

    The number of input neurons is defined as a frame_size*chunk_length,
    since it will take a time-window as an input

    """

    def __init__(self, shape, sess, args):
        """DAE initializer

        Args:
          shape:          list of ints specifying
                          num input, hidden1 units,...hidden_n units, num outputs
          sess:           tensorflow session object to use
          varience_coef:  multiplicative factor for the variance of noise wrt the variance of data
        """

        self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim,output_dim]
        self.__variables = {}
        self.__sess = sess

        self.num_hidden_layers = np.size(shape) - 2

        self.batch_size = args.batch_size
        self.sequence_length = 1

        self.scaling_factor = 1


        self.max_val = np.load(args.max_val_file)
        self.mean_pose = np.load(args.mean_pose_file)


        ### Specify tensorflow setup  ###
        with sess.graph.as_default():

            ##############        SETUP VARIABLES       ######################

            with tf.variable_scope("AE_Variables"):

                for i in range(self.num_hidden_layers + 1):  # go over layers

                    # create variables for matrices and biases for each layer
                    self._create_variables(i, args.weight_decay)

                ##############        DEFINE THE NETWORK     ##################

                ''' 1 - Setup network for TRAINing '''
                # Input noisy data and reconstruct the original one
                # as in Denoising AutoEncoder
                self._input_ = tf.placeholder(dtype=tf.float32,
                                                      shape=[args.batch_size, shape[0]])
                #self._train_batch # add_noise(self._train_batch, variance_coef, data_info.data_sigma)
                self._target_ = self._input_

                # Define output and loss for the training data
                self._output, self._encode, self._decode = self.construct_graph(self._input_, args.dropout)
                self._reconstruction_loss = loss_reconstruction(self._output,
                                                                self._target_, self.max_val)
                tf.add_to_collection('losses', self._reconstruction_loss)  # add weight decay loses
                self._loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

                """ ''' 2 - Setup network for TESTing '''
                self._valid_input_ = self._valid_batch
                self._valid_target_ = self._valid_batch

                # Define output (no dropout)
                self._valid_output, self._encode, self._decode = \
                    self.construct_graph(self._valid_input_, 1)

                # Define loss
                self._valid_loss = loss_reconstruction(self._valid_output,
                                                       self._valid_target_, self.max_val) """

    def _create_variables(self, i, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if 'wd' is specified.
        If 'wd' is None, weight decay is not added for this Variable.

        This function was taken from the web

        Args:
          i: number of hidden layer
          wd: add L2Loss weight decay multiplied by this float.
        Returns:
          Nothing
        """

        # Initialize Train weights
        w_shape = (self.__shape[i], self.__shape[i + 1])
        a = tf.multiply(2.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
        name_w = "matrix"+str(i + 1)
        self[name_w] = tf.get_variable("Variables/"+name_w,
                                       initializer=tf.random_uniform(w_shape, -1 * a, a))

        # Add weight to the loss function for weight decay
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(self[name_w]), wd, name='wgt_'+str(i)+'_loss')
            tf.add_to_collection('losses', weight_decay)

        # Add the histogram summary
        tf.summary.histogram(name_w, self[name_w])

        # Initialize Train biases
        name_b = "bias"+str(i + 1)
        b_shape = (self.__shape[i + 1],)
        self[name_b] = tf.get_variable("Variables/"+name_b, initializer=tf.zeros(b_shape))

    def __getitem__(self, item):
        """Get AutoEncoder tf variable

        Returns the specified variable created by this object.
        Names are weights#, biases#, biases#_out, weights#_fixed,
        biases#_fixed.

        Args:
         item: string, variables internal name
        Returns:
         Tensorflow variable
        """
        return self.__variables[item]

    def __setitem__(self, key, value):
        """Store a TensorFlow variable

        NOTE: Don't call this explicitly. It should
        be used only internally when setting up
        variables.

        Args:
          key: string, name of variable
          value: tensorflow variable
        """
        self.__variables[key] = value

    @property
    def session(self):
        """ Interface for the session"""
        return self.__sess

    # Make more comfortable interface to the network weights

    def _w(self, n, suffix=""):
        return self["matrix"+str(n)+suffix]

    def _b(self, n, suffix=""):
        return self["bias"+str(n)+suffix]

    @staticmethod
    def _feedforward(x, w, b):
        """
        Traditional feedforward layer: multiply on weight matrix, add bias vector
         and apply activation function

        Args:
            x: input ( usually - batch of vectors)
            w: matrix to be multiplied on
            b: bias to be added

        Returns:
            y: result of applying this feedforward layer
        """

        y = tf.tanh(tf.nn.bias_add(tf.matmul(x, w), b))
        return y

    def construct_graph(self, input_seq_pl, dropout):

        """ Construct a TensorFlow graph for the AutoEncoding network

        Args:
          input_seq_pl:     tf placeholder for input data: size [batch_size, sequence_length * DoF]
          dropout:          how much of the input neurons will be activated, value in range [0,1]
        Returns:
          output:           output tensor: result of running input placeholder through the network
          middle_layer:     tensor which is encoding input placeholder into a representation
          decoding:         tensor which is decoding a representation back into the input vector
        """

        network_input = input_seq_pl

        curr_layer = tf.reshape(network_input, [self.batch_size, self.__shape[0]])

        numb_layers = self.num_hidden_layers + 1

        with tf.name_scope("Joint_run"):

            # Pass through the network
            for i in range(numb_layers):

                if i == 1:
                    # Save middle layer
                    with tf.name_scope('middle_layer'):
                        middle_layer = tf.identity(curr_layer)

                with tf.name_scope('hidden'+str(i)):

                    # First - Apply Dropout
                    curr_layer = tf.nn.dropout(curr_layer, dropout)

                    w = self._w(i + 1)
                    b = self._b(i + 1)

                    curr_layer = self._feedforward(curr_layer, w, b)

            output = curr_layer

        # Now create a decoding network

        with tf.name_scope("Decoding"):

            layer = self._representation = tf.placeholder\
                (dtype=tf.float32, shape=middle_layer.get_shape().as_list(), name="Respres.")

            for i in range(1, numb_layers):

                with tf.name_scope('hidden' + str(i)):

                    # First - Apply Dropout
                    layer = tf.nn.dropout(layer, dropout)

                    w = self._w(i + 1)
                    b = self._b(i + 1)

                    layer = self._feedforward(layer, w, b)

            decoding = layer

        return output, middle_layer, decoding


def load_dae(shape, args):
    """ Training of the network

    Args:
        shape: shape of DAE
        args:  various arguments

    Returns:
        nn:             Neural Network restored from a given checkpoint
    """


    with tf.Graph().as_default():


        # Allow TensorFlow to change device allocation when needed
        config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True)
        # Adjust configuration so that multiple executions are possible
        config.gpu_options.allow_growth = True

        # Start a session
        sess = tf.Session(config=config)

        # Create a neural network

        nn = DAE(shape, sess, args)
        print('\nDAE with the following shape was created : ', shape)

        # Initialize input_producer
        sess.run(tf.local_variables_initializer())

        max_val = nn.max_val

        with tf.variable_scope("Restore"):


            print("Initializing variables ...\n")
            sess.run(tf.global_variables_initializer())

            # Create a saver
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            chkpt_file = args.checkpoint_dir + '/chkpt-final'

            # restore model, if needed
            saver.restore(sess, chkpt_file)
            print("Model restored from the file " + str(chkpt_file) + '.')

            return nn


def smoothing(motion):

    smoothed = [savgol_filter(motion[:,i], 7, 3) for i in range(motion.shape[1])]

    new_motion = np.array(smoothed).transpose()

    return new_motion

def loss_reconstruction(output, target, max_vals, pretrain=False):
    """ Reconstruction error. Square of the RMSE

    Args:
      output:    tensor of net output
      target:    tensor of net we are trying to reconstruct
      max_vals:  array of absolute maximal values in the dataset,
                is used for scaling an error to the original space
      pretrain:  wether we are using it during the pretraining phase
    Returns:
      Scalar tensor of mean squared Eucledean distance
    """
    with tf.name_scope("reconstruction_loss"):
        net_output_tf = tf.convert_to_tensor(tf.cast(output, tf.float32), name='input')
        target_tf = tf.convert_to_tensor(tf.cast(target, tf.float32), name='target')

        # Euclidean distance between net_output_tf,target_tf
        error = tf.subtract(net_output_tf, target_tf)

        if not pretrain:
            # Convert it back from the [-1,1] to original values
            error_scaled = tf.multiply(error, max_vals[np.newaxis, :] + 1e-15)
        else:
            error_scaled = error

        squared_error = tf.reduce_mean(tf.square(error_scaled, name="square"), name="averaging")
    return squared_error



if __name__ == '__main__':
    # Parse command line params



    parser = argparse.ArgumentParser(
        description='Decode "z" into the motion using Denoising Autoencoder')
    # Model params
    parser.add_argument('--checkpoint_dir', '-chkp', default="models/DAE_checkpoints",
                        help='Variance of the noise to be injected to DAE')
    parser.add_argument('--variance', '-v', default=0.2,
                        help='Variance of the noise to be injected to DAE')
    parser.add_argument('--batch_size', '-bt_sz', default=8,
                        help='Batch size')
    parser.add_argument('--weight_decay', '-wd', default=0.5,
                        help='Weight decay coefficient')
    parser.add_argument('--dropout', '-drop', default=0.8,
                        help='Dropout coefficient')
    # Dataset params
    parser.add_argument('--max_val_file', '-max_f', default="models/max_val.npy",
                        help='Address to a file with maximal values in the dataset (for denormalization)')
    parser.add_argument('--mean_pose_file', '-mean_f', default="models/mean_pose.npy",
                        help='Address to a file with mean values in the dataset (for denormalization)')

    # Input / output files
    parser.add_argument('--encoding_file', '-enc_f', default="data/encoded_motion.txt",
                        help='Address to a file with encoded motion (result of the "predict" script')
    parser.add_argument('--decoded_file', '-dec_f', default="result/gestures.txt",
                        help='Address to where we want to save the decoded motion')

    args = parser.parse_args()

    shape = [138, 112, 138]
    dae = load_dae(shape, args)

    # read encoded motion
    encoding = np.loadtxt(args.encoding_file)

    # Decode the motion
    decoding = decode(dae, encoding)

    print(decoding.shape)

    # Smoothen it
    skeletons = smoothing(decoding)

    np.savetxt(args.decoded_file, skeletons, delimiter=' ')

    # Close Tf session
    dae.session.close()
