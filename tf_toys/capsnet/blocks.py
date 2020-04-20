#-*- coding:utf-8 -*-
import tensorflow as tf

def squash(s, axis=-1, epsilon=1e-7):
  squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
  safe_norm = tf.sqrt(squared_norm + epsilon)
  squash_factor = squared_norm / (1. + squared_norm)
  unit_vector = s / safe_norm
  return squash_factor * unit_vector

def safe_norm(s, axis=-1, epsilon=1e-7, keepdims=False):
  squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims)
  return tf.sqrt(squared_norm + epsilon)

class CapsuleNetwork(tf.keras.layers.Layer):
  def __init__(self):
    super(CapsuleNetwork, self).__init__()
    self.caps1_n = caps1_n = 32
    self.caps2_n = caps2_n = 10
    self.caps1_dim = caps1_dim = 8
    self.caps2_dim = caps2_dim = 16
    self.caps_n = caps_n = caps1_n * 6 * 6
    ks = 9

    self.conv1 = tf.keras.layers.Conv2D(filters=256,
                                        kernel_size=[ks, ks], strides=[1, 1],
                                        padding="valid", activation="relu")

    self.conv2 = tf.keras.layers.Conv2D(filters=caps1_n * caps1_dim,
                                        kernel_size=[ks, ks], strides=[2, 2],
                                        padding="valid", activation="relu")
    #init_sigma = 0.1
    #self.wgt = tf.Variable(tf.random.normal([1, caps_n, caps2_n, caps2_dim,
    #                                         caps1_dim], stddev=init_sigma),
    #                       trainable=True, name="W")

    #self.reconstructor = ReconstructionLayer()
    init_sigma = 0.1
    self.wgt = tf.Variable(tf.random.normal([1, caps_n, caps2_n, caps2_dim,
                                             caps1_dim], stddev=init_sigma),
                           trainable=True, name="W")

  def call(self, inputs, **kwargs):
    image, target = inputs
    caps1_n_maps = 32 # number of capsule maps
    caps2_n_caps = 10 # number of classes
    caps1_n_caps = caps1_n_maps * 6 * 6
    caps1_n_dims = 8
    is_train = kwargs["training"]
    self.caps1_n = caps1_n = 32
    self.caps2_n = caps2_n = 10
    self.caps1_dim = caps1_dim = 8
    self.caps2_dim = caps2_dim = 16
    self.caps_n = caps_n = caps1_n * 6 * 6

    batch_size = tf.shape(image)[0]
    conv1 = self.conv1(image)
    conv2 = self.conv2(conv1)
    caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims],
                           name="caps1_raw")
    caps1_output = squash(caps1_raw)


    W_tiled = tf.tile(self.wgt, [batch_size, 1, 1, 1, 1], name="W_tiled")
    caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                           name="caps1_output_expanded")
    caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                       name="caps1_output_tile")
    caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                                 name="caps1_output_tiled")
    u_hat = tf.matmul(W_tiled, caps1_output_tiled, name="u_hat")

    # ==========================================================================
    # Dynamic Routing
    # ==========================================================================
    raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                           dtype=tf.float32, name="raw_weights")

    # Round1, Line 4
    routing_weights = tf.nn.softmax(raw_weights, axis=2, name="routing_weights")
    # Round1, Line 5
    weighted_predictions = tf.multiply(routing_weights, u_hat,
                                       name="weighted_predictions")
    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True,
                                 name="weighted_sum")
    # Round1, Line 6
    caps2_output_round_1 = squash(weighted_sum, axis=-2)
    # Round1, Line 7
    caps2_output_round_1_tiled = tf.tile(caps2_output_round_1,
                                         [1, caps1_n_caps, 1, 1, 1],
                                         name="caps2_output_round_1_tiled")
    raw_weights2 = raw_weights + tf.matmul(u_hat, caps2_output_round_1_tiled,
                                           transpose_a=True)

    # Round2, Line 4
    routing_weights_round_2 = tf.nn.softmax(raw_weights2, axis=2,
                                            name="routing_weights_round_2")
    # Round2, Line 5
    weighted_predictions_round_2 = tf.multiply(routing_weights_round_2, u_hat,
                                               name="weighted_predictions_round_2")
    weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2, axis=1,
                                         keepdims=True,
                                         name="weighted_sum_round_2")
    # Round2, Line 6
    caps2_output = squash(weighted_sum_round_2, axis=-2)
    # ==========================================================================
    y_proba = tf.squeeze(safe_norm(caps2_output, axis=-2), 1)
    #y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
    #y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2], name="y_pred")
    """
    # Reconstruction
    recon_tgts = tf.cast(target, tf.int32) if is_train else y_pred
    recon_mask = tf.reshape(tf.one_hot(recon_tgts, depth=caps2_n),
                            [-1, 1, caps2_n, 1, 1], name="recon_mask")
    masked_v2 = tf.multiply(caps2_output, recon_mask, name="masked_round_2_out")
    dec_out = self.reconstructor(masked_v2)
    """
    caps2_output_norm = safe_norm(caps2_output, axis=-2, keepdims=True)
    return caps2_output_norm, y_proba#, dec_out


class ReconstructionLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(ReconstructionLayer, self).__init__()
    self.hiddens = None
    self.decoder_output = None
    n_output = 28 * 28
    n_caps2 = 10
    caps2_n_dims = 16

    self.reshape = tf.keras.layers.Reshape([-1, n_caps2 * caps2_n_dims],
                                         name="inp_dec")
    self.hidden1 = tf.keras.layers.Dense(512, activation="relu")
    self.hidden2 = tf.keras.layers.Dense(1024, activation="relu")
    self.decoder_output = tf.keras.layers.Dense(n_output, activation="sigmoid",
                                           name="decoder_output")

  def call(self, inputs, **kwargs):
    inputs = self.reshape(inputs)
    inputs = self.hidden1(inputs)
    inputs = self.hidden2(inputs)
    return self.decoder_output(inputs)
