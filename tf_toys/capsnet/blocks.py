#-*- coding:utf-8 -*-
import tensorflow as tf


def squash(s, axis=-1, epsilon=1e-7):
  # for better numerical stability
  squared_norm = tf.math.reduce_sum(tf.square(s), axis=axis, keepdims=True)
  safe_norm = tf.sqrt(squared_norm + epsilon)
  squash_factor = squared_norm / (1. + squared_norm)
  unit_vector = s / safe_norm
  return squash_factor * unit_vector

def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
  with tf.name_scope(name):
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keep_dims)
    return tf.sqrt(squared_norm + epsilon)

class CapsuleNetwork(tf.keras.layers.Layer):
  def __init__(self):
    super(CapsuleNetwork, self).__init__()
    self.capsuler = CapsulizationLayer()
    self.reconstructor = ReconstructionLayer()

  def call(self, inputs, **kwargs):
    image, target = inputs
    caps1_n_maps = 32 # number of capsule maps
    caps2_n = 10 # number of classes
    n_caps1 = caps1_n_maps * 6 * 6
    is_train = kwargs["training"]

    batch_size = tf.shape(image)[0]
    # [batch, pc width, pc height, n_maps * n_dims]
    u_hat = self.capsuler(image)

    # ==========================================================================
    # Dynamic Routing
    # ==========================================================================
    # Line2: b: initialization
    raw_wgts = tf.zeros([batch_size, n_caps1, caps2_n, 1, 1], dtype=tf.float32,
                        name="raw_wgts")

    # Round1
    # Line4: for all capsule i in layer l: c_i <- softmax(b_i)
    # softmax over caps2_n
    c_1 = tf.nn.softmax(raw_wgts, axis=2, name="routing_wgts_1")
    # Line5: for all capsule j in layer (l + 1): s_j <- sum_i c_ij * u_hat_j|i
    each_s_1 = tf.multiply(c_1, u_hat)
    s_1 = tf.reduce_sum(each_s_1, axis=1, keepdims=True,
                        name="wgted_sum_1")
    # Line6: for all capsule j in layer (l + 1): v_j <- squash(s_j)
    v_1 = squash(s_1, axis=-2)
    # Line 7: for all capsule i in layer l and capsule j in layer (l+1):
    #         b_ij <- b_ij + u_hat_j_i x v_j
    v_1_tiled = tf.tile(v_1, [1, n_caps1, 1, 1, 1], name="round_1_out_tiled")
    uv_1 = tf.matmul(u_hat, v_1_tiled, transpose_a=True, name="agreement")
    raw_wgts2 = raw_wgts + uv_1

    # Round2
    # Line4: for all capsule i in layer l: c_i <- softmax(b_i)
    c_2 = tf.nn.softmax(raw_wgts2, axis=2, name="routing_wgts_2")
    # Line5: for all capsule j in layer (l + 1): s_j <- sum_i c_ij * u_hat_j|i
    s_2 = tf.reduce_sum(tf.multiply(c_2, u_hat), axis=1, keepdims=True,
                        name="wgted_sum_2")
    # Line6: for all capsule j in layer (l + 1): v_j <- squash(s_j)
    # tf.print("s_2 at round 2:", tf.shape(s_2))
    v_2 = squash(s_2, axis=-2)
    # Line7: won't be used the updated raw_wgts for next iteration
    #===========================================================================

    # [batch, 1, caps2_n, 1, 1]
    norm_v2 = safe_norm(v_2, axis=-2, keep_dims=True, name="normed_round_2_out")
    max_args = tf.argmax(norm_v2, axis=2)
    # [batch, 1]
    y_pred = tf.squeeze(max_args, axis=[1, 2], name="y_pred")

    # Reconstruction
    recon_tgts = tf.cast(target, tf.int32) if is_train else y_pred
    recon_mask = tf.reshape(tf.one_hot(recon_tgts, depth=caps2_n),
                            [-1, 1, caps2_n, 1, 1], name="recon_mask")
    masked_v2 = tf.multiply(v_2, recon_mask, name="masked_round_2_out")
    dec_out = self.reconstructor(masked_v2)

    return norm_v2, y_pred, dec_out


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


class CapsulizationLayer(tf.keras.layers.Layer):
  def __init__ (self):
    super(CapsulizationLayer, self).__init__()
    self.caps1_n = caps1_n = 32
    self.caps2_n = caps2_n = 10
    self.caps1_dim = caps1_dim = 8
    self.caps2_dim = caps2_dim = 16
    self.caps_n = caps_n = caps1_n * 6 * 6
    ks = 9
    init_sigma = 0.1

    # caps1_n = 32 * caps1_wth * caps1_hgt, caps2_n = 10,
    self.wgt = tf.Variable(tf.random.normal([1, caps_n, caps2_n, caps2_dim,
                                             caps1_dim], stddev=init_sigma),
                           trainable=True, name="W")
    self.cnns = list()
    self.cnns.append(tf.keras.layers.Conv2D(filters=256,
                                            kernel_size=[ks, ks], strides=[1, 1],
                                            padding="valid", activation="relu"))

    self.cnns.append(tf.keras.layers.Conv2D(filters=caps1_n * caps1_dim,
                                            kernel_size=[ks, ks], strides=[2, 2],
                                            padding="valid", activation="relu"))

  def call(self, inputs, **kwargs):
    batch_size = tf.shape(inputs)[0]
    for cnn in self.cnns:
      inputs = cnn(inputs)

    # [batch, pc width * pc height * n_caps, n_dims]: Primary Capsules
    inputs = tf.reshape(inputs, [-1, self.caps_n, self.caps1_dim])
    pc = squash(inputs)
    pc_tiled = tf.tile(tf.expand_dims(tf.expand_dims(pc, -1), 2),
                       [1, 1, self.caps2_n, 1, 1], name="caps1_output_tiled")
    W_tiled = tf.tile(self.wgt, [batch_size, 1, 1, 1, 1], name="W_tiled")
    result = tf.matmul(W_tiled, pc_tiled, name="digit_caps")

    return result
