#-*- coding:utf-8 -*-
import tensorflow as tf


def squash(s, axis=-1, epsilon=1e-7, name=None):
  with tf.name_scope(name):
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
    self.conv = ConvolutionalLayer()
    self.decs = DecoderLayers()
    self.caps = CapsuleLayer()

  def call(self, inputs, **kwargs):
    image, target = inputs
    caps1_n_maps = 32
    caps2_n = 10
    n_caps1 = caps1_n_maps * 6 * 6
    is_train = kwargs["training"]

    batch_size = tf.shape(image)[0]
    # [batch, pc width, pc height, n_maps * n_dims]
    u_hat = self.caps(self.conv(image))

    # Dynamic routing
    # Line2: b: initialization
    raw_wgts = tf.zeros([batch_size, n_caps1, caps2_n, 1, 1],
                          dtype=tf.float32, name="raw_wgts")

    # Round1
    # Line4: for all capsule i in layer l: c_i <- softmax(b_i)
    c_1 = tf.nn.softmax(raw_wgts, axis=2, name="routing_wgts_1")
    # Line5: for all capsule j in layer (l + 1): s_j <- sum_i c_ij * u_hat_j|i
    s_1 = tf.reduce_sum(tf.multiply(c_1, u_hat), axis=1, keepdims=True,
                        name="wgted_sum_1")
    # Line6: for all capsule j in layer (l + 1): v_j <- squash(s_j)
    v_1 = squash(s_1, axis=-2, name="round_1_out")

    # Line 7: for all capsule i in layer l and capsule j in layer (l+1):
    #         b_ij <- b_ij + u_hat_j_i x v_j
    v_1_tiled = tf.tile(v_1, [1, n_caps1, 1, 1, 1], name="round_1_out_tiled")
    uv_1 = tf.matmul(u_hat, v_1_tiled, transpose_a=True, name="agreement")
    raw_wgts += uv_1

    # Round2
    # Line4: for all capsule i in layer l: c_i <- softmax(b_i)
    c_2 = tf.nn.softmax(raw_wgts, axis=2, name="routing_wgts_2")
    # Line5: for all capsule j in layer (l + 1): s_j <- sum_i c_ij * u_hat_j|i
    s_2 = tf.reduce_sum(tf.multiply(c_2, u_hat), axis=1, keepdims=True,
                        name="wgted_sum_2")
    # Line6: for all capsule j in layer (l + 1): v_j <- squash(s_j)
    v_2 = squash(s_2, axis=-2, name="round_2_out")
    # Line7: won't be used the updated raw_wgts for next iteration

    norm_v2 = safe_norm(v_2, axis=-2, keep_dims=True, name="normed_round_2_out")
    y_pred = tf.squeeze(tf.argmax(norm_v2, axis=2), axis=[1, 2], name="y_pred")

    # Reconstruction
    recon_tgts = tf.cast(target, tf.int32) if is_train else y_pred
    recon_mask = tf.reshape(tf.one_hot(recon_tgts, depth=caps2_n),
                            [-1, 1, caps2_n, 1, 1], name="recon_mask")
    masked_v2 = tf.multiply(v_2, recon_mask, name="masked_round_2_out")
    dec_out = self.decs(masked_v2)

    return norm_v2, y_pred, dec_out


class ConvolutionalLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(ConvolutionalLayer, self).__init__()
    self.cnns = list()
    caps1_n_maps = 32
    caps1_n_dims = 8
    ks = 9

    self.cnns.append(tf.keras.layers.Conv2D(filters=caps1_n_maps * caps1_n_dims,
                                            kernel_size=[ks, ks], strides=[1, 1],
                                            padding="valid", activation="relu"))

    self.cnns.append(tf.keras.layers.Conv2D(filters=caps1_n_maps * caps1_n_dims,
                                            kernel_size=[ks, ks], strides=[2, 2],
                                            padding="valid", activation="relu"))

  def call(self, inputs, **kwargs):
    for cnn in self.cnns:
      inputs = cnn(inputs)
    return inputs


class DecoderLayers(tf.keras.layers.Layer):
  def __init__(self):
    super(DecoderLayers, self).__init__()
    self.hiddens = None
    self.decoder_output = None
    hidden_dims = [512, 1024]
    n_output = 28 * 28

    n_caps2 = 10
    caps2_n_dims = 16

    self.reshape = tf.keras.layers.Reshape([-1, n_caps2 * caps2_n_dims],
                                         name="inp_dec")
    self.hiddens = [tf.keras.layers.Dense(hidden_dim, activation="relu",
                                          name="hidden%d"%(idx + 1)) \
                    for idx, hidden_dim in enumerate(hidden_dims)]
    self.decoder_output = tf.keras.layers.Dense(n_output, activation="sigmoid",
                                           name="decoder_output")

  def call(self, inputs, **kwargs):
    inputs = self.reshape(inputs)

    for hidden in self.hiddens:
      inputs = hidden(inputs)

    return self.decoder_output(inputs)


class CapsuleLayer(tf.keras.layers.Layer):
  def __init__ (self):
    super(CapsuleLayer, self).__init__()

    caps1_n_maps = 32
    caps1_n_dims = 8
    caps2_n = 10
    caps2_dim = 16
    n_caps1 = caps1_n_maps * 6 * 6

    init_sigma = 0.1
    self.wgt = tf.Variable(tf.random.normal([1, n_caps1, caps2_n, caps2_dim,
                                             caps1_n_dims], stddev=init_sigma),
                           trainable=True, name="W")

  def call(self, inputs, **kwargs):
    caps2_n = 10
    batch_size = tf.shape(inputs)[0]
    caps1_n_maps = 32
    n_caps1 = caps1_n_maps * 6 * 6
    caps1_n_dims = 8
    # [batch, pc width * pc height * n_caps, n_dims]: Primary Capsules
    inputs = tf.reshape(inputs, [-1, n_caps1, caps1_n_dims])
    pc = squash(inputs, name="caps1_output")
    pc_tiled = tf.tile(tf.expand_dims(tf.expand_dims(pc, -1), 2),
                       [1, 1, caps2_n, 1, 1], name="caps1_output_tiled")
    W_tiled = tf.tile(self.wgt, [batch_size, 1, 1, 1, 1], name="W_tiled")
    return tf.matmul(W_tiled, pc_tiled, name="digit_caps")
