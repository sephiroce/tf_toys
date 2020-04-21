import tensorflow as tf
import tensorflow.examples.tutorials.mnist as Mnist

caps1_n = 32
caps1_dim = 8
caps2_n = 10
caps2_dim = 16
exp_caps1_n = caps1_n * 6 * 6  # 1152 primary capsules
dec_out_dim = 28 * 28
init_sigma_for_W = 0.1

m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

conv1_params = {
    "filters": 256,
    "kernel_size": 9,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
}

conv2_params = {
    "filters": caps1_n * caps1_dim, # 256 convolutional filters
    "kernel_size": 9,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
}

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
    self.conv1 = tf.keras.layers.Conv2D(**conv1_params)
    self.conv2 = tf.keras.layers.Conv2D(**conv2_params)

    self.wgt = tf.Variable(tf.random.normal(shape=(1, exp_caps1_n, caps2_n, caps2_dim, caps1_dim),
                                            stddev=init_sigma_for_W,
                                            dtype=tf.float32),
                           trainable=True)

    self.dense1 = tf.keras.layers.Dense(512, activation="relu")
    self.dense2 = tf.keras.layers.Dense(1024, activation="relu")
    self.dense3 = tf.keras.layers.Dense(dec_out_dim, activation="sigmoid")

  def call(self, inputs, **kwargs):
    X = inputs[0]
    Y = inputs[1]
    batch_size = tf.shape(X)[0]

    # CapsuleizationLayer
    conv1 = self.conv1(X)
    conv2 = self.conv2(conv1)
    caps1_raw = tf.reshape(conv2, [-1, exp_caps1_n, caps1_dim])
    caps1_output = squash(caps1_raw)

    W_tiled = tf.tile(self.wgt, [batch_size, 1, 1, 1, 1])

    caps1_output_expanded = tf.expand_dims(caps1_output, -1)
    caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2)
    caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n, 1, 1])
    u_hat = tf.matmul(W_tiled, caps1_output_tiled)

    # ==========================================================================
    # Dynamic Routing
    # ==========================================================================
    raw_weights = tf.zeros([batch_size, exp_caps1_n, caps2_n, 1, 1], dtype=tf.float32)

    # Round1, Line 4
    routing_weights = tf.nn.softmax(raw_weights, axis=2)
    # Round1, Line 5
    weighted_predictions = tf.multiply(routing_weights, u_hat)
    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True)
    # Round1, Line 6
    caps2_output_round_1 = squash(weighted_sum, axis=-2)
    # Round1, Line 7
    caps2_output_round_1_tiled = tf.tile(caps2_output_round_1, [1, exp_caps1_n, 1, 1, 1])
    raw_weights2 = raw_weights + tf.matmul(u_hat, caps2_output_round_1_tiled, transpose_a=True)

    # Round2, Line 4
    routing_weights_round_2 = tf.nn.softmax(raw_weights2, axis=2)
    # Round2, Line 5
    weighted_predictions_round_2 = tf.multiply(routing_weights_round_2, u_hat)
    weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2, axis=1, keepdims=True)
    # Round2, Line 6
    caps2_output = squash(weighted_sum_round_2, axis=-2)
    # ==========================================================================

    y_proba = safe_norm(caps2_output, axis=-2)
    y_proba_argmax = tf.argmax(y_proba, axis=2)
    y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2])

    # Reconstruction Network
    reconstruction_targets = Y
    reconstruction_mask = tf.one_hot(reconstruction_targets, depth=caps2_n)
    reconstruction_mask_reshaped = tf.reshape(reconstruction_mask,
                                              [-1, 1, caps2_n, 1, 1])
    caps2_output_masked = tf.multiply(caps2_output,
                                      reconstruction_mask_reshaped)
    decoder_input = tf.reshape(caps2_output_masked,
                               [-1, caps2_n * caps2_dim])
    hidden1 = self.dense1(decoder_input)
    hidden2 = self.dense2(hidden1)
    dec_out = self.dense3(hidden2)

    return caps2_output, y_pred, dec_out

@tf.function
def train(inputs, model, optimizer):
  X = tf.reshape(tf.cast(inputs[0], tf.float32), [-1, 28, 28, 1])
  Y = tf.cast(inputs[1], tf.int64)

  with tf.GradientTape() as tape:
    caps2_output, y_pred, dec_out = model((X, Y))

    # Loss: Margin loss
    T = tf.one_hot(Y, depth=caps2_n)
    caps2_output_norm = safe_norm(caps2_output, axis=-2, keepdims=True)
    present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm))
    present_error = tf.reshape(present_error_raw, shape=(-1, 10))
    absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus))
    absent_error = tf.reshape(absent_error_raw, shape=(-1, 10))
    L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error)
    margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1))

    # Loss: Reconstruction loss
    X_flat = tf.reshape(X, [-1, dec_out_dim])
    squared_difference = tf.square(X_flat - dec_out)
    reconstruction_loss = tf.reduce_mean(squared_difference)

    # Final Loss
    alpha = 0.0005
    loss = margin_loss + alpha * reconstruction_loss
    correct = tf.equal(Y, y_pred)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return loss, accuracy

def main():
  #pylint: disable=too-many-locals
  tf.random.set_seed(42)

  mnist = Mnist.input_data.read_data_sets("data/")
  caps_net = CapsuleNetwork()
  opti = tf.keras.optimizers.Adam()

  batch_size = 50
  n_iterations_per_epoch = mnist.train.num_examples // batch_size

  for epoch in range(1, 5):
    epoch_loss_avg = tf.keras.metrics.Mean(name='train_loss')
    epoch_accuracy = tf.keras.metrics.Mean(name='train_acc')
    for idx in range(1, n_iterations_per_epoch + 1):
      X_batch, Y_batch = mnist.train.next_batch(batch_size)
      loss, acc = train((X_batch, Y_batch), caps_net, opti)
      epoch_loss_avg.update_state(loss)
      epoch_accuracy.update_state(acc)
      print('\rIteration: %d/%d loss %.3f accuracy %.3f' % (idx,
                                                          n_iterations_per_epoch,
                                                          loss, acc * 100.0),
            end="")
    print("Epoch %d loss %.3f, accuracy %.3f"%(epoch, epoch_loss_avg.result(),
                                               epoch_accuracy.result()))

if __name__ == "__main__":
  main()