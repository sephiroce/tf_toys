import tensorflow as tf
import tensorflow.examples.tutorials.mnist as Mnist
import time
import matplotlib.pyplot as plt
import numpy as np

caps1_n = 32
caps1_dim = 8
caps2_n = 10
caps2_dim = 16
exp_caps1_n = caps1_n * 6 * 6  # 1152 primary capsules
dec_out_dim = 28 * 28
routing_iter = 2

def squash(s, axis=-1, epsilon=1e-7):
  squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
  safe_norm = tf.sqrt(squared_norm + epsilon)
  squash_factor = squared_norm / (1. + squared_norm)
  unit_vector = s / safe_norm
  return squash_factor * unit_vector

def safe_norm(s, axis=-1, epsilon=1e-7, keepdims=False):
  squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims)
  return tf.sqrt(squared_norm + epsilon)

class ReconstructionLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(ReconstructionLayer, self).__init__()
    self.dense1 = tf.keras.layers.Dense(512, activation="relu")
    self.dense2 = tf.keras.layers.Dense(1024, activation="relu")
    self.dense3 = tf.keras.layers.Dense(dec_out_dim, activation="sigmoid")

  def call(self, inputs, **kwargs):
    caps2_output, reconstruction_targets = inputs
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
    return dec_out

""" For while loop
def condition(u_hat, weight, counter, caps_output):
  return tf.less_equal(counter, routing_iter)

def loop_body(u_hat, raw_weights, counter, caps_output):
  # Round1, Line 4
  routing_weights = tf.nn.softmax(raw_weights, axis=2)
  # Round1, Line 5
  weighted_predictions = tf.multiply(routing_weights, u_hat)
  weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True)
  # Round1, Line 6
  caps2_output_round_1 = squash(weighted_sum, axis=-2)
  # Round1, Line 7
  caps2_output_round_1_tiled = tf.tile(caps2_output_round_1,
                                       [1, exp_caps1_n, 1, 1, 1])
  raw_weights2 = raw_weights + tf.matmul(u_hat, caps2_output_round_1_tiled,
                                         transpose_a=True)
  return u_hat, raw_weights2, tf.add(counter, 1), caps2_output_round_1
"""

class CapsuleNetwork(tf.keras.layers.Layer):
  def __init__(self):
    super(CapsuleNetwork, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[9, 9],
                                        strides=[1, 1], padding="valid",
                                        activation="relu")
    self.conv2 = tf.keras.layers.Conv2D(filters=caps1_n * caps1_dim,
                                        kernel_size=[9, 9], strides=[2, 2],
                                        padding="valid", activation="relu")
    self.wgt = tf.Variable(tf.random.normal(shape=(1, exp_caps1_n, caps2_n, caps2_dim, caps1_dim),
                                            stddev=0.1, dtype=tf.float32),
                           trainable=True)
    self.recon_ = ReconstructionLayer()

  @property
  def recon(self):
    return self.recon_

  def call(self, inputs, **kwargs):
    X = inputs[0]
    Y = inputs[1]
    batch_size = tf.shape(X)[0]
    training = kwargs["training"]

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
    raw_weights = tf.zeros([batch_size, exp_caps1_n, caps2_n, 1, 1],
                            dtype=tf.float32)
    #dummy_value = tf.zeros([batch_size, 1, caps2_n, caps2_dim, 1])
    #counter = tf.constant(1)
    #_, _, _, caps2_output1 = tf.while_loop(condition, loop_body,
    #                                      [u_hat, raw_weights1, counter,
    #                                       dummy_value])
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
    #total_dim = tf.cast(tf.reduce_prod(tf.shape(caps2_output)), tf.int64)
    #tf.print("NOT EQUAL:", total_dim - tf.math.count_nonzero(tf.equal(
    #  caps2_output, caps2_output1)))
    # ==========================================================================

    y_proba = safe_norm(caps2_output, axis=-2)
    y_proba_argmax = tf.argmax(y_proba, axis=2)
    y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2])

    # Reconstruction Network
    dec_out = self.recon_((caps2_output, Y if training else y_pred))
    return caps2_output, dec_out, y_pred

def loss_function(X, Y, caps2_output, dec_out, y_pred):
  # Loss: Margin loss
  m_plus, m_minus, lambda_ = 0.9, 0.1, 0.5
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
  return loss, accuracy

@tf.function
def train(inputs, model, optimizer):
  X = tf.reshape(tf.cast(inputs[0], tf.float32), [-1, 28, 28, 1])
  Y = tf.cast(inputs[1], tf.int64)

  with tf.GradientTape() as tape:
    caps2_output, dec_out, y_pred = model((X, Y), training=True)
    loss, accuracy = loss_function(X, Y, caps2_output, dec_out, y_pred)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return loss, accuracy

@tf.function
def evaluation(inputs, model):
  X = tf.reshape(tf.cast(inputs[0], tf.float32), [-1, 28, 28, 1])
  Y = tf.cast(inputs[1], tf.int64)

  caps2_output, dec_out, y_pred = model((X, None), training=False)
  return loss_function(X, Y, caps2_output, dec_out, y_pred)

@tf.function
def pred(inputs, model):
  X = tf.reshape(tf.cast(inputs, tf.float32), [-1, 28, 28, 1])
  return model((X, None), training=False)

@tf.function
def recon(inputs, model):
  return model.recon(inputs)

def tweak_pose_parameters(output_vectors, min=-0.5, max=0.5, n_steps=11):
  """
  It will tweak each of the 16 pose parameters (dimensions) in all output
  vectors. Each tweaked output vector will be identical to the original
  output vector, except that one of its pose parameters will be incremented
  by a value varying from -0.5 to 0.5. By default there will be 11 steps (
  -0.5, -0.4, ..., +0.4, +0.5). This function will return an array of shape (
  tweaked pose parameters=16, steps=11, batch size=5, 1, 10, 16, 1)

  :param output_vectors:
  :param min:
  :param max:
  :param n_steps:
  :return:
  """
  steps = np.linspace(min, max, n_steps)  # -0.25, -0.15, ..., +0.25
  pose_parameters = np.arange(caps2_dim)  # 0, 1, ..., 15
  tweaks = np.zeros([caps2_dim, n_steps, 1, 1, 1, caps2_dim, 1])
  tweaks[pose_parameters, :, 0, 0, 0, pose_parameters, 0] = steps
  output_vectors_expanded = output_vectors[np.newaxis, np.newaxis]
  return tweaks + output_vectors_expanded

def main():
  #pylint: disable=too-many-locals
  tf.random.set_seed(42)

  mnist = Mnist.input_data.read_data_sets("data/")
  caps_net = CapsuleNetwork()
  opti = tf.keras.optimizers.Adam()

  batch_size = 50
  n_iterations_per_epoch = mnist.train.num_examples // batch_size
  n_iterations_validation = mnist.validation.num_examples // batch_size
  n_iterations_test = mnist.test.num_examples // batch_size
  best_loss_val = 1e14

  epoch = 1

  # Training
  for epoch in range(1, epoch + 1):
    train_avg_loss = tf.keras.metrics.Mean(name='train_loss')
    train_avg_accu = tf.keras.metrics.Mean(name='train_acc')
    start_time = time.time()
    for idx in range(1, n_iterations_per_epoch + 1):
      X_batch, Y_batch = mnist.train.next_batch(batch_size)
      loss, acc = train((X_batch, Y_batch), caps_net, opti)
      train_avg_loss.update_state(loss)
      train_avg_accu.update_state(acc)
      print('\rTraining the model: %d/%d Loss %.3f Acc %.3f' %
            (idx, n_iterations_per_epoch, loss, acc * 100.0), end="")
    print("\nEpoch: %d Loss %.3f, Acc %.3f, %.3f secs "
          "elapsed"%(epoch, train_avg_loss.result(), train_avg_accu.result(),
                     (time.time() - start_time)))

    valid_avg_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_avg_accu = tf.keras.metrics.Mean(name='valid_acc')
    # At the end of each epoch,
    # measure the validation loss and accuracy:
    for idx in range(1, n_iterations_validation + 1):
      X_batch, Y_batch = mnist.validation.next_batch(batch_size)
      loss, acc = evaluation((X_batch, Y_batch), caps_net)
      valid_avg_loss.update_state(loss)
      valid_avg_accu.update_state(acc)
      print("\rEvaluating the model: {}/{} ({:.1f}%)".format(idx, n_iterations_validation, idx * 100 / n_iterations_validation),
        end=" " * 10)
    print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
      epoch, valid_avg_accu.result() * 100, valid_avg_loss.result(),
      " (improved)" if valid_avg_loss.result() < best_loss_val else ""))

    if valid_avg_loss.result() < best_loss_val:
      best_loss_val = valid_avg_loss.result()

  print("Evaluation!")
  # Evaluation
  """
  test_avg_loss = tf.keras.metrics.Mean(name='test_loss')
  test_avg_accu = tf.keras.metrics.Mean(name='test_acc')
  for idx in range(1, n_iterations_test + 1):
    X_batch, Y_batch = mnist.test.next_batch(batch_size)
    loss, acc = evaluation((X_batch, Y_batch), caps_net)
    test_avg_loss.update_state(loss)
    test_avg_accu.update_state(acc)
    print("\rEvaluating the model: {}/{} ({:.1f}%)".format(idx,
                                                           n_iterations_test,
                                                           idx * 100 / n_iterations_test),
          end=" " * 10)
  print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
    test_avg_accu.result() * 100, test_avg_loss.result()))
  """

  # Predictions
  n_samples = 5
  sample_images = mnist.test.images[:n_samples].reshape([-1, 28, 28, 1])
  caps2_output_value, decoder_output_value, y_pred_value = pred(sample_images, caps_net)
  sample_images = tf.reshape(sample_images,(-1, 28, 28))
  reconstructions = tf.reshape(decoder_output_value, (-1, 28, 28))

  plt.figure(figsize=(n_samples * 2, 3))
  plt.title("Labels")
  for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.imshow(sample_images[index], cmap="binary")
    plt.title("Label:" + str(mnist.test.labels[index]))
    plt.axis("off")

  plt.tight_layout()
  plt.show()

  plt.figure(figsize=(n_samples * 2, 3))
  plt.title("Predictions")
  for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.title("Predicted:" + str(y_pred_value[index]))
    plt.imshow(reconstructions[index], cmap="binary")
    plt.axis("off")

  plt.tight_layout()
  plt.show()

  # Interpreting the Output Vectors
  n_steps = 11
  tweaked_vectors = tweak_pose_parameters(caps2_output_value, n_steps=n_steps)
  tweaked_vectors_reshaped = tf.reshape(tweaked_vectors, [-1, 1, caps2_n, caps2_dim, 1])
  tweak_labels = np.tile(mnist.test.labels[:n_samples], caps2_dim * n_steps)
  decoder_output_value = recon((tweaked_vectors_reshaped, tweak_labels), caps_net)
  tweak_reconstructions = tf.reshape(decoder_output_value,
                                     [caps2_dim, n_steps, n_samples, 28, 28])

  for dim in range(3):
    print("Tweaking output dimension #{}".format(dim))
    plt.figure(figsize=(n_steps / 1.2, n_samples / 1.5))
    for row in range(n_samples):
      for col in range(n_steps):
        plt.subplot(n_samples, n_steps, row * n_steps + col + 1)
        plt.imshow(tweak_reconstructions[dim, col, row], cmap="binary")
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
  main()
