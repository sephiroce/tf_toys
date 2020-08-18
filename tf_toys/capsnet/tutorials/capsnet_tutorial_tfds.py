import tensorflow as tf
import tensorflow_datasets as tfds
import time
import numpy as np

caps1_n = 32 # number of capsules
caps1_dim = 8 # dimension of capsules
caps2_n = 10 # class number
caps2_dim = 16 # dimension of capsules
exp_caps1_n = caps1_n * 6 * 6  # 1152 primary capsules
dec_out_dim = 28 * 28
routing_iter = 2

def squash(s, axis=-1, epsilon=1e-7):
  squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
  safe_norm = tf.sqrt(squared_norm + epsilon)
  squash_factor = squared_norm / (1. + squared_norm)
  unit_vector = s / safe_norm
  return squash_factor * unit_vector

def length(s, axis=-1, epsilon=1e-7, keepdims=False):
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
    reconstruction_targets = tf.cast(reconstruction_targets, tf.int32)
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

    tf.print("\nu_hat:",tf.shape(u_hat))
    # ==========================================================================
    # Dynamic Routing
    # ==========================================================================
    raw_weights = tf.zeros([batch_size, exp_caps1_n, caps2_n, 1, 1],
                            dtype=tf.float32)
    # Round1, Line 4
    routing_weights = tf.nn.softmax(raw_weights, axis=2)
    # Round1, Line 5
    weighted_predictions = tf.multiply(routing_weights, u_hat)
    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1,
                                              keepdims=True)
    # Round1, Line 6
    caps2_output_round_1= squash(weighted_sum, axis=-2)
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
    tf.print(tf.shape(caps2_output))

    y_proba = length(caps2_output, axis=-2)
    y_proba_argmax = tf.argmax(y_proba, axis=2)
    y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2])

    # Reconstruction Network
    dec_out = self.recon_((caps2_output, Y if training else y_pred))
    return caps2_output, dec_out, y_pred

def loss_function(X, Y, caps2_output, dec_out, y_pred):
  # Loss: Margin loss
  m_plus, m_minus, lambda_ = 0.9, 0.1, 0.5
  Y = tf.cast(Y, tf.int32)
  T = tf.one_hot(Y, depth=caps2_n)
  caps2_output_norm = length(caps2_output, axis=-2, keepdims=True)
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
  y_pred = tf.cast(y_pred, tf.int32)
  correct = tf.equal(Y, y_pred)
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
  return loss, accuracy

@tf.function
def train(inputs, model, optimizer):
  X = tf.cast(inputs["image"], tf.float32)
  Y = tf.cast(inputs["label"], tf.float32)

  with tf.GradientTape() as tape:
    caps2_output, dec_out, y_pred = model((X, Y), training=True)
    loss, accuracy = loss_function(X, Y, caps2_output, dec_out, y_pred)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return loss, accuracy

@tf.function
def evaluation(inputs, model):
  X = tf.cast(inputs["image"], tf.float32)
  Y = tf.cast(inputs["label"], tf.float32)

  caps2_output, dec_out, y_pred = model((X, None), training=False)
  return loss_function(X, Y, caps2_output, dec_out, y_pred)

@tf.function
def pred(inputs, model):
  X = tf.cast(inputs["image"], tf.float32)
  return model((X, None), training=False)

@tf.function
def recon(inputs, model):
  X = tf.cast(inputs["image"], tf.float32)
  Y = tf.cast(inputs["label"], tf.float32)
  return model.recon((X,Y))

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

  mnist_builder = tfds.builder("mnist")
  mnist_builder.download_and_prepare()
  ds_train = mnist_builder.as_dataset(split="train")
  ds_test = mnist_builder.as_dataset(split="test")

  ds_train = ds_train.repeat(1).shuffle(1024).batch(32)
  ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

  ds_test = ds_test.repeat(1).batch(32)
  ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

  caps_net = CapsuleNetwork()
  opti = tf.keras.optimizers.Adam()

  batch_size = 50
  n_iterations_per_epoch = 60000 // batch_size
  n_iterations_test = n_iterations_validation = 5000 // batch_size
  best_loss_val = 1e14

  epoch = 10

  # Training
  for epoch in range(1, epoch + 1):
    train_avg_loss = tf.keras.metrics.Mean(name='train_loss')
    train_avg_accu = tf.keras.metrics.Mean(name='train_acc')
    start_time = time.time()
    for i, datum in enumerate(iter(ds_train)):
      idx = i + 1
      loss, acc = train(datum, caps_net, opti)
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
    for i, datum in enumerate(iter(ds_test)):
      idx = i + 1
      loss, acc = evaluation(datum, caps_net)
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
  test_avg_loss = tf.keras.metrics.Mean(name='test_loss')
  test_avg_accu = tf.keras.metrics.Mean(name='test_acc')
  for i, datum in enumerate(iter(ds_test)):
    idx = i + 1
    loss, acc = evaluation((datum), caps_net)
    test_avg_loss.update_state(loss)
    test_avg_accu.update_state(acc)
    print("\rEvaluating the model: {}/{} ({:.1f}%)".format(idx,
                                                           n_iterations_test,
                                                           idx * 100 / n_iterations_test),
          end=" " * 10)
  print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
    test_avg_accu.result() * 100, test_avg_loss.result()))


if __name__ == "__main__":
  main()
