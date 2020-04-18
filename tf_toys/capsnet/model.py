#-*- coding:utf-8 -*-
"""model.py: the simplest example for the capsule network"""
# from https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb
import tensorflow as tf
import tensorflow_datasets as tfds


caps1_n_maps = 32
caps1_n_dims = 8
caps1_n_caps = caps1_n_maps * 6 * 6
caps2_n_caps = 10
caps2_n_dims = 16

m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5


class ConvolutionalLayers(tf.keras.layers.Layer):
  def __init__(self):
    super(ConvolutionalLayers, self).__init__()
    self.cnn1 = None
    self.cnn2 = None
    self.reshape = None

  def build(self, input_shape):
    self.cnn1 = tf.keras.layers.Conv2D(input_shape=input_shape,
                                      filters=256, kernel_size=[9, 9],
                                       strides=[1, 1],
                                      padding="valid",
                                      activation="relu")
    # [batch, pc width, pc height, n_maps * n_dims]
    self.cnn2 = tf.keras.layers.Conv2D(filters=caps1_n_maps * caps1_n_dims,
                                       kernel_size=[9, 9],
                                      strides=[2, 2],
                                      padding="valid",
                                      activation="relu")
    # [batch, pc width * pc height * n_maps, n_dims]
    self.reshape = tf.keras.layers.Reshape([caps1_n_caps, caps1_n_dims])

  def call(self, inputs, **kwargs):
    pc = self.cnn2(self.cnn1(inputs))
    return self.reshape(pc)

class DecoderLayers(tf.keras.layers.Layer):
  def __init__(self):
    super(DecoderLayers, self).__init__()
    self.hidden1 = None
    self.hidden2 = None
    self.decoder_output = None

  def build(self, input_shape):
    n_hidden1 = 512
    n_hidden2 = 1024
    n_output = 28 * 28
    self.hidden1 = tf.keras.layers.Dense(n_hidden1,
                                    activation="relu",
                                    name="hidden1")
    self.hidden2 = tf.keras.layers.Dense(n_hidden2,
                                    activation="relu",
                                    name="hidden2")
    self.decoder_output = tf.keras.layers.Dense(n_output,
                                           activation="sigmoid",
                                           name="decoder_output")

  def call(self, inputs, **kwargs):
    return self.decoder_output(self.hidden2(self.hidden1(inputs)))

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
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                 keepdims=keep_dims)
    return tf.sqrt(squared_norm + epsilon)

def main():
  #pylint: disable=too-many-locals
  print(tf.__version__)
  mnist_builder = tfds.builder("mnist")
  mnist_builder.download_and_prepare()
  ds_train = mnist_builder.as_dataset(split="train")
  ds_test = mnist_builder.as_dataset(split="test")

  ds_train = ds_train.repeat(1).shuffle(1024).batch(32)
  ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

  ds_test = ds_test.repeat(1).batch(32)
  ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

  cnn_model = ConvolutionalLayers()
  dec_model = DecoderLayers()
  optimizer = tf.keras.optimizers.Adam()
  cnn_model.build(input_shape=[-1, 28, 28, 1])

  loss_object = \
    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  @tf.function
  def train(inputs, loss_state, acc_state):
    input_image = tf.cast(inputs["image"], tf.float32)
    input_label = tf.cast(inputs["label"], tf.float32)
    batch_size = tf.shape(input_image)[0]

    with tf.GradientTape() as tape:
      caps1_raw = cnn_model(input_image)
      #tf.print("caps1_raw:", tf.shape(caps1_raw))
      # this is u
      caps1_output = squash(caps1_raw, name="caps1_output")
      #tf.print("caps1_output:", tf.shape(caps1_output))

      init_sigma = 0.1
      W = tf.random.normal([1, caps1_n_caps, caps2_n_caps, caps2_n_dims,
                                 caps1_n_dims],stddev=init_sigma,name="W_init")
      W_tiled = tf.tile(W, [batch_size, 1,1,1,1], name="W_tiled")

      caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                             name="caps1_output_expanded")
      caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                         name="caps1_output_tile")
      caps1_output_tiled = tf.tile(caps1_output_tile,
                                   [1, 1, caps2_n_caps, 1, 1],
                                   name="caps1_output_tiled")
      caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                                  name="caps2_predicted")

      #tf.print("W_tiled:", tf.shape(W_tiled))
      #tf.print("caps1_output_tiled:", tf.shape(caps1_output_tiled))

      # this is b, weight between 1, 2, Initializing before the loop
      raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                             dtype=tf.float32, name="raw_weights")

      # Loop Start!, Round1
      routing_weights = tf.nn.softmax(raw_weights, axis=2,
                                      name="routing_weights")
      weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                         name="weighted_predictions")
      weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True,
                                   name="weighted_sum")
      caps2_output_round_1 = squash(weighted_sum, axis=-2,
                                    name="caps2_output_round_1")

      # Round2
      caps2_output_round_1_tiled = tf.tile(
        caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
        name="caps2_output_round_1_tiled")
      agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                            transpose_a=True, name="agreement")
      raw_weights_round_2 = tf.add(raw_weights, agreement,
                                   name="raw_weights_round_2")

      # just like round 1
      routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                              axis=2,
                                              name="routing_weights_round_2")
      weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                                 caps2_predicted,
                                                 name="weighted_predictions_round_2")
      weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                           axis=1, keepdims=True,
                                           name="weighted_sum_round_2")
      caps2_output_round_2 = squash(weighted_sum_round_2,
                                    axis=-2,
                                    name="caps2_output_round_2")

      with tf.name_scope("margin_loss"):
        T = tf.one_hot(tf.cast(input_label, tf.int32), depth=caps2_n_caps,
                               name="T")
        caps2_output_norm = safe_norm(caps2_output_round_2, axis=-2, keep_dims=True,
                                      name="caps2_output_norm")

        present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                                      name="present_error_raw")
        present_error = tf.reshape(present_error_raw, shape=(-1, 10),
                                   name="present_error")

        absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                                     name="absent_error_raw")
        absent_error = tf.reshape(absent_error_raw, shape=(-1, 10),
                                  name="absent_error")

        L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
                   name="L")
        margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

      with tf.name_scope("reconstruction_loss"):
        # To see prediction
        y_proba = safe_norm(caps2_output_round_2, axis=-2, name="y_proba")
        y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
        y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2], name="y_pred")

        reconstruction_targets = tf.cast(input_label, tf.int32)
        """
        reconstruction_targets = tf.cond(True,  # condition
                                         lambda: input_label,  # if True
                                         lambda: y_pred,  # if False
                                         name="reconstruction_targets")
        """
        reconstruction_mask = tf.one_hot(reconstruction_targets,
                                         depth=caps2_n_caps,
                                         name="reconstruction_mask")
        reconstruction_mask_reshaped = tf.reshape(
          reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1],
          name="reconstruction_mask_reshaped")
        caps2_output_masked = tf.multiply(
          caps2_output_round_2, reconstruction_mask_reshaped,
          name="caps2_output_masked")

        decoder_input = tf.reshape(caps2_output_masked,
                                   [-1, caps2_n_caps * caps2_n_dims],
                                   name="decoder_input")

        decoder_output = dec_model(decoder_input)

        X_flat = tf.reshape(input_image, [-1, 28 * 28], name="X_flat")
        squared_difference = tf.square(X_flat - decoder_output,
                                       name="squared_difference")
        reconstruction_loss = tf.reduce_mean(squared_difference,
                                             name="reconstruction_loss")

      alpha = 0.0005

      loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")
      grads = tape.gradient(loss, cnn_model.trainable_variables)
      optimizer.apply_gradients(zip(grads, cnn_model.trainable_variables))

    loss_state.update_state(loss)
    correct = tf.equal(tf.cast(input_label, tf.float32),
                       tf.cast(y_pred, tf.float32), name="correct")
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    acc_state.update_state(accuracy)


  for epoch in range(1, 10):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.Mean()
    for datum in iter(ds_train):
      train(datum, epoch_loss_avg, epoch_accuracy)
    print("Epoch %d loss %.3f, accuracy %.3f"%(epoch, epoch_loss_avg.result(),
                                               epoch_accuracy.result()))

  loss_state = tf.keras.metrics.Mean()
  acc_state = tf.keras.metrics.SparseCategoricalAccuracy()
  for inputs in iter(ds_test):
    input_image = tf.cast(inputs["image"], tf.float32)
    input_label = tf.cast(inputs["label"], tf.float32)
    y_pred = cnn_model(input_image)
    loss = loss_object(input_label, y_pred)
    loss_state.update_state(loss)
    acc_state.update_state(input_label, y_pred)
  print("Test loss %.3f, accuracy %.3f" % (loss_state.result(),
                                           acc_state.result()))


if __name__ == "__main__":
  main()