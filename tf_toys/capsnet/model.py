#-*- coding:utf-8 -*-
"""model.py: the simplest example for the capsule network"""
# from https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb

from tf_toys.capsnet.blocks import CapsuleNetwork

import tensorflow as tf
import tensorflow_datasets as tfds


caps1_n_maps = 32
caps1_n_dims = 8
caps2_n = 10
caps2_dim = 16
n_caps1 = caps1_n_maps * 6 * 6

def main():
  #pylint: disable=too-many-locals
  batch_size = 50
  tf.random.set_seed(42)

  mnist_builder = tfds.builder("mnist")
  mnist_builder.download_and_prepare()
  ds_train = mnist_builder.as_dataset(split="train")
  ds_test = mnist_builder.as_dataset(split="test")

  ds_train = ds_train.repeat(1).shuffle(1024).batch(batch_size)
  ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

  ds_test = ds_test.repeat(1).batch(batch_size)
  ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

  caps_net = CapsuleNetwork()
  optimizer = tf.keras.optimizers.Adam()
  loss_object =\
    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  n_iterations_per_epoch = 55000 // batch_size
  @tf.function
  def train(inputs, loss_state, acc_state, training=True):
    image = tf.cast(inputs["image"], tf.float32)
    targets = tf.cast(inputs["label"], tf.float32)

    with tf.GradientTape() as tape:
      norm_v2, y_pred, dec_out = caps_net([image, targets], training=training)

      m_plus = 0.9
      m_minus = 0.1
      lambda_ = 0.5

      # margin loss
      present_err = tf.reshape(tf.square(tf.maximum(0., m_plus - norm_v2)),
                               shape=(-1, 10), name="present_error")
      absent_err = tf.reshape(tf.square(tf.maximum(0., norm_v2 - m_minus)),
                              shape=(-1, 10), name="absent_error")
      T = tf.one_hot(tf.cast(targets, tf.int32), depth=caps2_n, name="T")
      margin_loss = T * present_err + lambda_ * (1.0 - T) * absent_err
      margin_loss = tf.reduce_mean(tf.reduce_sum(margin_loss, axis=1),
                                   name="margin_loss")

      # reconstruction loss
      flat_inp = tf.reshape(image, [-1, 28 * 28], name="X_flat")
      recon_loss = tf.reduce_mean(tf.square(flat_inp - dec_out),
                                  name="recon_loss")

      alpha = 5e-4
      #loss = margin_loss + alpha * recon_loss
      loss = margin_loss
      #tf.print(margin_loss, recon_loss)
    grads = tape.gradient(loss, caps_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, caps_net.trainable_variables))

    loss_state.update_state(loss)
    y_pred = tf.squeeze(y_pred, axis=-1)
    correct = tf.equal(tf.cast(targets, tf.float32),
                       tf.cast(y_pred, tf.float32), name="correct")
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    acc_state.update_state(accuracy)

  for epoch in range(1, 5):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.Mean()
    for idx, datum in enumerate(iter(ds_train)):
      train(datum, epoch_loss_avg, epoch_accuracy)
      print('\rIteration: %d/%d loss %.3f accuracy %.3f' % (idx + 1,
                                                          n_iterations_per_epoch,
                                                          epoch_loss_avg.result(),
                                                          epoch_accuracy.result() * 100.0),
                                                            end="")
    print("Epoch %d loss %.3f, accuracy %.3f"%(epoch, epoch_loss_avg.result(),
                                               epoch_accuracy.result()))

if __name__ == "__main__":
  main()