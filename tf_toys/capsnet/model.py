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


@tf.function
def train(inputs, loss_state, acc_state, model, optimizer, training=True):
  image = tf.cast(inputs["image"], tf.float32)
  targets = inputs["label"]

  with tf.GradientTape() as tape:
    norm_v2, y_pred = model([image, targets], training=training)

    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    # margin loss
    T = tf.one_hot(targets, depth=caps2_n, name="T")
    present_error_raw = tf.square(tf.maximum(0., m_plus - norm_v2),
                                  name="present_error_raw")
    present_error = tf.reshape(present_error_raw, shape=(-1, 10),
                               name="present_error")
    absent_error_raw = tf.square(tf.maximum(0., norm_v2 - m_minus),
                                 name="absent_error_raw")
    absent_error = tf.reshape(absent_error_raw, shape=(-1, 10),
                              name="absent_error")
    L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
               name="L")
    loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  loss_state.update_state(loss)
  #correct = tf.equal(tf.cast(targets, tf.float32),
  #                   tf.cast(y_pred, tf.float32), name="correct")
  #accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
  acc_state.update_state(targets, y_pred)


def main():
  #pylint: disable=too-many-locals
  batch_size = 50
  tf.random.set_seed(42)

  mnist_builder = tfds.builder("mnist")
  mnist_builder.download_and_prepare()
  ds_train = mnist_builder.as_dataset(split="train")

  ds_train = ds_train.repeat(1).shuffle(1024).batch(batch_size)
  ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

  caps_net = CapsuleNetwork()
  opti = tf.keras.optimizers.Adam()

  n_iterations_per_epoch = 55000 // batch_size

  for epoch in range(1, 5):
    epoch_loss_avg = tf.keras.metrics.Mean(name='train_loss')
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_acc')
    for idx, datum in enumerate(iter(ds_train)):
      train(datum, epoch_loss_avg, epoch_accuracy, caps_net, opti)
      print('\rIteration: %d/%d loss %.3f accuracy %.3f' % (idx + 1,
                                                          n_iterations_per_epoch,
                                                          epoch_loss_avg.result(),
                                                          epoch_accuracy.result() * 100.0),
                                                            end="")
    print("Epoch %d loss %.3f, accuracy %.3f"%(epoch, epoch_loss_avg.result(),
                                               epoch_accuracy.result()))

if __name__ == "__main__":
  main()