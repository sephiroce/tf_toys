#-*- coding:utf-8 -*-
"""cnn_cifar10.py: the simplest example for the custom training loop"""
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.regularizers import l2

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"


class ConvolutionalNeuralNetwork(tf.keras.layers.Layer):
  def __init__(self):
    super(ConvolutionalNeuralNetwork, self).__init__()
    self.cnn11 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3],
                                      padding="same",
                                      activation="relu",
                                        kernel_regularizer=l2(0.001))
    self.cnn12 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3],
                                      padding="same",
                                      activation="relu",
                                        kernel_regularizer=l2(0.001))
    self.pool1 = tf.keras.layers.MaxPool2D((2,2))

    self.cnn21 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3],
                                      padding="same",
                                      activation="relu",
                                        kernel_regularizer=l2(0.001))
    self.cnn22 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3],
                                      padding="same",
                                      activation="relu",
                                        kernel_regularizer=l2(0.001))
    self.pool2 = tf.keras.layers.MaxPool2D((2,2))

    self.cnn31 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3],
                                      padding="same",
                                      activation="relu",
                                        kernel_regularizer=l2(0.001))
    self.cnn32 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3],
                                      padding="same",
                                      activation="relu",
                                        kernel_regularizer=l2(0.001))
    self.pool3 = tf.keras.layers.MaxPool2D((2,2))


    self.flat = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(128, activation='relu',
                                        kernel_regularizer=l2(0.001))  # since Mnist
    self.dropout = tf.keras.layers.Dropout(0.2)
    # has 10 classes
    self.dense2 = tf.keras.layers.Dense(10) # since Mnist has 10 classes


  def call(self, inputs, **kwargs):
    x = self.pool1(self.cnn12(self.cnn11(inputs)))
    x = self.pool2(self.cnn22(self.cnn21(x)))
    x = self.pool3(self.cnn32(self.cnn31(x)))
    x = self.flat(x)
    x = self.dense1(x)
    x = self.dropout(x)
    return self.dense2(x)


def main():
  #pylint: disable=too-many-locals
  print(tf.__version__)
  mnist_builder = tfds.builder("svhn_cropped")
  mnist_builder.download_and_prepare()
  ds_train = mnist_builder.as_dataset(split="train")
  ds_test = mnist_builder.as_dataset(split="test")

  ds_train = ds_train.repeat(1).shuffle(1024).batch(64)
  ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

  ds_test = ds_test.repeat(1).batch(64)
  ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

  cnn_model = ConvolutionalNeuralNetwork()
  optimizer = tf.keras.optimizers.Adam()

  loss_object = \
    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


  @tf.function
  def train(inputs, loss_state, acc_state):
    input_image = tf.cast(inputs["image"], tf.float32)
    input_label = tf.cast(inputs["label"], tf.float32)

    with tf.GradientTape() as tape:
      y_pred = cnn_model(input_image)
      loss = loss_object(input_label, y_pred)
    grads = tape.gradient(loss, cnn_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, cnn_model.trainable_variables))
    loss_state.update_state(loss)
    acc_state.update_state(input_label, y_pred)

  for epoch in range(1, 45):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for datum in iter(ds_train):
      train(datum, epoch_loss_avg, epoch_accuracy)
    print("Epoch %d loss %.3f, accuracy %.3f"%(epoch, epoch_loss_avg.result(),
                                               epoch_accuracy.result() * 100.0))

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
                                             acc_state.result() * 100.0))


if __name__ == "__main__":
  main()