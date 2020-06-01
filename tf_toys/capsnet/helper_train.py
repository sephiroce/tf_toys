import tensorflow as tf

class MultiStepLR(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, learning_rate, milestones, gamma):
    self.lr = learning_rate
    self.gamma = tf.cast(tf.constant(gamma), tf.float32)
    self.a = tf.cast(tf.constant(milestones[0]), tf.float32)
    self.b = tf.cast(tf.constant(milestones[1]), tf.float32)

  def __call__(self, epoch):
    epoch = tf.cast(epoch, tf.float32)
    """
    if epoch < self.a:
      lr = self.lr
    elif self.a <= epoch < self.b:
      lr = self.lr * self.gamma
    else:
      lr = self.lr * self.gamma * self.gamma
    tf.print("epoch, lr", epoch, self.lr)
    """
    return self.lr

def loss_nll(y_true, y_pred):
  # y_true: [batch], index, int
  # y_pred: [batch, classes], logit, float
  mask = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
  return -tf.reduce_sum(y_pred * mask) / \
         tf.cast(tf.reduce_prod(tf.shape(y_true)), y_pred.dtype)

def squash(s, axis=-1, epsilon=1e-7):
  squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
  safe_norm = tf.sqrt(squared_norm + epsilon)
  squash_factor = squared_norm / (1. + squared_norm)
  unit_vector = s / safe_norm
  return squash_factor * unit_vector

def length(s, axis=-1, epsilon=1e-7, keepdims=False):
  squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims)
  return tf.sqrt(squared_norm + epsilon)

def unfold(tensor, kernel_size, dilation=1, padding=0, stride=1):
  """
      Tensorflow 2 version of unfold in Torch (1.2)
      Thanks to tf.image.extract_patches,
      we just need to reshape, pad, and transpose before and after the operation.

      tensor: [b, channel, width, height], float
      kernel_size: [], int
      dilation: [], int
      padding: [], int
      stride: [], int
      * The four scalars above are broadcast to width and height
  """
  if dilation != 1:
    print("WARNING!!: dilation != 1 might work not as intended.")

  b, c = tf.shape(tensor)[0], tf.shape(tensor)[1]
  # tensor: from [b, channel, width, height]
  #           to [b, width, height, channel]
  tensor = tf.transpose(tensor, [0, 2, 3, 1])
  # tensor: from [b, width, height, channel]
  #           to [b, width + padding * 2, height + padding * 2, channel]
  if padding > 0:
    tensor = tf.keras.layers.ZeroPadding2D(padding=padding)(tensor)
  # this implementation is tf.image.extract_patches
  kernel_size = [1, kernel_size, kernel_size, 1]
  stride = [1, stride, stride, 1]
  dilation = [1, dilation, dilation, 1]
  tensor = tf.image.extract_patches(images=tensor, sizes=kernel_size,
                                    strides=stride, rates=dilation,
                                    padding='VALID')
  # it needs to be refactored
  w = tf.shape(tensor)[1]
  h = tf.shape(tensor)[2]
  tensor = tf.reshape(tensor, [b, w, h, -1, c])
  tensor = tf.transpose(tensor, [0, 1, 2, 4, 3])
  tensor = tf.reshape(tensor, [b, w * h, -1])
  tensor = tf.transpose(tensor, [0, 2, 1])

  return tensor

def main():
  c = MultiStepLR(0.05, milestones=[30, 80], gamma=0.1)
  for epoch in range(100):
    print(c(epoch))

  import numpy as np
  pose = np.load("sample/data/data.npy")
  pose = tf.constant(pose, dtype=tf.float32)
  unfolded_pose = unfold(pose, 3, 1, 2, 2)
  # loading reference
  ref = np.load("sample/data/result.npy")

  print(np.array_equal(unfolded_pose.numpy(), ref))

if __name__ == "__main__":
  main()