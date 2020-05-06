import math
import tensorflow as tf
from tf_toys.capsnet.helper_train import unfold, squash

eps = 1e-12


class DynamicRouting2d(tf.keras.layers.Layer):
  def __init__(self, A, B, C, D, kernel_size=1, stride=1, padding=1, iters=3, std_dev=1.0):
    super(DynamicRouting2d, self).__init__()
    # initialize so that output logits are in reasonable range (0.1-0.9)
    # nn.init.normal_(self.fc.W, 0, 0.1) <<= 0.1: std_dev
    self.A = A
    self.B = B
    self.C = C
    self.D = D

    self.k = kernel_size
    self.kk = kernel_size ** 2
    self.kkA = self.kk * A

    self.stride = stride
    self.pad = padding

    self.iters = iters

    if std_dev == 1.0:
      init=tf.keras.initializers.he_uniform(seed=None)
      self.W = tf.Variable(init(shape=(self.kkA, B * D, C)), trainable=True)
    else:
      self.W = tf.Variable(tf.random.normal([self.kkA, B * D, C], mean=0.0,
                                            stddev=std_dev), trainable=True)

  def call(self, inputs, **kwargs):
    # x: [b, AC, h, w]
    pose = inputs
    batch, _, h, w = tf.shape(pose)[0], tf.shape(pose)[1], tf.shape(pose)[2], \
                 tf.shape(pose)[3]
    # [b, channel, width, height] to [b, ACkk, l]
    pose = unfold(pose, self.k, stride=self.stride, padding=self.pad)

    l = tf.shape(pose)[-1]
    # [b, A, C, kk, l]
    pose = tf.reshape(pose, [batch, self.A, self.C, self.kk, l])
    # [b, l, kk, A, C]
    pose = tf.transpose(pose, [0, 4, 3, 1, 2])
    # [b, l, kkA, C, 1]
    pose = tf.reshape(pose, [batch, l, self.kkA, self.C, 1])

    # [b, l, kkA, BD]
    pose_out = tf.squeeze(tf.matmul(self.W, pose), -1)
    # [b, l, kkA, B, D]
    pose_out = tf.reshape(pose_out, (batch, l, self.kkA, self.B, self.D))

    # [b, l, kkA, B, 1]
    b = tf.zeros([batch, l, self.kkA, self.B, 1], dtype=pose.dtype)

    # batch, 1, 1, classes, caps_size
    v = tf.zeros([batch, 1, 1, self.B, self.D])

    _, _, _, v, _ = tf.while_loop(self._condition, self._loop_body,
                               [pose_out, b, tf.constant(0), v, self.iters])

    # [b, l, B, D], removing 1
    v = tf.squeeze(v, 2)

    # [b, l, BD]
    v = tf.reshape(v, [batch, l, -1])

    # [b, BD, l]
    v = tf.transpose(v, [0, 2, 1])

    l = tf.cast(l, tf.float32)
    oh = ow = tf.cast(tf.floor(l ** (1.0 / 2)), tf.int32)

    # [b, BD, oh, ow]
    return tf.reshape(v, [batch, -1, oh, ow])

  @staticmethod
  def _condition(pose_out, b, counter, v, routing_iter):
    return tf.less_equal(counter, routing_iter)

  @staticmethod
  def _loop_body(pose_out, b, counter, v, routing_iter):
    c = tf.nn.softmax(b, axis=3)

    # [b, l, 1, B, D]
    s = tf.reduce_sum(c * pose_out, axis=2, keepdims=True)

    # [b, l, 1, B, D]
    v = squash(s)

    b += tf.reduce_sum(v * pose_out, axis=-1, keepdims=True)
    return pose_out, b, tf.add(counter, 1), v, routing_iter


class EmRouting2d(tf.keras.layers.Layer):
  def __init__(self, A, B, caps_size, kernel_size=3, stride=1, padding=1, iters=3, final_lambda=1e-2):
    super(EmRouting2d, self).__init__()

    self.A = A
    self.B = B
    self.psize = caps_size
    self.mat_dim = int(caps_size ** 0.5)

    self.k = kernel_size
    self.kk = kernel_size ** 2
    self.kkA = self.kk * A

    self.stride = stride
    self.pad = padding

    self.iters = iters

    init=tf.keras.initializers.he_uniform(seed=None)
    self.W = tf.Variable(init(shape=(self.kkA, B, self.mat_dim, self.mat_dim)), trainable=True)

    self.beta_u = tf.Variable(tf.zeros([1, 1, B, 1]), trainable=True)
    self.beta_a = tf.Variable(tf.zeros([1, 1, B]), trainable=True)

    self.final_lambda = final_lambda
    self.ln_2pi = tf.math.log(2 * math.pi)

    self.lambda_ = None

  def m_step(self, v, a_in, r):
    # v: [b, l, kkA, B, psize]
    # a_in: [b, l, kkA]
    # r: [b, l, kkA, B, 1]
    b, l = tf.shape(v)[0], tf.shape(v)[1]

    # r: [b, l, kkA, B, 1]
    r = r * tf.reshape(a_in, (b, l, -1, 1, 1))
    # r_sum: [b, l, 1, B, 1]
    r_sum = tf.reduce_sum(r, axis=2, keepdims=True)
    # coeff: [b, l, kkA, B, 1]
    coeff = r / (r_sum + eps)

    # mu: [b, l, 1, B, psize]
    mu = tf.reduce_sum(coeff * v, axis=2, keepdims=True)
    # sigma_sq: [b, l, 1, B, psize]
    sigma_sq = tf.reduce_sum(coeff * (v - mu) ** 2, axis=2, keepdims=True) + eps

    # [b, l, B, 1]
    r_sum = tf.squeeze(r_sum, 2)
    # [b, l, B, psize]
    sigma_sq = tf.squeeze(sigma_sq, 2)
    # [1, 1, B, 1] + [b, l, B, psize] * [b, l, B, 1]
    cost_h = (self.beta_u + tf.math.log(tf.math.sqrt(sigma_sq))) * r_sum

    # [b, l, B]
    a_out = tf.nn.sigmoid(self.lambda_ * (self.beta_a - tf.reduce_sum(cost_h,
                                                                      axis=3)))

    return a_out, mu, sigma_sq

  def e_step(self, v, a_out, mu, sigma_sq):
    b, l = tf.shape(a_out)[0], tf.shape(a_out)[1]
    # v: [b, l, kkA, B, psize]
    # a_out: [b, l, B]
    # mu: [b, l, 1, B, psize]
    # sigma_sq: [b, l, B, psize]

    # [b, l, 1, B, psize]
    sigma_sq = tf.expand_dims(sigma_sq, 2)

    ln_p_j = -0.5 * tf.reduce_sum(tf.math.log(sigma_sq*self.ln_2pi), axis=-1) \
             - tf.reduce_sum((v-mu)**2 / (2 * sigma_sq), axis=-1)

    # [b, l, kkA, B]
    ln_ap = ln_p_j + tf.math.log(tf.reshape(a_out, (b, l, 1, self.B)))
    # [b, l, kkA, B]
    r = tf.nn.softmax(ln_ap, axis=-1)
    # [b, l, kkA, B, 1]
    return tf.expand_dims(r, -1)

  def em_iteration(self, v, a_in, r):
    """
    Iteration range is from 1 to 4.

    # the below line is from open review
    self.lambda_ = self.final_lambda * (1 - 0.95 ** (i + 1))
    :param v:
    :param a_in:
    :param r:
    :return:
    """
    if self.iters < 0 or self.iters > 3:
      return None, None

    i = 0
    self.lambda_ = self.final_lambda * (1 - 0.95 ** (i + 1))
    a_out, pose_out, sigma_sq = self.m_step(v, a_in, r)
    if self.iters == 1:
      return a_out, pose_out

    i = 1
    r = self.e_step(v, a_out, pose_out, sigma_sq)
    self.lambda_ = self.final_lambda * (1 - 0.95 ** (i + 1))
    a_out, pose_out, sigma_sq = self.m_step(v, a_in, r)
    if self.iters == 2:
      return a_out, pose_out

    i = 2
    r = self.e_step(v, a_out, pose_out, sigma_sq)
    self.lambda_ = self.final_lambda * (1 - 0.95 ** (i + 1))
    a_out, pose_out, sigma_sq = self.m_step(v, a_in, r)
    if self.iters == 3:
      return a_out, pose_out


  def call(self, inputs, **kwargs):
    a_in, pose = inputs

    # a: [b, A, h, w]
    # pose: [b, A*psize, h, w]
    b, h, w = tf.shape(a_in)[0], tf.shape(a_in)[2], tf.shape(a_in)[3]

    # [b, A*psize*kk, l]
    pose = unfold(pose, self.k, stride=self.stride, padding=self.pad)
    l = tf.shape(pose)[-1]
    # [b, A, psize, kk, l]
    pose = tf.reshape(pose, (b, self.A, self.psize, self.kk, l))
    # [b, l, kk, A, psize]
    pose = tf.transpose(pose, (0, 4, 3, 1, 2))
    # [b, l, kkA, psize]
    pose = tf.reshape(pose, (b, l, self.kkA, self.psize))
    # [b, l, kkA, 1, mat_dim, mat_dim]
    pose = tf.expand_dims(tf.reshape(pose, (b, l, self.kkA, self.mat_dim,
                                     self.mat_dim)), 3)

    # [b, l, kkA, B, mat_dim, mat_dim]
    pose_out = tf.matmul(pose, self.W)

    # [b, l, kkA, B, psize]
    v = tf.reshape(pose_out, (b, l, self.kkA, self.B, -1))

    # [b, kkA, l]
    a_in = unfold(a_in, self.k, stride=self.stride, padding=self.pad)
    # [b, A, kk, l]
    a_in = tf.reshape(a_in, (b, self.A, self.kk, l))
    # [b, l, kk, A]
    a_in = tf.transpose(a_in, (0, 3, 2, 1))
    # [b, l, kkA]
    a_in = tf.reshape(a_in, (b, l, self.kkA))

    r = tf.ones((b, l, self.kkA, self.B, 1), dtype=a_in.dtype)

    # I will fix it
    a_out, pose_out = self.em_iteration(v, a_in, r)

    # [b, l, B*psize]
    pose_out = tf.reshape(tf.expand_dims(pose_out, 2), (b, l, -1))
    # [b, B*psize, l]
    pose_out = tf.transpose(pose_out, (0, 2, 1))
    # [b, B, l]
    a_out = tf.transpose(a_out, (0, 2, 1))

    l = tf.cast(l, tf.float32)
    oh = ow = tf.cast(tf.floor(l ** (1.0 / 2)), tf.int32)

    a_out = tf.reshape(a_out, (b, -1, oh, ow))
    pose_out = tf.reshape(pose_out, (b, -1, oh, ow))

    return a_out, pose_out


class SelfRouting2d(tf.keras.layers.Layer):
  def __init__(self, A, B, C, D, kernel_size=3, stride=1, padding=1, pose_out=False):
    super(SelfRouting2d, self).__init__()
    self.A = A
    self.B = B
    self.C = C
    self.D = D

    self.k = kernel_size
    self.kk = kernel_size ** 2
    self.kkA = self.kk * A

    self.stride = stride
    self.pad = padding

    self.pose_out = pose_out

    if pose_out:
      init=tf.keras.initializers.he_uniform(seed=None)
      self.W1 = tf.Variable(init(shape=(self.kkA, B * D, C)), trainable=True)

    self.W2 = tf.Variable(tf.zeros([self.kkA, B, C]), trainable=True)
    self.b2 = tf.Variable(tf.zeros([1, 1, self.kkA, B]), trainable=True)

  def call(self, inputs, **kwargs):
    # a: [b, A, h, w]
    # pose: [b, AC, h, w]
    a, pose = inputs
    b, h, w = tf.shape(a)[0], tf.shape(a)[2], tf.shape(a)[3]
    pose_out = None

    # [b, ACkk, l]
    pose = unfold(pose, self.k, stride=self.stride, padding=self.pad)
    l = tf.shape(pose)[-1]
    # [b, A, C, kk, l]
    pose = tf.reshape(pose, (b, self.A, self.C, self.kk, l))
    # [b, l, kk, A, C]
    pose = tf.transpose(pose, (0, 4, 3, 1, 2))
    # [b, l, kkA, C, 1]
    pose = tf.reshape(pose, (b, l, self.kkA, self.C, 1))

    if hasattr(self, 'W1'):
      # [b, l, kkA, BD]
      pose_out = tf.squeeze(tf.matmul(self.W1, pose), -1)
      # [b, l, kkA, B, D]
      pose_out = tf.reshape(pose_out, (b, l, self.kkA, self.B, self.D))

    # [b, l, kkA, B]
    logit = tf.squeeze(tf.matmul(self.W2, pose), -1) + self.b2

    # [b, l, kkA, B]
    r = tf.nn.softmax(logit, axis=3)

    # [b, kkA, l]
    a = unfold(a, self.k, stride=self.stride, padding=self.pad)
    # [b, A, kk, l]
    a = tf.reshape(a, (b, self.A, self.kk, l))
    # [b, l, kk, A]
    a = tf.transpose(a, (0, 3, 2, 1))
    # [b, l, kkA, 1]
    a = tf.reshape(a, (b, l, self.kkA, 1))

    # [b, l, kkA, B]
    ar = a * r
    # [b, l, 1, B]
    ar_sum = tf.reduce_sum(ar, axis=2, keepdims=True)
    # [b, l, kkA, B, 1]
    coeff = tf.expand_dims((ar / ar_sum), -1)

    # [b, l, B]
    a_out = ar_sum / tf.reduce_sum(a, axis=2, keepdims=True)
    a_out = tf.squeeze(a_out, 2)

    # [b, B, l]
    a_out = tf.transpose(a_out, (0, 2, 1))

    if hasattr(self, 'W1'):
      # [b, l, B, D]
      pose_out = tf.reduce_sum((coeff * pose_out), axis=2)
      # [b, l, BD]
      pose_out = tf.reshape(pose_out, (b, l, -1))
      # [b, BD, l]
      pose_out = tf.transpose(pose_out, (0, 2, 1))

    l = tf.cast(l, tf.float32)
    oh = ow = tf.cast(tf.floor(l ** (1.0 / 2)), tf.int32)

    a_out = tf.reshape(a_out, (b, -1, oh, ow))
    if hasattr(self, 'W1'):
      pose_out = tf.reshape(pose_out, (b, -1, oh, ow))

    return a_out, pose_out
