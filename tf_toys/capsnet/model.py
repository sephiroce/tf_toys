from tf_toys.capsnet.routing import DynamicRouting2d, EmRouting2d, SelfRouting2d
from tf_toys.capsnet.helper_train import squash, length
import tensorflow as tf

DATASET_CONFIGS = {
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'svhn_cropped': {'size': 32, 'channels': 3, 'classes': 10},
    'smallnorb': {'size': 32, 'channels': 1, 'classes': 5},
}

def create(name, conf, mode):
  name = name.lower()
  if name.startswith("resnet"):
    return ResNet(conf, DATASET_CONFIGS[conf.dataset], BasicBlock, [3, 3, 3],
                  mode)
  """
  if name.startwith("resnet32"):
    return ResNet(conf, BasicBlock, [5, 5, 5])
  if name.startwith("resnet44"):
    return ResNet(conf, BasicBlock, [7, 7, 7])
  if name.startwith("resnet56"):
    return ResNet(conf, BasicBlock, [9, 9, 9])
  if name.startwith("resnet110"):
    return ResNet(conf, BasicBlock, [18, 18, 18])
  """
  if name.startswith("convnet"):
    return ConvNet(conf, DATASET_CONFIGS[conf.dataset], mode)
  if name.startswith("small"):
    return SmallNet(conf, DATASET_CONFIGS[conf.dataset])

  raise NotImplementedError

def padding(args):
  # padding to channel, I'll fix it
  x, planes = args
  x = tf.pad(x[:, ::2, ::2, :],
                [[0, 0], [0, 0], [0, 0], [planes // 4, planes // 4]],
                "CONSTANT")
  return x

class BasicBlock(tf.keras.layers.Layer):
  expansion = 1
  def __init__(self, in_planes, planes, stride=1, option='A'):
    super(BasicBlock, self).__init__()
    self.conv1 = tf.keras.Sequential()
    self.conv1.add(tf.keras.layers.ZeroPadding2D(padding=1))
    self.conv1.add(tf.keras.layers.Conv2D(planes, kernel_size=3, strides=stride,
                                          use_bias=False))
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.Sequential()
    self.conv2.add(tf.keras.layers.ZeroPadding2D(padding=1))
    self.conv2.add(tf.keras.layers.Conv2D(planes, kernel_size=3, strides=1,
                                          use_bias=False))
    self.bn2 = tf.keras.layers.BatchNormalization()

    self.shortcut = tf.keras.Sequential() # bypass
    self.planes = None
    if stride != 1 or in_planes != planes:
      if option == 'A':
        """
        For CIFAR10 ResNet paper uses option A.
        """
        self.shortcut = tf.keras.layers.Lambda(padding)
        self.planes = planes
      elif option == 'B':
        self.shortcut = tf.keras.Sequential()
        self.shortcut.add(tf.keras.layers.Conv2D(self.expansion * planes,
                                                 kernel_size=1,
                                                 strides=stride,
                                                 use_bias=False))
        self.shortcut.add(tf.keras.layers.BatchNormalization())

  def call(self, inputs, **kwargs):
    out = tf.nn.relu(self.bn1(self.conv1(inputs)))
    out = self.bn2(self.conv2(out))
    if self.planes is not None: # it means option == 'A'
      out += self.shortcut((inputs, self.planes))
    else:
      out += self.shortcut(inputs)
    out = tf.nn.relu(out)
    return out


class ResNet(tf.keras.layers.Layer):
  def __init__(self, config, cfg_data, block, num_blocks, mode):
    super(ResNet, self).__init__()

    _, classes = cfg_data['channels'], cfg_data['classes']
    self.num_caps = num_caps = config.num_caps
    self.caps_size = caps_size = config.caps_size
    self.relu = tf.keras.layers.ReLU()
    self.relu2 = tf.keras.layers.ReLU()
    self.mode = mode
    self.depth = depth = config.depth
    self.planes = planes = config.planes
    self.in_planes = planes  # always same?

    self.conv1 = tf.keras.Sequential()
    self.conv1.add(tf.keras.layers.ZeroPadding2D(padding=1))
    self.conv1.add(tf.keras.layers.Conv2D(planes, kernel_size=3, strides=1,
                                          use_bias=False))
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.layer1 = self._make_layer(block, planes, num_blocks[0], strides=1)
    self.layer2 = self._make_layer(block, 2 * planes, num_blocks[1], strides=2)
    self.layer3 = self._make_layer(block, 4 * planes, num_blocks[2], strides=2)

    self.conv_layers = []
    self.norm_layers = []

    stride = 1
    if self.mode in ['DR', 'EM', 'SR']:
      for d in range(1, depth):
        stride = 2 if d == 1 else 1
        if self.mode == 'DR':
          self.conv_layers.append(
            DynamicRouting2d(num_caps, num_caps, caps_size, caps_size, kernel_size=3,
                             stride=stride, padding=1))
          self.norm_layers.append(tf.keras.layers.BatchNormalization())
        elif self.mode == 'EM':
          self.conv_layers.append(
            EmRouting2d(num_caps, num_caps, caps_size, kernel_size=3, strides=stride,
                        padding=1))
          self.norm_layers.append(tf.keras.layers.BatchNormalization())
        elif self.mode == 'SR':
          self.conv_layers.append(
            SelfRouting2d(num_caps, num_caps, caps_size, caps_size, kernel_size=3,
                          strides=stride, padding=1, pose_out=True))
          self.norm_layers.append(tf.keras.layers.BatchNormalization())

    final_shape = 8 if depth == 1 else 4

    # Routings: ConvCaps to Class Capsules
    if self.mode in ['DR', 'EM', 'SR']:
      self.conv_pose = tf.keras.Sequential()
      self.conv_pose.add(tf.keras.layers.ZeroPadding2D(padding=1))
      self.conv_pose.add(tf.keras.layers.Conv2D(num_caps * caps_size,
                                                kernel_size=3, strides=1,
                                                use_bias=False))
      self.bn_pose = tf.keras.layers.BatchNormalization()
      if self.mode == 'DR':
        self.fc = DynamicRouting2d(num_caps, classes, caps_size, caps_size,
                                   kernel_size=final_shape, padding=0)
      elif self.mode in ['EM', 'SR']:
        self.conv_a = tf.keras.Sequential()
        self.conv_a.add(tf.keras.layers.ZeroPadding2D(padding=1))
        self.conv_a.add(tf.keras.layers.Conv2D(num_caps, kernel_size=3,
                                               strides=1, use_bias=False))
        self.bn_a = tf.keras.layers.BatchNormalization()

        if self.mode == 'EM':
          self.fc = EmRouting2d(num_caps, classes, caps_size,
                                kernel_size=final_shape, padding=0)
        else:
          self.fc = SelfRouting2d(num_caps, classes, caps_size, 1,
                                  kernel_size=final_shape, padding=0,
                                  pose_out=False)
    else:
      # avg pooling
      if self.mode == 'AVG':
        self.pool = tf.keras.layers.AveragePooling2D(final_shape)
        self.fc = tf.keras.layers.Dense(classes)

      # max pooling
      if self.mode == 'MAX':
        self.pool = tf.keras.layers.MaxPool2D(final_shape)
        self.fc = tf.keras.layers.Dense(classes)

      # What is this?
      if self.mode == 'FC':
        self.conv_ = tf.keras.Sequential()
        self.conv_.add(tf.keras.layers.ZeroPadding2D(padding=1))
        self.conv_.add(tf.keras.layers.Conv2D(num_caps * caps_size,
                                              kernel_size=3, strides=stride,
                                              use_bias=True))
        self.bn_ = tf.keras.layers.BatchNormalization()
        self.fc = tf.keras.layers.Dense(classes)

  def _make_layer(self, block, planes, num_blocks, strides):
    strides = [strides] + [1] * (num_blocks - 1)
    layers = tf.keras.Sequential()
    for stride in strides:
      layers.add(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion

    return layers

  def call(self, inputs, **kwargs):
    out = self.relu(self.bn1(self.conv1(inputs)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)

    # DR
    if self.mode == 'DR':
      pose = self.bn_pose(self.conv_pose(out))
      b, w, h, c = tf.shape(pose)[0], tf.shape(pose)[1], tf.shape(pose)[2], \
                   tf.shape(pose)[3]

      pose = squash(tf.reshape(pose, (b, w, h, self.num_caps, self.caps_size)))
      pose = tf.reshape(pose, [b, w, h, c])
      # [b, h, w, c]
      pose = tf.transpose(pose, [0, 3, 2, 1])

      # routing methods
      for m in self.conv_layers:
        pose = m(pose)
      out = tf.squeeze(self.fc(pose))
      out = tf.reshape(out, [b, -1, self.caps_size])
      out = length(out, axis=-1)
      out = out / tf.reduce_sum(out, keepdims=True)
      out = tf.math.log(out)

    # Routing
    elif self.mode in ['EM', 'SR']:
      a, pose = self.conv_a(out), self.conv_pose(out)
      a, pose = tf.nn.sigmoid(self.bn_a(a)), self.bn_pose(pose)

      # [b, h, w, c]
      a = tf.transpose(a, [0, 3, 2, 1])
      pose = tf.transpose(pose, [0, 3, 2, 1])
      for m, bn in zip(self.conv_layers, self.norm_layers):
        a, pose = m((a, pose))
        pose = bn(pose)

      a, _ = self.fc((a, pose)) # we don't use pose anymore, it only for routing
      out = tf.reshape(a, [tf.shape(a)[0], -1])

      if self.mode == 'EM':
        out = out / tf.reduce_sum(out, keepdims=True)

      out = tf.math.log(out)

    # No Routing
    else:
      if self.mode in ['AVG', 'MAX']:
        out = self.pool(out)
      elif self.mode == 'FC':
        out = self.relu2(self.bn_(self.conv_(out)))
      out = tf.reshape(out, [tf.shape(out)[0], -1])
      out = self.fc(out)

    return out

  def forward_activations(self, x):
    """
    What is this for?
    :param x:
    :return:
    """
    out = tf.nn.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)

    if self.mode == 'DR':
      pose = self.bn_pose(self.conv_pose(out))
      b, c, h, w = tf.shape(pose)
      pose = tf.transpose(pose, [0, 2, 3, 1])
      pose = squash(tf.reshape(pose(b, h, w, self.num_caps, self.caps_size)))
      pose = tf.reshape(pose, [b, h, w, -1])
      pose = tf.transpose(pose, [0, 3, 1, 2])
      a = length(pose, axis=1)
    elif self.mode in ['EM', 'SR']:
      a = tf.nn.sigmoid(self.bn_a(self.conv_a(out)))
    else:
      raise NotImplementedError

    return a


class ConvNet(tf.keras.layers.Layer):
  def __init__(self, conf, cfg_data, mode):
    super(ConvNet, self).__init__()
    _, classes = cfg_data["channels"], cfg_data["classes"]
    self.num_caps = num_caps = conf.num_caps
    self.caps_size = caps_size = conf.caps_size #16
    self.depth = depth = conf.depth
    self.mode = mode
    assert self.mode in ['DR', 'EM', 'SR', 'AVG', 'MAX', 'FC']
    planes = conf.planes

    self.backbone = tf.keras.Sequential()
    self.backbone.add(tf.keras.layers.ZeroPadding2D(padding=1))
    if self.mode == 'DR':
      self.backbone.add(tf.keras.layers.Conv2D(planes, kernel_size=3,
                                               strides=1, use_bias=False,
                                               kernel_initializer=tf.keras.initializers.random_normal(mean=0.0, stddev=0.5)))
    else:
      self.backbone.add(tf.keras.layers.Conv2D(planes, kernel_size=3,
                                               strides=1, use_bias=False))
    self.backbone.add(tf.keras.layers.BatchNormalization())
    self.backbone.add(tf.keras.layers.ReLU())
    self.backbone.add(tf.keras.layers.ZeroPadding2D(padding=1))
    self.backbone.add(tf.keras.layers.Conv2D(planes * 2, kernel_size=3,
                                             strides=1, use_bias=False

))
    self.backbone.add(tf.keras.layers.BatchNormalization())
    self.backbone.add(tf.keras.layers.ReLU())
    self.backbone.add(tf.keras.layers.Conv2D(planes * 2, kernel_size=3,
                                             strides=1, use_bias=False

))
    self.backbone.add(tf.keras.layers.BatchNormalization())
    self.backbone.add(tf.keras.layers.ReLU())
    self.backbone.add(tf.keras.layers.ZeroPadding2D(padding=1))
    self.backbone.add(tf.keras.layers.Conv2D(planes * 4, kernel_size=3,
                                             strides=1, use_bias=False

))
    self.backbone.add(tf.keras.layers.BatchNormalization())
    self.backbone.add(tf.keras.layers.ReLU())
    self.backbone.add(tf.keras.layers.ZeroPadding2D(padding=1))
    self.backbone.add(tf.keras.layers.Conv2D(planes * 4, kernel_size=3,
                                             strides=[1,1], use_bias=False

))
    self.backbone.add(tf.keras.layers.BatchNormalization())
    self.backbone.add(tf.keras.layers.ReLU())
    self.backbone.add(tf.keras.layers.ZeroPadding2D(padding=1))
    self.backbone.add(tf.keras.layers.Conv2D(planes * 8, kernel_size=3,
                                             strides=[1,1], use_bias=False

))
    self.backbone.add(tf.keras.layers.BatchNormalization())
    self.backbone.add(tf.keras.layers.ReLU())

    self.conv_layers = []
    self.norm_layers = []

    ####
    # ConvCaps Layers
    for d in range(1, depth):
      if self.mode == 'DR':
        self.conv_layers.append(
          DynamicRouting2d(num_caps, num_caps, caps_size, caps_size,
                           kernel_size=3, stride=1, padding=1))
        #nn.init.normal_(self.conv_layers[0].W, 0, 0.5)
      elif self.mode == 'EM':
        self.conv_layers.append(
          EmRouting2d(num_caps, num_caps, caps_size, kernel_size=3, stride=1,
                      padding=1))
        self.norm_layers.append(tf.keras.layers.BatchNormalization())
      elif self.mode == 'SR':
        self.conv_layers.append(
          SelfRouting2d(num_caps, num_caps, caps_size, caps_size, kernel_size=3,
                        stride=1, padding=1, pose_out=True))
        self.norm_layers.append(tf.keras.layers.BatchNormalization())
      else:
        break

    final_shape = 4

    ####
    # Routings: ConvCaps to Class Capsules
    if self.mode in ['DR', 'EM', 'SR']:
      self.conv_pose = tf.keras.Sequential()
      self.conv_pose.add(tf.keras.layers.ZeroPadding2D(padding=1))
      self.conv_pose.add(tf.keras.layers.Conv2D(num_caps * caps_size,
                                                kernel_size=3,
                                                strides=1, use_bias=False))
      self.bn_pose = tf.keras.layers.BatchNormalization()
      if self.mode == 'DR':
        self.fc = DynamicRouting2d(num_caps, classes, caps_size, caps_size,
                                   kernel_size=final_shape, padding=0)
      elif self.mode in ['EM', 'SR']:
        self.conv_a = tf.keras.Sequential()
        self.conv_a.add(tf.keras.layers.ZeroPadding2D(padding=1))
        self.conv_a.add(tf.keras.layers.Conv2D(num_caps, kernel_size=3,
                                               strides=1, use_bias=False))
        self.bn_a = tf.keras.layers.BatchNormalization()

        if self.mode == 'EM':
          self.fc = EmRouting2d(num_caps, classes, caps_size,
                                kernel_size=final_shape, padding=0)
        else:
          self.fc = SelfRouting2d(num_caps, classes, caps_size, 1,
                                  kernel_size=final_shape, padding=0,
                                  pose_out=False)
    else:
      self.fc = tf.keras.layers.Dense(classes)
      # avg pooling
      if self.mode == 'AVG':
        self.pool = tf.keras.layers.AveragePooling2D(final_shape)

      # max pooling
      if self.mode == 'MAX':
        self.pool = tf.keras.layers.MaxPool2D(final_shape)

      # What is this?
      if self.mode == 'FC':
        self.conv_ = tf.keras.Sequential()
        self.conv_.add(tf.keras.layers.ZeroPadding2D(padding=(1,1)))
        self.conv_.add(tf.keras.layers.Conv2D(num_caps * caps_size,
                                              kernel_size=3, strides=1,
                                              use_bias=False))
        self.bn_ = tf.keras.layers.BatchNormalization()

  def call(self, inputs, **kwargs):
    # backbone
    out = self.backbone(inputs)

    # DR
    if self.mode == 'DR':
      pose = self.bn_pose(self.conv_pose(out))

      b, c, h, w = tf.shape(pose)
      pose = tf.transpose(pose, [0, 2, 3, 1])
      pose = squash(tf.reshape(pose(b, h, w, self.num_caps, self.caps_size)))
      pose = tf.reshape(pose, [b, h, w, -1])
      pose = tf.transpose(pose, [0, 3, 1, 2])

      for m in self.conv_layers:
        pose = m(pose)

      out = self.fc(pose)
      out = tf.reshape(out, [b, -1, self.caps_size])
      out = length(out, axis=-1)

    # Routing
    elif self.mode in ['EM', 'SR']:
      a, pose = self.conv_a(out), self.conv_pose(out)
      a, pose = tf.nn.sigmoid(self.bn_a(a)), self.bn_pose(pose)

      for m, bn in zip(self.conv_layers, self.norm_layers):
        a, pose = m(a, pose)
        pose = bn(pose)

      a, _ = self.fc(a, pose) # we don't use pose anymore, it only for routing?
      out = tf.reshape(a, [tf.shape(a)[0], -1])

      if self.mode == 'SR':
        out = tf.math.log(out)

    # No Routing
    else:
      if self.mode in ['AVG', 'MAX']:
        out = self.pool(out)
      elif self.mode == 'FC':
        out = tf.nn.relu(self.bn_(self.conv_(out)))
      out = tf.reshape(out, [tf.shape(out)[0], -1])
      out = self.fc(out)

    return out


class SmallNet(tf.keras.layers.Layer):
  def __init__(self, conf, cfg_data):
    super(SmallNet, self).__init__()
    # most of configuration is hard coded.
    _, classes = cfg_data["channels"], cfg_data["classes"]
    self.conv1 = tf.keras.Sequential()
    self.conv1.add(tf.keras.layers.ZeroPadding2D(padding=1))
    self.conv1.add(tf.keras.layers.Conv2D(256,
                                          kernel_size=[7, 7], strides=[2, 2],
                                          use_bias=False))
    self.bn1 = tf.keras.layers.BatchNormalization()

    self.mode = conf["mode"]
    assert self.mode in ['DR', 'EM', 'SR', 'AVG', 'MAX', 'FC']

    self.num_caps = 16

    planes = 16 # channels
    last_size = 6

    if self.mode == 'SR':
      self.conv_a = tf.keras.Sequential()
      self.conv_a.add(tf.keras.layers.ZeroPadding2D(padding=1))
      self.conv_a.add(tf.keras.layers.Conv2D(self.num_caps,
                                            kernel_size=[5, 5], strides=1,
                                            use_bias=False))
      self.conv_pose = tf.keras.Sequential()
      self.conv_pose.add(tf.keras.layers.ZeroPadding2D(padding=1))
      self.conv_pose.add(tf.keras.layers.Conv2D(self.num_caps * planes,
                                            kernel_size=[5, 5], strides=1,
                                            use_bias=False))
      self.bn_a = tf.keras.layers.BatchNormalization()
      self.bn_pose = tf.keras.layers.BatchNormalization()

      self.conv_caps = SelfRouting2d(self.num_caps, self.num_caps, planes, planes,
                                     kernel_size=3, strides=2, padding=1,
                                     pose_out=True)
      self.bn_pose_conv_caps = tf.keras.layers.BatchNormalization()

      self.fc_caps = SelfRouting2d(self.num_caps, classes, planes, 1,
                                   kernel_size=last_size, padding=0, pose_out=False)

    elif self.mode == 'DR':
      self.conv_pose = tf.keras.Sequential()
      self.conv_pose.add(tf.keras.layers.ZeroPadding2D(padding=1))
      self.conv_pose.add(tf.keras.layers.Conv2D(self.num_caps * planes,
                                            kernel_size=[5, 5], strides=1,
                                            use_bias=False))

      self.conv_caps = DynamicRouting2d(self.num_caps, self.num_caps, 16, 16,
                                        kernel_size=3, strides=2, padding=1,
                                        std_dev=0.5)

      self.fc_caps = DynamicRouting2d(self.num_caps, classes, 16, 16,
                                      kernel_size=last_size, padding=0,
                                      stf_dev=0.05)

    elif self.mode == 'EM':
      self.conv_a = tf.keras.Sequential()
      self.conv_a.add(tf.keras.layers.ZeroPadding2D(padding=1))
      self.conv_a.add(tf.keras.layers.Conv2D(self.num_caps,
                                            kernel_size=[5, 5], strides=1,
                                            use_bias=False))
      self.conv_pose = tf.keras.Sequential()
      self.conv_pose.add(tf.keras.layers.ZeroPadding2D(padding=1))
      self.conv_pose.add(tf.keras.layers.Conv2D(self.num_caps * planes,
                                            kernel_size=[5, 5], strides=1,
                                            use_bias=False))
      self.bn_a = tf.keras.layers.BatchNormalization()
      self.bn_pose = tf.keras.layers.BatchNormalization()

      self.conv_caps = EmRouting2d(self.num_caps, self.num_caps, 16, kernel_size=3,
                                   strides=2, padding=1)
      self.bn_pose_conv_caps = tf.keras.layers.BatchNormalization()

      self.fc_caps = EmRouting2d(self.num_caps, classes, 16, kernel_size=last_size,
                                 padding=0)

    else:
      raise NotImplementedError

  def call(self, inputs, **kwargs):
    out = tf.nn.relu(self.bn1(self.conv1(inputs)))

    if self.mode == 'DR':
      pose = self.conv_pose(out)

      b, c, h, w = tf.shape(pose)
      pose = tf.transpose([0, 2, 3, 1])
      pose = squash(tf.reshape(pose, [b, h, w, self.num_caps, 16]))
      pose = tf.reshape(pose, [b, h, w, -1])
      pose = tf.transpose(pose, [0, 3, 1, 2])

      pose = self.conv_caps(pose)

      out = self.fc_caps(pose)
      out = tf.reshape(out, [b, -1, 16])
      out = length(out, axis=-1)

    elif self.mode == 'EM':
      a, pose = self.conv_a(out), self.conv_pose(out)
      a, pose = tf.nn.sigmoid(self.bn_a(a)), self.bn_pose(pose)
      a, pose = self.conv_caps(a, pose)
      pose = self.bn_pose_conv_caps(pose)

      a, _ = self.fc_caps(a, pose)
      out = tf.reshape(a, [tf.shape(out)[0], -1])

    elif self.mode == 'SR':
      a, pose = self.conv_a(out), self.conv_pose(out)
      a, pose = tf.nn.sigmoid(self.bn_a(a)), self.bn_pose(pose)

      a, pose = self.conv_caps(a, pose)
      pose = self.bn_pose_conv_caps(pose)

      a, _ = self.fc_caps(a, pose)

      out = tf.reshape(a, [tf.shape(out)[0], -1])
      out = tf.math.log(out)

    return out
