import tensorflow as tf
import tensorflow_datasets as tfds
import time
from tf_toys.capsnet.config import get_config
import tf_toys.capsnet.model as Model
from tf_toys.capsnet.helper_train import loss_nll, MultiStepLR

@tf.function
def train(inputs, model, optimizer, loss_func, loss_state, acc_state):
  x = tf.cast(inputs["image"], tf.float32)
  y = tf.cast(inputs["label"], tf.int32)

  with tf.GradientTape() as tape:
    out = model(x)
    loss = loss_func(y, out)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  # compute accuracy
  pred = tf.cast(tf.argmax(out, 1), tf.int32)
  correct = tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.int32))
  len_y = tf.shape(y)[0]
  acc = 100.0 * (tf.cast(correct, tf.float32) / tf.cast(len_y, tf.float32))
  loss_state.update_state(loss)
  acc_state.update_state(acc)

@tf.function
def evaluation(inputs, model, loss_func, loss_state, acc_state):
  x = tf.cast(inputs["image"], tf.float32)
  y = tf.cast(inputs["label"], tf.int32)

  out = model(x)
  loss = loss_func(y, out)

  # compute accuracy
  pred = tf.cast(tf.argmax(out, 1), tf.int32)
  correct = tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.int32))
  len_y = tf.shape(y)[0]
  acc = 100.0 * (tf.cast(correct, tf.float32) / tf.cast(len_y, tf.float32))
  loss_state.update_state(loss)
  acc_state.update_state(acc)


def main(config):
  #pylint: disable=too-many-locals
  """python3 main.py --dataset=cifar10 --name=resnet_[routing_method] --is_train=False"""
  #tf.random.set_seed(42)
  gpu_devices = tf.config.experimental.list_physical_devices('GPU')
  for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

  # data params
  data_builder = tfds.builder(config.dataset)
  data_builder.download_and_prepare()
  ds_train = data_builder.as_dataset(split="train")
  ds_test = data_builder.as_dataset(split="test")

  ds_train = ds_train.repeat(1).shuffle(1024).batch(config.batch_size)
  ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

  ds_test = ds_test.repeat(1).batch(config.batch_size)
  ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
  if config.dataset in ["cifar10", "svhn_cropped", "smallnorb"]:
    train_data_number = 60000
    test_data_number = 5000
  else:
    raise NotImplementedError
  n_iterations_per_epoch = train_data_number // config.batch_size
  n_iterations_test = n_iterations_validation = test_data_number // config.batch_size

  # training params
  epochs = config.epochs
  start_epoch = 0
  momentum = config.momentum
  weight_decay = config.weight_decay
  lr = config.init_lr

  # misc params
  best = config.best
  ckpt_dir = config.ckpt_dir
  logs_dir = config.logs_dir
  best_valid_acc = -1e13
  counter = 0
  train_patience = config.train_patience
  resume = config.resume
  print_freq = config.print_freq

  attack_type = config.attack_type
  attack_eps = config.attack_eps
  targeted = config.targeted

  name = config.name

  if config.name.endswith('dynamic_routing'):
    mode = 'DR'
  elif config.name.endswith('em_routing'):
    mode = 'EM'
  elif config.name.endswith('self_routing'):
    mode = 'SR'
  elif config.name.endswith('max'):
    mode = 'MAX'
  elif config.name.endswith('avg'):
    mode = 'AVG'
  elif config.name.endswith('fc'):
    mode = 'FC'
  else:
    raise NotImplementedError("Unknown model postfix")

  model = Model.create(name=name, conf=config, mode=mode)

  lr = 1e-3
  loss_func = None
  if mode in ['DR', 'EM', 'SR']:
    if config.dataset in ['cifar10', 'svhn_cropped']:
      print("using NLL loss, lr %.3f"%lr)
      loss_func = loss_nll
    elif config.dataset == "smallnorb":
      if mode == 'DR':
        print("using DR loss, lr %.3f"%lr)
        #self.loss = DynamicRoutingLoss().to(device)
        loss_func = None
      elif mode == 'EM':
        print("using EM loss, lr %.3f"%lr)
        #self.loss = EmRoutingLoss(self.epochs).to(device)
        loss_func = None
      elif mode == 'SR':
        print("using NLL loss, lr %.3f"%lr)
        loss_func = loss_nll
  else:
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    print("using CE loss, lr %.3f" % lr)

  # weight decay will role as regularizer like reconstruction loss in DR.
  opti = None
  print("mode: %s" % mode)
  if config.dataset == "cifar10":
    #MultiStepLR(lr/100.0, [150, 250], gamma=0.1)
    if mode in ["DR", "EM", "SR"]:
      opti = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=weight_decay)
    elif mode in ["AVG", "MAX", "FC"]:
      opti = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=weight_decay)
      #opti = tf.keras.optimizers.SGD(lr=1e-4, momentum=momentum,
      #                               decay=weight_decay)
  elif config.dataset == "svhn_cropped":
    opti = tf.keras.optimizers.SGD(MultiStepLR(lr, [100, 150], gamma=0.1),
                                   momentum=momentum, decay=weight_decay)
  elif config.dataset == "smallnorb":
    opti = tf.keras.optimizers.SGD(MultiStepLR(lr, [100, 150], gamma=0.1),
                                   momentum=momentum, decay=weight_decay)

  # Training
  for epoch in range(start_epoch, epochs + 1):
    train_avg_loss = tf.keras.metrics.Mean(name='train_loss')
    train_avg_accu = tf.keras.metrics.Mean(name='train_acc')
    start_time = time.time()
    for i, datum in enumerate(iter(ds_train)):
      idx = i + 1
      train(datum, model, opti, loss_func, train_avg_loss, train_avg_accu)
      loss = train_avg_loss.result()
      acc = train_avg_accu.result()
      print('\rTraining the model: %d/%d Loss %.3f Acc %.3f' %
            (idx, n_iterations_per_epoch, loss, acc), end="")
    print("\nEpoch: %d Loss %.3f, Acc %.3f, %.3f secs "
          "elapsed"%(epoch + 1, train_avg_loss.result(),
                     train_avg_accu.result(), (time.time() - start_time)))

    valid_avg_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_avg_accu = tf.keras.metrics.Mean(name='valid_acc')
    # At the end of each epoch,
    # measure the validation loss and accuracy:
    for i, datum in enumerate(iter(ds_test)):
      idx = i + 1
      evaluation(datum, model, loss_func, valid_avg_loss, valid_avg_accu)
      print("\rEvaluating the model: {}/{} ({:.1f}%)".format(idx, n_iterations_validation, idx * 100 / n_iterations_validation),
        end=" " * 10)
    print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
      epoch + 1, valid_avg_accu.result(), valid_avg_loss.result(),
      " (improved)" if valid_avg_loss.result() < best_valid_acc else ""))

    if valid_avg_loss.result() < best_valid_acc:
      best_valid_acc = valid_avg_loss.result()

  print("Evaluation!")
  # Evaluation
  test_avg_loss = tf.keras.metrics.Mean(name='test_loss')
  test_avg_accu = tf.keras.metrics.Mean(name='test_acc')
  for i, datum in enumerate(iter(ds_test)):
    idx = i + 1
    evaluation(datum, model, loss_func, test_avg_loss, test_avg_accu)
    print("\rEvaluating the model: {}/{} ({:.1f}%)".format(idx,
                                                           n_iterations_test,
                                                           idx * 100 / n_iterations_test),
          end=" " * 10)
  print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
    test_avg_accu.result(), test_avg_loss.result()))


if __name__ == "__main__":
  config, unparsed = get_config()
  main(config)
