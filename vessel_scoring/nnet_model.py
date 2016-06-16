from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time

def get_polynomial_cols(x, windows):
    colnames = []
    #colnames.append("speed")
    for window in windows:
        colnames.append('measure_speedavg_%s' % window)
        colnames.append('measure_speedstddev_%s_log' % window)
        colnames.append('measure_coursestddev_%s_log' % window)
    cols = [x[col] for col in colnames]
    return cols



# Some of this code comes from Google Tensor flow demo: check license
# and license this appropriately or rewrite from scratch.

# Much of this based one:
# https://github.com/tensorflow/tensorflow/blob/r0.9/tensorflow/examples/tutorials/mnist/fully_connected_feed.py

LEARNING_RATE = 1.0
MAX_EPOCHS = 20
HIDDEN = 128
BATCH_SIZE = 128
TRAIN_DIR = "."
DECAY_SCALE = 0.95
# Basic model parameters as external flags.
# flags = tf.app.flags
#
#
#
#
# FLAGS = flags.FLAGS
# flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
# flags.DEFINE_integer('max_steps', 200, 'Number of steps to run.')
# flags.DEFINE_integer('hidden', 1024, 'Number of units in hidden layers.')
# flags.DEFINE_integer('batch_size', 100, 'Batch size.  ' # XXX
#                      'Must divide evenly into the dataset sizes.')

N_WINDOWS = 6
N_BASE_FEATURES = 3
N_FEATURES = N_WINDOWS * N_BASE_FEATURES

def leaky_relu(x, alpha=0.01):
    return tf.maximum(alpha*x,x)

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    featurse_placeholder: Features placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  features_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           N_FEATURES))
  labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
  return features_placeholder, labels_placeholder


def fill_feed_dict(data_set, features_pl, labels_pl):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of features and labels, from input_data.read_data_sets()
    features_pl: The features placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  features_feed, labels_feed = data_set.next_batch(BATCH_SIZE)
  feed_dict = {
      features_pl: features_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_eval(sess,
            eval_correct,
            features_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    features_placeholder: The features placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of features and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // BATCH_SIZE
  num_examples = steps_per_epoch * BATCH_SIZE
  for step in range(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               features_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))



def maxout(tensor, pool_size):
    """Apply maxout to a dense (batch_size, n_features) tensor
    """
    batch_size = tf.shape(tensor)[0]
    return tf.reshape(
                tf.max_pool(
                    tf.reshape(tensor, (batch_size, 1, 1, -1)),
                    (1, 1, 1, pool_size)),
                (batch_size, -1))




def inference(features, hidden_units):
    """Build the model up to where it may be used for inference.
    Args:
    features: features placeholder, from inputs().
    hidden_units: Size of the hidden layers.
#     maxout_pooling: Degree of maxout to use
    Returns:
    softmax_linear: Output tensor with the computed logits.
    """


    batch_size = tf.shape(features)[0]
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([N_FEATURES, hidden_units],
                                stddev=1.0 / np.sqrt(N_FEATURES)),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden_units]),
                             name='biases')
        hidden1 = leaky_relu(tf.matmul(features, weights) + biases)
    # Dropout 1
    dropout1 = tf.nn.dropout(hidden1, 0.5)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden_units, hidden_units],
                                stddev=1.0 / np.sqrt(hidden_units)),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden_units]),
                             name='biases')
        hidden2 = leaky_relu((tf.matmul(dropout1, weights) + biases))
    # Dropout2
    dropout2 = tf.nn.dropout(hidden2, 0.5)
    # Linear
    with tf.name_scope('logit'):
        weights = tf.Variable(
            tf.truncated_normal([hidden_units, 1],
                                stddev=1.0 / np.sqrt(hidden_units)),
            name='weights')
        biases = tf.Variable(tf.zeros([1]),
                             name='biases')
        logits = tf.reshape(tf.matmul(dropout2, weights) + biases, (-1,))
    return logits



def lossfunc(logits, labels):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss


def training(loss, learning_rate):
  """Sets up the training Ops.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size].
    labels: Labels tensor, float - [batch_size]
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.equal(tf.round(tf.sigmoid(logits)), labels)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))



class DataSet(object):

  def __init__(self,
               features,
               labels):
    """Construct a DataSet.
    """
    dtype = 'float32'

    assert features.shape[0] == labels.shape[0], (
      'features.shape: %s labels.shape: %s' % (features.shape, labels.shape))
    self._num_examples = features.shape[0]
    self._features = features
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def features(self):
    return self._features

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._features = self._features[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._features[start:end], self._labels[start:end]



def run_training(train_ds, eval_ds):
  """Train for a number of steps."""
  # Get the sets of features and labels for training, validation, and
  # test on .
#   data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)




  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the features and labels.
    features_placeholder, labels_placeholder = placeholder_inputs(
        BATCH_SIZE)

    # Build a Graph that computes predictions from the inference model.
    logits = inference(features_placeholder, HIDDEN)

    # Build a final output prediction
    predictions = tf.nn.sigmoid(logits)

    # Add to the Graph the Ops for loss calculation.
    loss = lossfunc(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    learning_rate = tf.Variable(LEARNING_RATE, name="learning_rate")
    #
    train_op = training(loss, learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = evaluation(logits, labels_placeholder)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(TRAIN_DIR, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    epoch = 0
    last_epoch = 0
    step = 0
    while epoch < MAX_EPOCHS:
      try:
          start_time = time.time()

          # Fill a feed dictionary with the actual set of features and labels
          # for this particular training step.
          feed_dict = fill_feed_dict(train_ds,
                                     features_placeholder,
                                     labels_placeholder)

          # Run one step of the model.  The return values are the activations
          # from the `train_op` (which is discarded) and the `loss` Op.  To
          # inspect the values of your Ops or variables, you may include them
          # in the list passed to sess.run() and the value tensors will be
          # returned in the tuple from the call.
          _, loss_value = sess.run([train_op, loss],
                                   feed_dict=feed_dict)

          duration = time.time() - start_time

          # Write the summaries and print an overview fairly often.
          if step % 100 == 0:
            # Print status to stdout.
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            # Update the events file.
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()


          epoch = (step * BATCH_SIZE) // train_ds.num_examples
          if epoch != last_epoch or epoch >= MAX_EPOCHS:
            learning_rate.assign(0.95 * learning_rate)
            # Save a checkpoint and evaluate the model .
            saver.save(sess, TRAIN_DIR, global_step=step)
            # Evaluate against the training set.
            print("Epoch:", epoch)
            print('Training Data Eval:')
            do_eval(sess,
                    eval_correct,
                    features_placeholder,
                    labels_placeholder,
                    train_ds)
            # Evaluate against the validation set.
            print('Validation Data Eval:')
            do_eval(sess,
                    eval_correct,
                    features_placeholder,
                    labels_placeholder,
                    eval_ds)
            # Evaluate against the test set.
    #         print('Test Data Eval:')
    #         do_eval(sess,
    #                 eval_correct,
    #                 features_placeholder,
    #                 labels_placeholder,
    #                 data_sets.test)
          last_epoch = epoch
          step += 1
      except:
          break

    return sess, logits, predictions, features_placeholder, labels_placeholder



def complete_batch(x):
    n = len(x)
    assert n > BATCH_SIZE // 2 # This limitation can be fixed
    if n % BATCH_SIZE == 0:
        return x
    else:
        while len(x) < BATCH_SIZE // 2:
            x = np.concatenate([x, x], axis=0)
        extra = BATCH_SIZE - n % BATCH_SIZE
        return np.concatenate([x, x[:extra]], axis=0)

class NNetModel:

    windows = ['10800', '1800', '21600', '3600', '43200', '86400']

    def __init__(self, **args):
        """
        windows - list of window sizes to use in features
        See RandomForestClassifier docs for other parameters.
        """
        self.ses = None
#         if args:
#             for key in args:
#                 args[key] = [np.array(item) for item in args[key]]
#             self.net.load_params_from(args)

    def dump_arg_dict(self):
        raise NotImplementedError()
#         def convert(data):
#             if isinstance(data, np.ndarray):
#                 return data.tolist()
#             elif isinstance(data, (list, tuple)):
#                 return [convert(item) for item in data]
#             elif isinstance(data, dict):
#                 return {key: convert(value)
#                         for key, value in data.items()}
#             else:
#                 return data
#         return convert(dict(self.get_all_param_values()))

    def _make_features(self, X):
        x = np.transpose(get_polynomial_cols(X, self.windows))
        return (x.astype('float32') - self.mean) / self.std

    def predict_proba(self, X):
        X = self._make_features(X)
        y = np.zeros([len(X), 2], dtype='float32')
        #
        sess, logits, predictions, features_placeholder, labels_placeholder = self.info
        X1 = complete_batch(X)
        ds = DataSet(X1,
                        complete_batch(y[:,0]) # This is kludgy to pass y in,
                                          # there is likely a way to pass
                                          # in only the feature vector and
                                          # not both
                        )
        chunks = []
        steps = len(X1) // BATCH_SIZE
        assert len(X1) % BATCH_SIZE == 0
        for step in range(steps):
            feed_dict = fill_feed_dict(ds,
                                       features_placeholder,
                                       labels_placeholder)

            chunks.append(sess.run(predictions, feed_dict=feed_dict))
        ps = np.concatenate(chunks)

        #
        y[:,1] = ps.reshape(-1)[:len(X)]
        y[:,0] = 1 - y[:,1]
        return y

    def fit(self, X, y):
        self.mean = 0
        self.std = 1
        X = self._make_features(X)
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.mean(axis=0, keepdims=True)
        X = (X - self.mean) / self.std
        #
        n = len(X)
        n_train = int(DECAY_SCALE * n)
        inds = np.arange(n)
        np.random.shuffle(inds)
        #
        train_ds = DataSet(X[inds[:n_train]], y[inds[:n_train]])
        eval_ds = DataSet(X[inds[n_train:]], y[inds[n_train:]])
        self.info = run_training(train_ds, eval_ds)

        return self


# def main(_):
#   run_training()
#
#
# if __name__ == '__main__':
#   tf.app.run()