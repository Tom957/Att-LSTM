import os
import pickle
import tensorflow as tf

class Model(object):
  """Abstract object representing an Reader model."""
  def __init__(self):
    self.vocab = None
    self.data = None

  def save(self, checkpoint_dir, dataset_name, detail):
    self.saver = tf.train.Saver()

    model_name = type(self).__name__ or "Reader"
    if self.batch_size:
      model_dir = "%s%s_%s" % (dataset_name, self.batch_size, detail)
    else:
      model_dir = dataset_name

    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    print("[*] Saving checkpoints...",checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name))

  def load(self, checkpoint_dir, dataset_name, detail):
    self.saver = tf.train.Saver()

    if self.batch_size:
      model_dir = "%s%s_%s" % (dataset_name, self.batch_size, detail)
    else:
      model_dir = dataset_name
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    print("[*] Loading checkpoints...%s" % (checkpoint_dir))
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print("[*] Loading checkpoints...%s....SUCCESS" % (checkpoint_dir) )
      return True
    else:
      print("[*] Loading checkpoints...%s....FAILED!" % (checkpoint_dir) )
      return False

  def save_obj(self, fname, obj):
    with open(fname, 'wb') as f:
      pickle.dump(obj, f)

  def load_obj(fname):
    with open(fname, 'rb') as f:
      return pickle.load(f)