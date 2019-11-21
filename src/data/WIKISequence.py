from tensorflow.keras.utils import Sequence

class WIKISequence(Sequence):
    """Base object for fitting to a sequence of data, such as a dataset.
    Every `Sequence` must implement the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement
    `on_epoch_end`.
    The method `__getitem__` should return a complete batch.

    Example: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __getitem__(self, index):
      """Gets batch at position `index`.
      Arguments:
          index: position of the batch in the Sequence.
      Returns:
          A batch
      """
      raise NotImplementedError

    def __len__(self):
      """Number of batch in the Sequence.
      Returns:
          The number of batches in the Sequence.
      """
      raise NotImplementedError

    def on_epoch_end(self):
      """Method called at the end of every epoch.
      """
      pass
