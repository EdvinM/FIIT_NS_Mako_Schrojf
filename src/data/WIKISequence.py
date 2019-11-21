import cv2
import math
import numpy as np
from tensorflow.keras.utils import Sequence

class WIKISequence(Sequence):
    """Base object for fitting to a sequence of data, such as a dataset.
    Every `Sequence` must implement the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement
    `on_epoch_end`.
    The method `__getitem__` should return a complete batch.

    Example: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """

    def __init__(self, dataset_df, batch_size, base_path = '../data/raw/wiki_crop/'):
        self.base_path = base_path
        self.dataset_df = dataset_df
        self.batch_size = batch_size

    def load_img(self, file_path):
        """Load single image from disk and resize and convert to np array
        :return:
        """
        im = cv2.imread(self.base_path + file_path)
        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return (np.array(im) / 255.0).astype(np.float32)

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
        return math.ceil(len(self.dataset_df) / self.batch_size)

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass
