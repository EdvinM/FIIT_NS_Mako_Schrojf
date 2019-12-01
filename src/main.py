import cv2
import math
import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import sys, getopt

from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras.layers import Activation, BatchNormalization, Dropout, ZeroPadding2D, Convolution2D

class VGGModel():
	def __init__(self):
		self.model = Sequential()
		self.model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
		self.model.add(Convolution2D(64, (3, 3), activation='relu'))
		self.model.add(ZeroPadding2D((1,1)))
		self.model.add(Convolution2D(64, (3, 3), activation='relu'))
		self.model.add(MaxPooling2D((2,2), strides=(2,2)))
		 
		self.model.add(ZeroPadding2D((1,1)))
		self.model.add(Convolution2D(128, (3, 3), activation='relu'))
		self.model.add(ZeroPadding2D((1,1)))
		self.model.add(Convolution2D(128, (3, 3), activation='relu'))
		self.model.add(MaxPooling2D((2,2), strides=(2,2)))
		 
		self.model.add(ZeroPadding2D((1,1)))
		self.model.add(Convolution2D(256, (3, 3), activation='relu'))
		self.model.add(ZeroPadding2D((1,1)))
		self.model.add(Convolution2D(256, (3, 3), activation='relu'))
		self.model.add(ZeroPadding2D((1,1)))
		self.model.add(Convolution2D(256, (3, 3), activation='relu'))
		self.model.add(MaxPooling2D((2,2), strides=(2,2)))
		 
		self.model.add(ZeroPadding2D((1,1)))
		self.model.add(Convolution2D(512, (3, 3), activation='relu'))
		self.model.add(ZeroPadding2D((1,1)))
		self.model.add(Convolution2D(512, (3, 3), activation='relu'))
		self.model.add(ZeroPadding2D((1,1)))
		self.model.add(Convolution2D(512, (3, 3), activation='relu'))
		self.model.add(MaxPooling2D((2,2), strides=(2,2)))
		 
		self.model.add(ZeroPadding2D((1,1)))
		self.model.add(Convolution2D(512, (3, 3), activation='relu'))
		self.model.add(ZeroPadding2D((1,1)))
		self.model.add(Convolution2D(512, (3, 3), activation='relu'))
		self.model.add(ZeroPadding2D((1,1)))
		self.model.add(Convolution2D(512, (3, 3), activation='relu'))
		self.model.add(MaxPooling2D((2,2), strides=(2,2)))
		 
		self.model.add(Convolution2D(4096, (7, 7), activation='relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Convolution2D(4096, (1, 1), activation='relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Convolution2D(2622, (1, 1)))
		self.model.add(Flatten())
		self.model.add(Activation('softmax'))

	def load_weights(self, weights_file='../../data/vgg_face_weights.h5'):
		self.model.load_weights(weights_file)

		for layer in self.model.layers[:-7]:
			layer.trainable = False

	def create(self, classes):
		base_model_output = Sequential()
		base_model_output = Convolution2D(classes, (1, 1), name='predictions')(self.model.layers[-4].output)
		base_model_output = Flatten()(base_model_output)
		base_model_output = Activation('softmax')(base_model_output)
		 
		return keras.Model(inputs=self.model.input, outputs=base_model_output)


class MiniVGGNetModel(keras.Model):
    def __init__(self, classes, chanDim=-1):
        # call the parent constructor
        super(MiniVGGNetModel, self).__init__()

        # initialize the layers in the first (CONV => RELU) * 2 => POOL
        # layer set
        self.conv1A = Conv2D(32, (3, 3), padding="same")
        self.act1A = Activation("relu")
        self.bn1A = BatchNormalization(axis=chanDim)
        self.conv1B = Conv2D(32, (3, 3), padding="same")
        self.act1B = Activation("relu")
        self.bn1B = BatchNormalization(axis=chanDim)
        self.pool1 = MaxPooling2D(pool_size=(2, 2))

        # initialize the layers in the second (CONV => RELU) * 2 => POOL
        # layer set
        self.conv2A = Conv2D(32, (3, 3), padding="same")
        self.act2A = Activation("relu")
        self.bn2A = BatchNormalization(axis=chanDim)
        self.conv2B = Conv2D(32, (3, 3), padding="same")
        self.act2B = Activation("relu")
        self.bn2B = BatchNormalization(axis=chanDim)
        self.pool2 = MaxPooling2D(pool_size=(2, 2))

        # initialize the layers in our fully-connected layer set
        self.flatten = Flatten()
        self.dense3 = Dense(512)
        self.act3 = Activation("relu")
        self.bn3 = BatchNormalization()
        self.do3 = Dropout(0.5)

        # initialize the layers in the softmax classifier layer set
        self.dense4 = Dense(classes)
        self.softmax = Activation("softmax")

    def call(self, inputs):
        # build the first (CONV => RELU) * 2 => POOL layer set
        x = self.conv1A(inputs)
        x = self.act1A(x)
        x = self.bn1A(x)
        x = self.conv1B(x)
        x = self.act1B(x)
        x = self.bn1B(x)
        x = self.pool1(x)

        # build the second (CONV => RELU) * 2 => POOL layer set
        x = self.conv2A(x)
        x = self.act2A(x)
        x = self.bn2A(x)
        x = self.conv2B(x)
        x = self.act2B(x)
        x = self.bn2B(x)
        x = self.pool2(x)

        # build our FC layer set
        x = self.flatten(x)
        x = self.dense3(x)
        x = self.act3(x)
        x = self.bn3(x)
        x = self.do3(x)

        # build the softmax classifier
        x = self.dense4(x)
        x = self.softmax(x)

        # return the constructed model
        return x


class WIKISequence(Sequence):
    """Base object for fitting to a sequence of data, such as a dataset.
    Every `Sequence` must implement the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement
    `on_epoch_end`.
    The method `__getitem__` should return a complete batch.

    Example: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """

    def load_img(self, file_path):
        """Load single image from disk and resize and convert to np array
        :return:
        """
        im = cv2.imread(self.base_path + file_path)
        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return (np.array(im) / 255.0).astype(np.float32)

    def __init__(self, x, y, batch_size, base_path='../../data/raw/wiki_crop/'):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.base_path = base_path

    def __getitem__(self, idx):
        """Gets batch at position `index`.
        Arguments:
            index: position of the batch in the Sequence.
        Returns:
            A batch
        """
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([self.load_img(full_path) for full_path in batch_x['full_path']]), np.array(batch_y['age'])

    def __len__(self):
        """Number of batch in the Sequence.
        Returns:
            The number of batches in the Sequence.
        """
        return math.ceil(len(self.x) / self.batch_size)

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass


class Model():

    def __init__(self, df_path, y_column_name):

        self.df = pd.read_csv(df_path, sep=';')

        if self.df is None:
            print("Unable to open dataframe")
            return

        if y_column_name not in self.df.columns:
            print("Column not found in dataframe columns")
            return

        self.x = self.df.loc[:, self.df.columns != y_column_name]
        self.y = self.df.loc[:, self.df.columns == y_column_name]

        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []

        self.model = None

    def split_df(self, test_size=0.3):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y, test_size=test_size)

        print("Train X length = " + str(len(self.train_x)))
        print("Train Y length = " + str(len(self.train_y)))
        print("Test X length = " + str(len(self.test_x)))
        print("Test Y length = " + str(len(self.test_y)))

    def compile(self, keras_model,
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']):

        self.model = keras_model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return self.model

    def train_data(self):
        return (self.train_x, self.train_y)

    def test_data(self):
        return (self.test_x, self.test_y)


class Train():
    def __init__(self, model, logs_path='logs'):
        self.model = model

        self.callbacks = callbacks = [
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(logs_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
                histogram_freq=1,
                profile_batch=0)
        ]

    def start(self, epochs=1, batch_size=16):
        train_seq = WIKISequence(
            self.model.train_data()[0],
            self.model.train_data()[1], batch_size=batch_size)
        test_seq = WIKISequence(
            self.model.test_data()[0],
            self.model.test_data()[1], batch_size=batch_size)

        print("Train sequence data length= " + str(len(train_seq)))
        print("Test sequence data length= " + str(len(test_seq)))

        history = self.model.model.fit_generator(
            train_seq,
            epochs=epochs,
            validation_data=test_seq,
            callbacks=self.callbacks
        )

        print("===== Training Summary =====")
        print(history)
        print("Accuracy: " + history.history['acc'])
        print("Validation Accuracy: " + history.history['val_acc'])
        print("Loss: " + history.history['loss'])
        print("Validation Loss: " + history.history['val_loss'])

    def summary(self):
        print(self.model.summary())


class Predict():
    def __init__(self, model):
        self.model = model


    def load_img(self, file_path):
        """
        Load single image from disk and resize and convert to np array
        :return:
        """
        im = cv2.imread(self.base_path + file_path)
        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return (np.array(im) / 255.0).astype(np.float32)

    def predict(self, images):
        x = np.array()

        for image in images:
            x.append(self.load_img(image))

        predictions = self.model.predict(x)

        return [np.argmax(p) for p in predictions]


# Compile mini vgg model & create data
model = Model(df_path='../../data/processed/wiki_df.csv', y_column_name='age')
model.split_df()

vgg = VGGModel()
vgg.load_weights()

keras_model = model.compile(vgg.create(classes=101))

print("Model compilation successfull...")

# Train the model
train = Train(model)
train.start(epochs=int(sys.argv[0]), batch_size=int(sys.argv[1]))
train.summary()
train.save_model("full-vgg")

print("Model train successfull...")

# Predict data
predict = Predict(keras_model)
print("Prediction = " + str(predict.predict(model.train_data()[0])))
