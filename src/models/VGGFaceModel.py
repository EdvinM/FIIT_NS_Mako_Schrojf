import tensorflow as tf
import tensorflow.keras as keras


def create_vgg_face_model():
    model = keras.models.Sequential()

    model.add(keras.layers.ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(keras.layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.ZeroPadding2D((1, 1)))
    model.add(keras.layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(keras.layers.ZeroPadding2D((1, 1)))
    model.add(keras.layers.Convolution2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.ZeroPadding2D((1, 1)))
    model.add(keras.layers.Convolution2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(keras.layers.ZeroPadding2D((1, 1)))
    model.add(keras.layers.Convolution2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.ZeroPadding2D((1, 1)))
    model.add(keras.layers.Convolution2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.ZeroPadding2D((1, 1)))
    model.add(keras.layers.Convolution2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(keras.layers.ZeroPadding2D((1, 1)))
    model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(keras.layers.ZeroPadding2D((1, 1)))
    model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(keras.layers.ZeroPadding2D((1, 1)))
    model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(keras.layers.ZeroPadding2D((1, 1)))
    model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(keras.layers.ZeroPadding2D((1, 1)))
    model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(keras.layers.ZeroPadding2D((1, 1)))
    model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(keras.layers.Convolution2D(4096, (7, 7), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Convolution2D(4096, (1, 1), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Convolution2D(2622, (1, 1)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Activation('softmax'))

    return model


def load_pretrained_weights(model):
    model.load_weights('../data/vgg_face_weights.h5')

# todo
