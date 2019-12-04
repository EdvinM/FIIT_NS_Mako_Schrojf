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


def build_new_class_model(classes, learning_rate=0.001, loss='sparse_categorical_crossentropy', metrics='accuracy',
                          lock_layers=True):
    vgg_face_model = create_vgg_face_model()
    load_pretrained_weights(vgg_face_model)

    if lock_layers:
        for layer in vgg_face_model.layers[:-7]:
            layer.trainable = False

    # Replace last 4 layers with classes and softmax
    face_model_output = keras.layers.Convolution2D(classes, (1, 1), name='predictions')(vgg_face_model.layers[-4].output)
    face_model_output = keras.layers.Flatten()(face_model_output)
    face_model_output = keras.layers.Activation('softmax')(face_model_output)

    age_model = keras.Model(inputs=vgg_face_model.input, outputs=face_model_output)

    age_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[metrics])

    return age_model


def build_new_age_model(learning_rate=0.001, loss='sparse_categorical_crossentropy', metrics='accuracy',
                        lock_layers=True):
    return build_new_class_model(101, learning_rate, loss, metrics, lock_layers)


def build_new_gender_model(learning_rate=0.001, loss='sparse_categorical_crossentropy', metrics='accuracy',
                           lock_layers=True):
    return build_new_class_model(2, learning_rate, loss, metrics, lock_layers)
