import tensorflow.keras as keras
from .MiniVGGNetModel import MiniVGGNetModel


def create_new_model(classes, learning_rate=0.001, loss='sparse_categorical_crossentropy', metrics='accuracy'):
    model = MiniVGGNetModel(classes=classes)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[metrics])

    return model


def create_new_regression_model(learning_rate=0.001):
    model = MiniVGGNetModel(classes=None)

    optimizer = keras.optimizers.RMSprop(learning_rate)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse']
                  )

    return model
