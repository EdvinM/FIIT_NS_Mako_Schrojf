import os
from datetime import datetime

import tensorflow as tf
import tensorflow.keras as keras

import models.VGGFaceModel as VGGFaceModel

import data.load_data as load_data

from data.IMDBSequence import IMDBSequence
from data.WIKISequence import WIKISequence

from pathlib import Path
# BASE_PATH = Path(__file__).parent.absolute()
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_PATH)

MODEL_NAME = 'VGG_FACE_AGE_PREDICT'

now = datetime.now()  # current date and time
datetime_stamp = now.strftime("%Y-%m-%d__%H-%M-%S")
print("= STARTED", datetime_stamp)

# Keras callbacks
logs_path = "../logs/"

if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    sys.path.insert(0, os.getcwd())

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(logs_path, MODEL_NAME, datetime_stamp),
            histogram_freq=1,
            profile_batch=0),

        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join("../models/", MODEL_NAME, "checkpoints", "age_model.hdf5"),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode='auto')
    ]

    if not os.path.isfile('../data/vgg_face_weights.h5'):
        raise Exception("File with pretrained model weight not available in: '../data/vgg_face_weights.h5'")

    # Build new model with loaded weights
    vgg_face_age_model = VGGFaceModel.build_new_age_model()
    print("= NEW VGG-FACE MODEL WITH PRETRAINED WEIGHTS WAS CREATED\n")

    # Load dataset
    wiki_df = load_data.load_wiki_df_from_csv('../data/processed/wiki_df.csv')
    print("= " + str(len(wiki_df)) + " ROWS OF WIKI DATA LOADED\n")

    # Split dataset by ratio: 0.7 / 0.3
    BATCH_SIZE = 64  # Todo: check if GPU has enough memory

    wiki_generator_train = WIKISequence(wiki_df[0:15052], 'age', BATCH_SIZE)
    wiki_generator_test = WIKISequence(wiki_df[15052:22578], 'age', BATCH_SIZE)

    # Train model
    EPOCH = 200

    history = vgg_face_age_model.fit_generator(
        wiki_generator_train,
        epochs=EPOCH,
        validation_data=wiki_generator_test,
        callbacks=callbacks
    )

    print(vgg_face_age_model.summary())

    print("= ===== Training Summary =====")
    print("= Accuracy: " + history.history['acc'])
    print("= Validation Accuracy: " + history.history['val_acc'])
    print("= Loss: " + history.history['loss'])
    print("= Validation Loss: " + history.history['val_loss'])

    # Save model

    # serialize model to JSON
    model_json = vgg_face_age_model.to_json()
    with open("trained_models/" + MODEL_NAME + "/" + MODEL_NAME + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    vgg_face_age_model.save_weights("trained_models/" + MODEL_NAME + "/" + MODEL_NAME + ".h5")
    print("= Model saved to disk")

    # exit
    sys.exit(0)
