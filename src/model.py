import os

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import model_from_json

import models.VGGFaceModel as VGGFaceModel

# load model from file
# load model from and then load weights
# create empty model

MODEL_NAME = 'VGG_FACE_AGE_PREDICT'

model_path = "../models/" + MODEL_NAME + "/model.json"
weight_path = "../models/" + MODEL_NAME + "/weights.h5"
checkpoint_path = "../models/" + MODEL_NAME + "/checkpoints/age_model.hdf5"


def check_if_model_exist():
    if not os.path.isfile(model_path) or not os.path.isfile(weight_path):
        raise Exception("model.json or weights.h5 not available in ", MODEL_NAME)


def load_trained_model():
    check_if_model_exist()

    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(weight_path)
    print("= TRAINED MODEL LOADED FROM DISK")

    return model


def create_new_model():
    model = VGGFaceModel.build_new_age_model()

    return model


def create_model_and_load_weights():
    model = create_new_model()
    model.load_weights(weight_path)
    print("= TRAINED MODEL LOADED FROM DISK")

    return model


def restore_model_from_checkpoint():
    model = create_new_model()
    model.load_weights(checkpoint_path)
    print("= TRAINED MODEL LOADED FROM DISK")

    return model


if __name__ == "__main__":
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_PATH)

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    sys.path.insert(0, os.getcwd())

    check_if_model_exist()

    model = load_trained_model()
    model.summary()

    print("= MODEL FILE AVAILABLE, USE load_trained_model() function to load your model")
