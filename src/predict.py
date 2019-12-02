import os

import cv2
import numpy as np

import data.load_data as load_data
from model import load_trained_model
from model import restore_model_from_checkpoint

def load_img(file_path):
    """Load single image from disk and resize and convert to np array
    """
    im = cv2.imread(file_path)
    im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return (np.array(im) / 255.0).astype(np.float32)


classes = np.array([i for i in range(0, 101)])
# classes.writable = False


def softmax_to_age(predictions):
    return np.sum(predictions * classes, axis=1)


if __name__ == "__main__":
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_PATH)

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    sys.path.insert(0, os.getcwd())

    DATA_NAME = 'wiki'  # or imdb

    # load model
    vgg_face_age_model = load_trained_model()

    # load data
    data_df = None
    base_path = '.'
    if DATA_NAME == 'wiki':
        data_df = load_data.load_wiki_df_from_csv('../data/processed/wiki_df.csv')
        base_path = '../data/raw/wiki_crop/'
        print("= " + str(len(data_df)) + " ROWS OF WIKI DATA LOADED\n")
    if DATA_NAME == 'imdb':
        data_df = load_data.load_imdb_df_from_pkl('../data/processed/imdb_meta_df.pkl')
        data_df['full_path'] = data_df['full_path'].map(lambda x: x[0])
        base_path = '../data/raw/imdb_crop/'
        print("= " + str(len(data_df)) + " ROWS OF IMDB DATA LOADED\n")
    if data_df is None:
        raise Exception("No data was loaded.")

    # loop data until crt+c or eof
    cur = 0
    step = 25
    length = len(data_df)

    while cur < length:
        for i in range(step):
            row = cur + i

            file_path = base_path + data_df['full_path'][row]
            age = data_df['age'][row]

            image = np.array([load_img(file_path)])
            predictions = vgg_face_age_model.predict([image])

            apparent_predictions = softmax_to_age(predictions)
            prediction = apparent_predictions[0]

            print(row, ":\tActual age:", age, "\tPredicted age:", prediction, "\tDiff:", age - prediction, "\targmax:",
                  np.argmax(predictions[0]))

        cur = cur + step

        print("PRESS <ENTER> KEYBOARD TO CONTINUE")
        input()
