import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import yaml
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split


wiki_df = pd.read_csv('../data/processed/wiki_df.csv', sep=';')
wiki_df['img_array'] = wiki_df['full_path'].apply(lambda x: cv2.resize(cv2.imread('../data/raw/wiki_crop/' + x), (224, 224), interpolation=cv2.INTER_LINEAR).reshape(1, -1)[0])


model = keras.models.Sequential()
model.add(keras.layers.ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(keras.layers.Convolution2D(64, (3, 3), activation='relu'))
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))
 
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(128, (3, 3), activation='relu'))
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(128, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))
 
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(256, (3, 3), activation='relu'))
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(256, (3, 3), activation='relu'))
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(256, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))
 
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))
 
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))
 
model.add(keras.layers.Convolution2D(4096, (7, 7), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Convolution2D(4096, (1, 1), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Convolution2D(2622, (1, 1)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Activation('softmax'))

model.load_weights('../data/vgg_face_weights.h5')

classes = 101 #0 to 100
target = wiki_df['age'].values
target_classes = keras.utils.to_categorical(target, classes)

features = []
 
for i in range(0, wiki_df.shape[0]):
    features.append(wiki_df['img_array'].values[i])
 
features = np.array(features)
features = features.reshape(features.shape[0], 224, 224, 3)

train_x, test_x, train_y, test_y = train_test_split(features, target_classes, test_size=0.30)

for layer in model.layers[:-7]:
    layer.trainable = False


base_model_output = keras.models.Sequential()
base_model_output = keras.layers.Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
base_model_output = keras.layers.Flatten()(base_model_output)
base_model_output = keras.layers.Activation('softmax')(base_model_output)


age_model = keras.Model(inputs=model.input, outputs=base_model_output)

age_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
 
scores = []
epochs = 250
batch_size = 50

for i in range(epochs):
	print("epoch ",i)
 
	ix_train = np.random.choice(train_x.shape[0], size=batch_size)
	 
	score = age_model.fit(train_x[ix_train], train_y[ix_train], epochs=1, validation_data=(test_x, test_y))
	 
	scores.append(score)