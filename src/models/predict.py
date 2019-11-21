import cv2
import numpy as np

class Predict():
	def __init__(self, model):
		self.model = model

	def load_img(self, file_path):
        """Load single image from disk and resize and convert to np array
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