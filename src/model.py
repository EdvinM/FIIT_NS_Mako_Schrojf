from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import pandas as pd

class Model():

	def __init__(self, df_path, y_column_name):
		
		self.df = pd.read_csv(df_path, sep=';')

		if self.df == None:
			print("Unable to open dataframe")
			break

		if y_column_name not in self.df.columns:
			print("Column not found in dataframe columns")
			break

		self.x = self.df.loc[:, self.df.columns != y_column_name]
		self.y = self.df.loc[:, self.df.columns == y_column_name]

		self.train_x = []
		self.train_y = []
		self.test_x = []
		self.test_y = []

		self.model = None

	def split_df(test_size = 0.3):
		self.train_x, self.test_x, 
		self.train_y, self.test_y = train_test_split(self.x, self.y, test_size=test_size)

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

	def train_data():
		return (self.train_x, self.train_y)

	def test_data():
		return (self.test_x, self.test_y)

	def load_model(name="model"):
		json_file = open("trained_models/" + name + '.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(loaded_model_json)

		# load weights into new model
		self.model.load_weights("trained_models/" + name + ".h5")
		print("Loaded model from disk")
