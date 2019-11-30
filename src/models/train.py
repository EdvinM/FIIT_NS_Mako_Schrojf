import os
import sys
sys.path.insert(0, os.getcwd())

from src.data import WIKISequence

# load data as df
# initialize train and test Sequence
# test = WIKISequence(...)
# train model

print("ok")

class Train():
	def __init__(self, model, logs_path = 'logs'):
		self.model = model

		self.callbacks = callbacks = [
		    keras.callbacks.TensorBoard(
		        log_dir=os.path.join(logs_path, timestamp()),
		        histogram_freq=1,
		        profile_batch=0)
		]

	def start(epochs = 1, batch_size = 16):
		train_seq = WIKISequence(
			self.model.train_data()[0], 
			self.model.train_data()[1], batch_size=batch_size)
		test_seq = WIKISequence(
			self.model.test_data()[0], 
			self.model.test_data()[1], batch_size=batch_size)

		print("Train sequence data length= " + str(len(train_seq)))
		print("Test sequence data length= " + str(len(test_seq)))

		history = self.model.fit_generator(
	   		train_seq,
	     	epochs=epochs,
		    validation_data=test_seq,
		    callbacks = self.callbacks
		)

		print("===== Training Summary =====")
		print("Accuracy: " + history.history['acc'])
		print("Validation Accuracy: " + history.history['val_acc'])
		print("Loss: " + history.history['loss'])
		print("Validation Loss: " + history.history['val_loss'])
        
	def summary():
		print(self.model.summary())

	def save_model(name="model")
		# serialize model to JSON
		model_json = self.model.to_json()
		with open("trained_models/" + name + ".json", "w") as json_file:
		    json_file.write(model_json)

		# serialize weights to HDF5
		self.model.save_weights("trained_models/" + name + ".h5")
		print("Saved model to disk")
