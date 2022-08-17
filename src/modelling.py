import nltk
import numpy as np
import itertools
from sklearn.metrics import classification_report
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
class BiLSTM_Model:
    
	def __init__(self, epochs=30, batch_size=256, callbacks=None):
		self.epochs = epochs
		self.batch_size = batch_size
		self.callbacks = callbacks

	def train(self, dataset, class_weights=False):
		# compute class weights 
		from sklearn.utils import class_weight
		if class_weights:
			self.class_weight_dict = class_weight.compute_class_weight(class_weight='balanced', classes= np.unique(dataset.train_labels), y=dataset.train_labels)
			self.class_weight_dict = dict(enumerate(self.class_weight_dict))
		else:
			self.class_weight_dict = None
		
		# get data from datadict (datadict: all data is exported to dictionary)
		embedding_matrix = dataset.embedding_matrix
		train_data = dataset.train_data
		validation_data = dataset.val_data
		train_labels_categorical = dataset.train_labels_categorical
		validation_labels_categorical = dataset.val_labels_categorical
		dim = embedding_matrix.shape[1]
		max_len = len(train_data[0])
		no_classes = len(np.unique(dataset.test_labels))
		# create model
		inputs = Input(shape=(max_len, ))
		embed = Embedding(embedding_matrix.shape[0], dim, weights=[embedding_matrix], input_length=max_len, trainable=False, mask_zero=False)(inputs)
		x = Bidirectional(LSTM(dim,  return_sequences=True, unit_forget_bias=True))(embed)
		attention = Dense(1, activation='tanh')(x)
		attention = Flatten()(attention)
		attention = Activation('softmax')(attention)
		attention = RepeatVector(2 * dim)(attention) #
		attention = Permute([2, 1])(attention)
		sent_representation = Multiply()([x, attention])
		sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2))(sent_representation)
		probabilities = Dense(no_classes, activation='softmax')(sent_representation)

		self.model = Model(inputs=[inputs], outputs=[probabilities])
		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		self.model.fit(train_data, 
					   train_labels_categorical, 
					   validation_data = (validation_data, validation_labels_categorical), 
					   batch_size = self.batch_size, 
					   epochs = self.epochs,
					   callbacks = self.callbacks,
        			   class_weight = self.class_weight_dict)		
	
	def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
		plt.figure(figsize=(14,8))
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=90)
		plt.yticks(tick_marks, classes)

		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			print("Normalized confusion matrix")
		else:
			print('Confusion matrix, without normalization')

		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, cm[i, j],
						horizontalalignment="center",
						color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
 
	def predict(self, test_data):
		import time
		print('Testing started...')
		start_time = time.time()

		self.class_probabilities = self.model.predict(test_data)
		self.predictions = np.argmax(self.class_probabilities, axis=1)

		print('Finished!')
		print(f'Test time: {time.time()-start_time}')
  
	def predict_custom(self, dataset, text_list):

		tokenized = []
		for text in text_list:
			padded_custom = [0 for i in range(dataset.max_len)]
			_ = [padded_custom.__setitem__(i, dataset.embeddings_index[word]) for i, word in enumerate(text.split()[:dataset.max_len]) if word in dataset.embeddings_index]
			tokenized.append(padded_custom)   
   
		class_probabilities = self.model.predict(tokenized)
		prediction = np.argmax(class_probabilities, axis=1)
		return prediction, class_probabilities
 
	def show_performance_result(self, test_labels, classes):
		print("Showing results for each class...")
		print(classification_report(test_labels, self.predictions))
  
		print("Plotting Confusion Matrix...")
		cm = metrics.confusion_matrix(test_labels, self.predictions)
		self.plot_confusion_matrix(cm, classes=classes)

	
