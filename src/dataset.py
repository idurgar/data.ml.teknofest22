import re
import numpy as np
import random
import math
import pandas as pd
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
class DeepDataset:
	def __init__(self, dataframe, text_column, label_column, chunk_size, word_vectors, max_len):
		self.dataframe = dataframe
		self.chunk_size = chunk_size
		self.word_vectors = word_vectors
		self.text_column = text_column
		self.label_column = label_column
		self.max_len = max_len

	def prepare_data(self):
		
		word_vectors_to_dict = {}
		self.embeddings_index = {}
		word_index = 1
          
		self.dataframe = self.dataframe.reset_index(drop=True)
		# tokenize texts
		text_tokenized = [[word for word in text.split()] for text in self.dataframe[self.text_column].values]
		for seq in text_tokenized:
			for word in seq:
				try:
					word_vectors_to_dict[word] = (self.word_vectors.get_vector(word))
					if not word in self.embeddings_index:
						self.embeddings_index[word] = word_index
						word_index += 1
				except:
					pass
    
		labels = self.dataframe[self.label_column].values
		le = LabelEncoder()
		le.fit_transform(self.dataframe[self.label_column])
		self.label_dict = dict(zip(le.classes_, le.transform(le.classes_)))
		#self.label_dict = {'Cumhurbaşkanlığı Kararnamesi': 0, 'Genelge' : 1,
		#					"Kanun":2, "Kanun Hükmünde Kararname":3, 'Komisyon Raporu':4, 'Resmi Gazete':5,
		#					'Tebliğ':6, 'Tüzük':7, 'Yönetmelik':8, 'Özelge':9}

		no_classes = len(self.label_dict)
		vocab_size = len(self.embeddings_index)

		labels = self.dataframe[self.label_column]
		le = LabelEncoder()
		le.fit_transform(self.dataframe[self.label_column])
		label_dict = dict(zip(le.classes_, le.transform(le.classes_)))

		# Split train, validation, test
		train_size = 0.7
		val_size = 0.15
		test_size = 0.15
		
		X_train, X_, y_train, y_ = train_test_split(self.dataframe[self.text_column], 
                                                    self.dataframe[self.label_column], 
                                                    train_size=train_size, 
                                                    random_state=42,
                                                    stratify=self.dataframe[self.label_column])
		X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, random_state=42, stratify=y_)

		train = pd.DataFrame({"text": X_train, "kategori": y_train})
		validation = pd.DataFrame({"text": X_val, "kategori": y_val})
		test = pd.DataFrame({"text": X_test, "kategori": y_test})
		train_indices = train.index.tolist()
		val_indices = validation.index.tolist()
		test_indices = test.index.tolist()

		train_list = [text_tokenized[ix] for ix in train_indices]
		val_list = [text_tokenized[ix] for ix in val_indices]
		test_list = [text_tokenized[ix] for ix in test_indices]

		self.train_labels = [] # will be able to create with chunk_size
		self.val_labels = [self.label_dict[labels[ix]] for ix in val_indices]
		self.test_labels = [self.label_dict[labels[ix]] for ix in test_indices]

		no_train = len(train_list)
		no_val = len(val_list)
		no_test = len(test_list)

		# dim: embedding dimension
		wv_ix_to_key = self.word_vectors.index_to_key
		dim = len(self.word_vectors.get_vector(wv_ix_to_key[0]))

		self.train_data = []
		self.val_data = []
		self.test_data = []

		for i, text in enumerate(train_list):
			padded = [0 for i in range(self.max_len)]
			_ = [padded.__setitem__(i, self.embeddings_index[word]) for i, word in enumerate(text[:self.max_len]) if word in self.embeddings_index]
			self.train_data.append(padded)
			
			max_ix = max([ix+1 for ix, word in enumerate(text[:self.max_len]) if word in self.embeddings_index])
			self.train_labels.append(self.label_dict[labels[train_indices[i]]])
			#chunking data for augmentation
			chunk_size = self.chunk_size

			for j in range(math.floor(max_ix / chunk_size)):
				padded_temp = [0 for i in range(self.max_len)]
				start_ind = j * chunk_size
				end_ind = (j + 1) * chunk_size
				if end_ind < self.max_len - 1:
					_ = [padded_temp.__setitem__(i, self.embeddings_index[word]) for i, word in enumerate(text[start_ind:end_ind]) if word in self.embeddings_index]
					self.train_data.append(padded_temp)
					self.train_labels.append(self.label_dict[labels[train_indices[i]]])

		for i, text in enumerate(val_list):
			padded = [0 for i in range(self.max_len)]
			_ = [padded.__setitem__(i, self.embeddings_index[word]) for i, word in enumerate(text[:self.max_len]) if word in self.embeddings_index]
			self.val_data.append(padded)

		for i, text in enumerate(test_list):
			padded = [0 for i in range(self.max_len)]
			_ = [padded.__setitem__(i, self.embeddings_index[word]) for i, word in enumerate(text[:self.max_len]) if word in self.embeddings_index]
			self.test_data.append(padded)

		self.embedding_matrix = np.zeros((vocab_size + 1, dim))
  
		for key, vec in word_vectors_to_dict.items():
			self.embedding_matrix[self.embeddings_index[key], :] = vec
  
		self.train_data = np.array(self.train_data, dtype='int')
		self.val_data = np.array(self.val_data, dtype='int')
		self.test_data = np.array(self.test_data, dtype='int')

		# One-hot encode labels
		self.train_labels_categorical = utils.to_categorical(self.train_labels)
		self.val_labels_categorical = utils.to_categorical(self.val_labels)
		self.test_labels_categorical = utils.to_categorical(self.test_labels)
