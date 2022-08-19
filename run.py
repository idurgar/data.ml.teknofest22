import numpy as np
import pandas as pd
from sklearn import metrics
from gensim.models import KeyedVectors
import argparse
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
import sys, os
import warnings
warnings.filterwarnings("ignore")

source_path = (os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))+ "/data.ml.teknofest22"+ '/src/')
sys.path.append(source_path)

from modelling import BiLSTM_Model
from dataset import DeepDataset

main_dir = os.getcwd()
outputs_dir = models_dir = os.path.join(main_dir, "outputs/")
models_dir = os.path.join(main_dir, "model/")

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
		plt.savefig(outputs_dir + f"{title} + .jpg")

def run(data_path, w2v_path, chunk_size, max_len, epochs=60, batch_size=256):
	
	df = pd.read_csv(data_path)
	word_vectors = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

	df = df[~df.kategori.isin(["Cumhurbaşkanlığı Kararnamesi", "Kanun Hükmünde Kararname"])]
	labels = df.kategori.values
	le = LabelEncoder()
	le.fit_transform(df.kategori)
	label_dict = dict(zip(le.classes_, le.transform(le.classes_)))

	# create callbacks using ModelCheckpoint and EarlyStopping (if-else is for alternative)
	checkpath = outputs_dir + "checkpoints/model_checkpoint.hdf5"
	checkpoint = ModelCheckpoint(checkpath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
	earlystopping = EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=8, mode="max")
	callbacks = [checkpoint, earlystopping]
	
	# prepare dataset objects 
 	# DeepDataset module includes word_index, embedding_matrix, train, validation and test datasets for use in training

	dataset = DeepDataset(df, text_column="text", label_column="kategori", chunk_size=chunk_size, word_vectors=word_vectors, max_len=max_len)
	dataset.prepare_data()
	
	# if you want, you can save dataset
	pickle.dump(dataset, open(outputs_dir + "deep_dataset.pkl", "wb"))
	
	# create model
	model = BiLSTM_Model(epochs=epochs, batch_size=batch_size, callbacks=callbacks)
	# training
	model.train(dataset, class_weights=True)
	
	# predictions
	model.predict(dataset.test_data)
	# show classification report and plot confusion matrix
	model.show_performance_result(dataset.test_labels, list(label_dict.keys()))
 
	model.model.save(models_dir + "model.hdf5")
	cr = classification_report(dataset.test_labels, model.predictions, target_names=list(label_dict.keys()), output_dict=True)
	pd.DataFrame(cr).T.to_csv(outputs_dir + f"classification_report.csv")
	cm = confusion_matrix(dataset.test_labels, model.predictions)
	plot_confusion_matrix(cm, classes=list(label_dict.keys()), title="Confusion Matrix")


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--data_path", required=True)
arg_parser.add_argument("--w2v_path", required=True)
arg_parser.add_argument("--chunk_size", required=True, default=300, type=int)
arg_parser.add_argument("--max_len", required=True, default=1024, type=int)
arg_parser.add_argument("--epochs", required=True, default=30, type=int)
arg_parser.add_argument("--batch_size", required=True, default=256, type=int)

if __name__ == '__main__':
    args = arg_parser.parse_args()
    run(**vars(args))