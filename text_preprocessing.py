import pandas as pd
import re
import numpy as np
import nltk
import numpy as np
from nltk.corpus import stopwords

# get stopwords from nltk
stop_words = set(stopwords.words('turkish'))
# get stopwords from text file and combine this stopword with nltk stopwords
def get_stopwords(filename):    
    with open(filename, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return list(frozenset(stop_set))

stopwords_path = "stopwords.txt"
tr_stopwords = get_stopwords(stopwords_path)
del tr_stopwords[0] # first element of list is space

all_stopwords = set(stop_words.union(set(tr_stopwords)))

# preprocessing
def preprocessing(df):
	df["cleaned_text"] = df.data_text.replace("\r", " ").replace("\n", " ")
	df["cleaned_text"] = df["cleaned_text"].str.strip()
	df["cleaned_text"] = df["cleaned_text"].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))
	df["cleaned_text"] = df["cleaned_text"].str.replace(":", "")
	df["cleaned_text"] = df["cleaned_text"].str.replace(",", "")
	# remove “,” from some texts
	df["cleaned_text"] = df["cleaned_text"].str.replace("”", "").str.replace("“", "").str.replace("(", "").str.replace(")", "")
	df["cleaned_text"] = df["cleaned_text"].str.lower()
	df["cleaned_text"] = df["cleaned_text"].str.strip()
	# remove stopwords using combined stopwords set
	df.cleaned_text = df.cleaned_text.apply(lambda x: " ".join([word for word in x.split() if word not in all_stopwords]))
	# remove various url types
	df.cleaned_text = df.cleaned_text.apply(lambda x: re.sub(r"http\S+", "", x))
	df.cleaned_text = df.cleaned_text.apply(lambda x: re.sub('http[s]?://\S+', '', x))
	df.cleaned_text = df.cleaned_text.apply(lambda x: re.sub('http://\S+|https://\S+', '', x))
	df.cleaned_text = df.cleaned_text.apply(lambda x: re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)","", x))
	# remove digits
	df.cleaned_text = df.cleaned_text.apply(lambda x: re.sub(r'[0-9]', '', x)) 
	# remove .inci, .ıncı, .uncu, .üncü  
	df.cleaned_text = df.cleaned_text.str.replace(" inci ", "").str.replace(" ıncı ", "").str.replace(" üncü ", "").str.replace(" uncu ", "")
	#remove punctuation
	df.cleaned_text = df.cleaned_text.apply(lambda x: re.sub(r'[^\w\s]', '', x)) 
	#remove single letters - 2 methods
	df.cleaned_text = df.cleaned_text.apply(lambda x: re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', x))
	df.cleaned_text = df.cleaned_text.apply(lambda x: ' '.join( [w for w in x.split() if len(w)>1] ))
 
	df.cleaned_text = df.cleaned_text.replace("  ", " ")
	df.cleaned_text = df.cleaned_text.str.replace(".", "").str.strip()
	# remove all multi spaces after all process
	df.cleaned_text = df.cleaned_text.str.replace(".", "").str.strip()
	df.cleaned_text = df.cleaned_text.apply(lambda x: re.sub(' +', ' ', x))
	

if __name__ == "__main__":
    preprocessing()