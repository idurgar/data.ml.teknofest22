import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
import os

main_dir = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
data_path = os.path.join(main_dir, "data/")

# get stopwords from nltk
stop_words = set(stopwords.words('turkish'))
# get stopwords from text file and combine this stopword with nltk stopwords
def get_stopwords(filename):    
    with open(filename, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return list(frozenset(stop_set))

stopwords_path = data_path + "stopwords.txt"
tr_stopwords = get_stopwords(stopwords_path)
del tr_stopwords[0] # first element of list is space

all_stopwords = set(stop_words.union(set(tr_stopwords)))

def preprocessing(text):
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    lower_map = {
    ord(u'I'): u'ı',
    ord(u'İ'): u'i',
    ord(u'Ş'): u'ş',
    ord(u'Ü'): u'ü',
    ord(u'Ö'): u'ö',
    ord(u'Ç'): u'ç',
    ord(u'Ğ'): u'ğ'
    }
    text = text.translate(lower_map).lower()
    # remove “,” from some texts
    text = text.replace("”", "").replace("“", "").replace("(", "").replace(")", "")
    # remove various url types
    text = re.sub(r"http\S+", "", text)
    text = re.sub('http[s]?://\S+', '', text)
    text = re.sub('http://\S+|https://\S+', '', text)
    #text = re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)","", text)
    # remove digits
    text = re.sub(r'[0-9]', '', text)
    # remove .inci, .ıncı, .uncu, .üncü  
    text = text.replace("inci", "").replace("ıncı", "").replace("üncü", "").replace("uncu", "")
    text = text.replace("ncu", "").replace("ncü", "").replace("ncı", "").replace("nci", "")
    #remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    text = text.replace("ý", "i")
    text = text.replace("ð", "ğ")
    text = text.replace("Þ", "ş")
    text = text.replace("â", "a")
    text = text.replace("û", "u")
    text = text.replace("α", "a")
    text = text.replace("ģ", "ş")
    text = text.replace("ð", "ğ")
    text = text.replace("õ", "ü")
    text = text.replace("î", "i")
    text = text.replace("í", "i")
    text = text.replace("ì", "i")
    
    invalid_chars = [' ', 'q', 'w', 'x', 'â', 'ä', 'é', 'í', 'î', 'ð', 'ô', 'õ', 'û', 'ý', 'ģ', '̇', 'м', 'р', 'т']
    text = " ".join([ele for ele in text.split() if all(ch not in ele for ch in invalid_chars)])
    #remove single letters - 2 methods
    #text = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', text)
    text = ' '.join( [w for w in text.split() if len(w)>1] )

    text = text.replace("  ", " ")
    text = text.replace(".", "").strip()
    # remove stopwords
    text = " ".join([word for word in text.split() if word not in all_stopwords])
    # remove all multi spaces after all process
    text = re.sub('\s+', ' ', text)
    text = re.sub(' +', ' ', text)

    return text
if __name__ == "__main__":
    preprocessing()