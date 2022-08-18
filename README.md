# Kamuda Mevzuat Arama Motoru Geliştirme

Belirlenen bu görevde; sınıflandırma, künye çıkarımı ve metin/bölüm analizi ile metni verilen bir içeriğin kategorisinin, buna bağlı aşağıda detayları verilmiş üst verileri ve ilgili kategoriler için "madde sayısı"nın tespiti hedeflenmektedir.

## Requirements
Python 3 (Versions 3.7, 3.8 and 3.9)

tensorflow-gpu 2.6

gensim 4.2 

## Training

For Turkish Word2vec pretrained model, you need to go to <a href="https://drive.google.com/drive/folders/1IBMTAGtZ4DakSCyAoA4j7Ch0Ft1aFoww">this address</a> and download it

If you want to train using notebooks, use <a href="https://github.com/idurgar/data.ml.teknofest22/blob/master/notebooks/classification.ipynb">this link</a>

```python

import pandas as pd
from gensim.models import KeyedVectors
from text_preprocessing import preprocessing

df = pd.read_csv(data_path)
word_vectors = KeyedVectors.load_word2vec_format(word_vectors_path, binary=True)

df["data_text"] = df["data_text"].apply(preprocessing)

```

Otherwise use run.py

```bash
  CUDA_VISIBLE_DEVICES=0    # gpu/device id: for specific GPUs
  python run.py
--data_path ~/csv_data_path
--w2v_path ~/word2vec_path/
--chunk_size 100     
--max_len 512   
--epochs 60   
--batch_size 256
```

