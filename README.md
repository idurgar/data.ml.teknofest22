# Kamuda Mevzuat Arama Motoru Geliştirme

Belirlenen bu görevde; sınıflandırma, künye çıkarımı ve metin/bölüm analizi ile metni verilen bir içeriğin kategorisinin, buna bağlı detayları verilmiş üst verileri ve ilgili kategoriler için "madde sayısı"nın tespiti hedeflenmektedir.

<b>Note: Except for the main data set of the competition, no data set was used.</b>

## Quickstart 

### Training

For Turkish Word2vec pretrained model, you need to go to <a href="https://drive.google.com/drive/folders/1IBMTAGtZ4DakSCyAoA4j7Ch0Ft1aFoww">this address</a> and download it

If you want to train using notebooks, use <a href="https://github.com/idurgar/data.ml.teknofest22/blob/master/notebooks/classification.ipynb">this link</a>

```python
# load data, word2vec then preprocess text
import pandas as pd
from gensim.models import KeyedVectors
from text_preprocessing import preprocessing

df = pd.read_csv(data_path)
word_vectors = KeyedVectors.load_word2vec_format(word_vectors_path, binary=True)

df["data_text"] = df["data_text"].apply(preprocessing)

```

```python
# prepare dataset objects using DeepDataset
from dataset import DeepDataset

CHUNK_SIZE = 300
MAX_LEN = 1024

dataset = DeepDataset(df, text_column="data_text", label_column="kategori", chunk_size=CHUNK_SIZE, word_vectors=word_vectors, max_len=MAX_LEN)
dataset.prepare_data()

```

```python
# training
from modelling import BiLSTM_Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

CLASS_WEIGHTS = True
EPOCHS = 80
BATCH_SIZE = 256

# create callbacks
checkpoint = ModelCheckpoint(models_dir + "model_checkpoint.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
earlystopping = EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=9, mode="max")
callbacks = [checkpoint, earlystopping]

# build model and start training
model = BiLSTM_Model(epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)
model.train(dataset, class_weights=CLASS_WEIGHTS)
```

Otherwise use run.py

```bash
  CUDA_VISIBLE_DEVICES=0    # gpu/device id: for specific GPUs
  python run.py
--data_path ~/csv_data_path
--w2v_path ~/word2vec_path
--chunk_size 300     
--max_len 1024   
--epochs 80   
--batch_size 256
```

### Testing

If you want to test using notebooks, use <a href="https://github.com/idurgar/data.ml.teknofest22/blob/master/notebooks/prediction.ipynb">this link</a>

```python
# testing 
kategori_pred
from predict_all import *

# predict kategori using model or regex; details are in src/predict_all.py
kategori_prediction = predict_category(model, word_index, label_dict, max_len, text)

#if kategori is "Kanun Hükmünde Kararname"
rega_no, mukerrer_no, rega_tarihi, mevzuat_no, mevzuat_tarihi, madde_sayisi = extract_kanun_hükmünde_kararname_info(text)
#if kategori is "cumhurbaskanlığı kararnamesi"
rega_no, mukerrer_no, rega_tarihi, mevzuat_no, mevzuat_tarihi, madde_sayisi = extract_cumhurbaskanligi_kararnamesi_info(text)
#if kategori is "Genelge"
mevzuat_tarihi, mevzuat_no, belge_sayi = extract_genelge_info(text)
#if kategori is "Kanun"
rega_no, mukerrer_no, rega_tarihi, mevzuat_tarihi, mevzuat_no, madde_sayisi = extract_kanun_info(text)
#if kategori is "Komisyon Raporu"
sira_no, donem = extract_komisyon_raporu_info(text)
#if kategori is "Resmi Gazete"
rega_no, mukerrer_no, rega_tarihi = extract_resmi_gazete_info(text)
#if kategori is "Tebliğ"
rega_no, mukerrer_no, rega_tarihi = extract_teblig_info(text)
#if kategori is "Tüzük"
rega_no, mukerrer_no, rega_tarihi, mevzuat_no, mevzuat_tarihi, madde_sayisi = extract_tuzuk_info(text)
#if kategori is "Yönetmelik"
rega_no, mukerrer_no, rega_tarihi, madde_sayisi = extract_yonetmelik_info(text)
#if kategori is "Özelge"
mevzuat_tarihi = extract_ozelge_info(text)

```