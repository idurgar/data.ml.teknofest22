from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import tensorflow as tf
import os
from tqdm import tqdm
import argparse
import sys

source_path = (os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))+ "/data.ml.teknofest22"+ '/src/')
sys.path.append(source_path)

from predict_all import predict_category, extract_kanun_hükmünde_kararname_info, \
						extract_cumhurbaskanligi_kararnamesi_info, extract_genelge_info, \
						extract_kanun_info,extract_komisyon_raporu_info, \
						extract_resmi_gazete_info,extract_teblig_info, \
						extract_tuzuk_info,extract_yonetmelik_info, extract_ozelge_info
      
main_dir =  os.getcwd()
data_dir = os.path.join(main_dir, "data/")
models_dir = os.path.join(main_dir, "model/")
outputs_dir = models_dir = os.path.join(main_dir, "outputs/")      

def predict(model_path, data_path, word_index_path, label_dict_path):

	# load data, model, word_index and label_dict
	df = pd.read_csv(data_path)

	model = load_model(model_path)
	word_index = pickle.load(open(word_index_path, "rb"))
	label_dict = pickle.load(open(label_dict_path, "rb"))
 
	# dataframe for results
	results_df = pd.DataFrame(columns=["data_text", "id", "rega_no", "mukerrer_no", "rega_tarihi", "mevzuat_no", "belge_sayi", "mevzuat_tarihi", "donem", "sira_no", "madde_sayisi"])
	results_df.loc[0, "mukerrer_no"] = 0
	results_df.loc[0, "madde_sayisi"] = 0

	for i, text in enumerate(tqdm(df.data_text.values)):
		results_df.loc[i, "data_text"] = text
		kategori_prediction = predict_category(model, word_index, label_dict, 64, text)
		results_df.loc[i, "id"] = kategori_prediction
		
		if kategori_prediction == "Kanun Hükmünde Kararname":
			rega_no, mukerrer_no, rega_tarihi, mevzuat_no, mevzuat_tarihi, madde_sayisi = extract_kanun_hükmünde_kararname_info(text)
			results_df.loc[i, "rega_no"] = rega_no
			results_df.loc[i, "mukerrer_no"] = mukerrer_no
			results_df.loc[i, "rega_tarihi"] = rega_tarihi
			results_df.loc[i, "mevzuat_tarihi"] = mevzuat_tarihi
			results_df.loc[i, "mevzuat_no"] = mevzuat_no
			results_df.loc[i, "madde_sayisi"] = madde_sayisi
		
		if kategori_prediction == "Cumhurbaşkanlığı Kararnamesi":
			rega_no, mukerrer_no, rega_tarihi, mevzuat_no, mevzuat_tarihi, madde_sayisi = extract_cumhurbaskanligi_kararnamesi_info(text)
			results_df.loc[i, "rega_no"] = rega_no
			results_df.loc[i, "mukerrer_no"] = mukerrer_no
			results_df.loc[i, "rega_tarihi"] = rega_tarihi
			results_df.loc[i, "mevzuat_tarihi"] = mevzuat_tarihi
			results_df.loc[i, "mevzuat_no"] = mevzuat_no
			results_df.loc[i, "madde_sayisi"] = madde_sayisi
		
		if kategori_prediction == "Genelge":
			mevzuat_tarihi, mevzuat_no, belge_sayi = extract_genelge_info(text)
			results_df.loc[i, "mevzuat_tarihi"] = mevzuat_tarihi
			results_df.loc[i, "belge_sayi"] = belge_sayi
			results_df.loc[i, "mevzuat_no"] = mevzuat_no
			
		if kategori_prediction == "Kanun":
			rega_no, mukerrer_no, rega_tarihi, mevzuat_tarihi, mevzuat_no, madde_sayisi = extract_kanun_info(text)
			results_df.loc[i, "rega_no"] = rega_no
			results_df.loc[i, "mukerrer_no"] = mukerrer_no
			results_df.loc[i, "rega_tarihi"] = rega_tarihi
			results_df.loc[i, "mevzuat_tarihi"] = mevzuat_tarihi
			results_df.loc[i, "mevzuat_no"] = mevzuat_no
			results_df.loc[i, "madde_sayisi"] = madde_sayisi
			
		if kategori_prediction == "Komisyon Raporu":
			sira_no, donem = extract_komisyon_raporu_info(text)
			results_df.loc[i, "sira_no"] = sira_no
			results_df.loc[i, "donem"] = donem
			
		if kategori_prediction == "Resmi Gazete":
			rega_no, mukerrer_no, rega_tarihi = extract_resmi_gazete_info(text)
			results_df.loc[i, "rega_no"] = rega_no
			results_df.loc[i, "mukerrer_no"] = mukerrer_no
			results_df.loc[i, "rega_tarihi"] = rega_tarihi
			
		if kategori_prediction == "Tebliğ":
			rega_no, mukerrer_no, rega_tarihi = extract_teblig_info(text)
			results_df.loc[i, "rega_no"] = rega_no
			results_df.loc[i, "mukerrer_no"] = mukerrer_no
			results_df.loc[i, "rega_tarihi"] = rega_tarihi
			
		if kategori_prediction == "Tüzük":
			rega_no, mukerrer_no, rega_tarihi, mevzuat_no, mevzuat_tarihi, madde_sayisi = extract_tuzuk_info(text)
			results_df.loc[i, "rega_no"] = rega_no
			results_df.loc[i, "mukerrer_no"] = mukerrer_no
			results_df.loc[i, "rega_tarihi"] = rega_tarihi
			results_df.loc[i, "mevzuat_no"] = mevzuat_no
			results_df.loc[i, "rega_tarihi"] = rega_tarihi
			results_df.loc[i, "mevzuat_tarihi"] = mevzuat_tarihi
			results_df.loc[i, "madde_sayisi"] = madde_sayisi
			
		if kategori_prediction == "Yönetmelik":
			rega_no, mukerrer_no, rega_tarihi, madde_sayisi = extract_yonetmelik_info(text)
			results_df.loc[i, "rega_no"] = rega_no
			results_df.loc[i, "mukerrer_no"] = mukerrer_no                        
			results_df.loc[i, "rega_tarihi"] = rega_tarihi    
			results_df.loc[i, "madde_sayisi"] = madde_sayisi    
			
		if kategori_prediction == "Özelge":
			mevzuat_tarihi = extract_ozelge_info(text)
			results_df.loc[i, "mevzuat_tarihi"] = mevzuat_tarihi

	results_df.to_csv(outputs_dir + "ornek-eval-dataset.csv", index=None)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model_path", required=True)
arg_parser.add_argument("--data_path", required=True)
arg_parser.add_argument("--word_index_path", required=True)
arg_parser.add_argument("--label_dict_path", required=True)

if __name__ == '__main__':
    args = arg_parser.parse_args()
    predict(**vars(args))