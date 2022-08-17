import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import pickle
from tensorflow.keras.models import load_model
from .modelling import *
from dateutil.parser import parse
import datetime
from .text_preprocessing import preprocessing

def predict(model, word_index, max_len, text):
	tokenized = []
	padded_s = [0 for i in range(max_len)]
	for j, word in enumerate(text.split()):
		if j < max_len:
			if word in word_index:
				padded_s[j] = word_index[word]
	tokenized.append(padded_s)   

	class_probabilities = model.predict(tokenized)
	prediction = np.argmax(class_probabilities, axis=1)
	return prediction

def predict_category(model, word_index, label_dict, max_len, text):
    
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " "))
 
	if (re.search("Karar Sayısı: KHK/(\d+) Kararnamenin", text)) or \
		(re.search("Karar Sayısı : KHK/(\d+) Kararnamenin", text)) or \
		(re.search("Karar Sayısı : KHK/(\d+)", text)) or \
		(re.search("Karar Sayısı: KHK/(\d+)", text)) or \
		("Kanun Hük.Kar.nin Tarihi" in text) or \
		("Kanun Hükmünde Kararnamenin Tarihi" in text) or \
		("Kanun Hükminde Kararnamenin Tarihi" in text) or \
		("Kanun Hük. Kar. nin Tarihi" in text) or \
		("Kanun Hük. Kar.nin Tarihi" in text) or \
		("kanun hük kar nin tarihi" in text) or \
		("karar sayısı khk kararnamenin" in text) or \
		("karar sayısı khk kararnamenin" in text) or \
		("kanun hük kar nin tarihi" in text):
		prediction = "Kanun Hükmünde Kararname" 

	elif (re.search("HAKKINDA CUMHURBAŞKANLIĞI KARARNAMESİ (Kararname Numarası: (\d+))", text)) or \
			(re.search("DAİR CUMHURBAŞKANLIĞI KARARNAMESİ (Kararname Numarası: (\d+))", text)) or \
			(re.search("İLİŞKİN CUMHURBAŞKANLIĞI KARARNAMESİ (Kararname Numarası: (\d+))", text)) or \
			(re.search("İLİŞKİN CUMHURBAŞKANLIĞI KARARNAMESİ (Kararname Numarası : (\d+))", text)) or \
			(re.search("İLİŞKİN CUMHURBAŞKANLIĞI KARARNAMESİ (Kararname Numarası:(\d+))", text)) or \
			(re.search("Cumhurbaşkanlığı Kararnamesinin Sayısı : (\d+)", text)) or \
			(re.search("Cumhurbaşkanlığı Kararnamesinin Sayısı: (\d+)", text)) or \
			(re.search("Cumhurbaşkanlığı Kararnamesinin Sayısı :(\d+)", text)) or \
			(re.search("Cumhurbaşkanlığı Kararnamesinin Sayısı:(\d+)", text)) or \
			(re.search("HAKKINDA CUMHURBAŞKANLIĞI KARARNAMESİ", text)) or \
			(re.search("DAİR CUMHURBAŞKANLIĞI KARARNAMESİ", text)) or \
			(re.search("DAİR CUMHURBAŞKANLIĞI KARARNAMESİ (Kararname Numarası: (\d+))", text)) or \
			(re.search("ilişkin cumhurbaşkanlığı kararnamesi kararname numarası", text)) or \
			(re.search("hakkında cumhurbaşkanlığı kararnamesi kararname numarası", text)) or \
			(re.search("dair cumhurbaşkanlığı kararnamesi kararname numarası", text)) or \
			(re.search("hakkında cumhurbaşkanlığı kararnamesi", text)) or \
			(re.search("cumhurbaşkanlığı kararnamesinin sayısı", text)) or \
			(re.search("dair cumhurbaşkanlığı kararnamesi", text)):
		prediction = "Cumhurbaşkanlığı Kararnamesi" 

	else:
		model_prediction = predict(model, word_index, max_len, preprocessing(text))[0]
		if model_prediction == 0:
			prediction = "Genelge"
			
		elif model_prediction == 1:
			prediction = "Kanun"

		elif model_prediction == 2:
			prediction = "Komisyon Raporu"

		elif model_prediction == 3:
			prediction = "Resmi Gazete"
			
		elif model_prediction == 4:
			prediction = "Tebliğ"
	
		elif model_prediction == 5:
			if (re.search("BKK No:", text)) or \
				(re.search("bkk no", text)) or \
				(re.search("Bakanlar Kurulu Kararının Tarihi : (\d+)/(\d+)/(\d+), No : (\d+)/(\d+)", text)) or \
				(re.search("Bakanlar Kurulu Kararının Tarihi : (\d+)/(\d+)/(\d+) No : (\d+)/(\d+)", text)) or \
				(re.search("Bakanlar Kurulu Kararının Tarihi : (\d+).(\d+).(\d+) No : (\d+)/(\d+)", text)) or \
				(re.search("Bakanlar Kurulu Kararının Tarihi : (\d+).(\d+).(\d+) No: (\d+)/(\d+)", text)) or \
				(re.search("Bakanlar Kurulu Kararının Tarihi : (\d+).(\d+).(\d+), No : (\d+)/(\d+)", text)) or \
				(re.search("Bakanlar Kurulu Kararının Tarihi : (\d+).(\d+).(\d+), No:(\d+)/(\d+)", text)) or \
				(re.search("Dayandığı Kanunun Tarihi : (\d+)/(\d+)/(\d+)", text)) or \
				(re.search("bakanlar kurulu kararının tarihi", text)):
				prediction = "Tüzük"
			else:
				prediction = "Yönetmelik"

		elif model_prediction == 6:
			prediction = "Yönetmelik"
	
		else:
			prediction = "Özelge"

	return prediction

ay_dict = {"Ocak":"01", "Şubat":"02", "Mart":"03", "Nisan":"04", "Mayıs":"05", "Haziran":"06", "Temmuz":"07", 
           "Ağustos":"08", "Eylül":"09", "Ekim":"10", "Kasım":"11", "Aralık":"12",
           "OCAK":"01", "ŞUBAT":"02", 
		   "MART":"03", "NİSAN":"04", 
		   "MAYIS":"05", "HAZİRAN":"06", 
		   "TEMMUZ":"07", "AĞUSTOS":"08", 
		   "EYLÜL":"09","EKİM":"10", 
		   "KASIM":"11","ARALIK":"12"}

def extract_kanun_info(text):
	"""
	rega_no:Kanun'un yayınlanmış olduğu Resmi Gazete Numarası
	mukerrer_no:İçeriğin Resmi Gazete Mükerrer Sayısı (0 ise mükerrer olmadığını)
	rega_tarihi(date:yyyy-mm-dd): İçeriğin yayınlandığı Resmi Gazete Tarihi
	mevzuat_no(string): Kanun'un Numarasını (örn: 657 Sayılı Devlet Memurları Kanunu için Kanun No:657 olur)
	mevzuat_tarihi(date:yyyy-mm-dd): Kanun'un Kabul Tarihi
	Kanun'un bölümlerini (Madde sayısını ve işlenemeyen hükümleri/maddeleri, 
	madde metinlerinin içindeki "Madde <sayı> - ..." ifadeleri hariç )
	"""	
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " "))
	# extract rega no
	no = re.search(r'([Numara]+:\s|[Numara] +:\s|[Sayı]+:\s|[Sayı] +:\s|[Sayısı] +:\s|[Sayısı]+:\s|[Numarası]+\s)([0-9]+)',text)
	if no:
		rega_no = int(no.group(2))
	else:
		rega_no = "Bulunamadı"
  
	# extract mükerrer no
	matching_list = re.findall(r'([0-9]+\s(Mükerrer)\sKarar|[0-9]+\s(Mükerrer)\sKanun)',text)
	count_mukerrer = re.search(r'((\s[0-9]).\sMükerrer\s[A-Za-z]+)',text)
	if matching_list:
		mukerrer_no = 1
	elif count_mukerrer == None :
		mukerrer_no = 0
	elif count_mukerrer:
		mukerrer_no = int(count_mukerrer.group(2))
	else:
		mukerrer_no = 0

	# extract rega tarihi, mevzuat no and mevzuat tarihi
	rega = re.search(r'(Tarihi +:\s|Tarihi+:\s|Tarihi+\s)([0-9]+.[0-9]+.[0-9]{4})',text)
	if rega:
		rega_tarihi = pd.to_datetime(rega.group(2)).strftime("%Y-%m-%d") #parse(rega.group(2)).strftime('%Y-%m-%d') 
	else:
		rega_tarihi = "Bulunamadı"
  
	mev_no = re.search(r'(No+.\s+|Numarası+.\s:\s|Sayısı\s+:\sKHK\/)([0-9]+)',text)
	if mev_no:
		mevzuat_no = mev_no.group(2)
	else:
		mevzuat_no = "Bulunamadı"
 
	mevzuat = re.search(r'(Tarihi+\s:\s([0-9]+.[0-9]+.[0-9]+))',text)
	if mevzuat:
		mevzuat_tarihi = pd.to_datetime(mevzuat.group(2)).strftime("%Y-%m-%d") #parse(mevzuat.group(2)).strftime('%Y-%m-%d')
	else:
		mevzuat_tarihi = "Bulunamadı"		
  
	matching_madde_list = re.findall(r'([^\“\"\']Madde\s[0-9]+\s–)|([^\“\"\']Madde–\s[0-9]+\/[0-9])|([^\“\"\']Madde\s[0-9]+\/[a-z]\s.)|([^\“\"\']Madde\s[0-9]+\/[A-Z]\s.)|([^\“\"\']Ek\sMadde\s[0-9]+\s[^a-z^0-9^A-Z][^a-z])|([^\“\"\']Geçici\sMadde\s[0-9]+\s[^a-z][^0-9])|([^\“\"\']Ek\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\s[A-Z][a-z]{4}\s[^0-9][^a-z])|(Geçici\sEk\sMadde\s[0-9]\s[^0-9])|(Geçici\sMadde\s[0-9]+\([0-9]+\)\s[^0-9][^a-z])|(Geçici\sMadde\s[0-9]+-[0-9][^0-9][^A-Z][^a-z])|([^\“\"\']Madde\s[0-9]+[a-z]+\s[^0-9][^a-z^A-Z])|(Madde\s[0-9]+\([0-9]+\)\s[^0-9^A-Z^a-z])|(Madde+\s–\s[0-9]+\/[0-9])|(Madde\s[0-9]+-[0-9]+\s[^0-9][^A-Za-z])',text)
	
	return rega_no, mukerrer_no, rega_tarihi, mevzuat_tarihi, mevzuat_no, len(matching_madde_list)

def extract_kanun_hükmünde_kararname_info(text):
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " "))
	# extract rega no
	no = re.search(r'([Numara]+:\s|[Numara] +:\s|[Sayı]+:\s|[Sayı] +:\s|[Sayısı] +:\s|[Sayısı]+:\s|[Numarası]+\s)([0-9]+)',text)
	if no:
		rega_no = int(no.group(2))
	else:
		rega_no = "Bulunamadı"

	# extract mükerrer no
	matching_list = re.findall(r'([0-9]+\s(Mükerrer)\sKarar|[0-9]+\s(Mükerrer)\sKanun|[0-9]+\s(Mükerrer)\s)',text)
	count_mukerrer = re.search(r'((\s[0-9]).\sMükerrer\s[A-Za-z]+)',text)
	if matching_list:
		mukerrer_no = 1
	elif count_mukerrer == None :
		mukerrer_no = 0
	elif count_mukerrer:
		mukerrer_no = int(count_mukerrer.group(2))
	else:
		mukerrer_no = 0    

	# extract rega tarihi, mevzuat no, mevzuat tarihi and madde
	rega = re.search(r'(Tarihi+\s|Tarihi +:\s|Tarihi+:\s|Tarihi+\s)([0-9]+.[0-9]+.[0-9]{4})',text)
	if rega:
		rega_tarihi = pd.to_datetime(rega.group(2)).strftime("%Y-%m-%d") #parse(rega.group(2)).strftime('%Y-%m-%d')
	else:
		rega_tarihi = "Bulunamadı"

	mev_no = re.search(r'(No+.\s+|Numarası+.\s:\s|Sayısı\s+:\sKHK\/)([0-9]+)',text)
	if mev_no:
		mevzuat_no = mev_no.group(2)
	else:
		mevzuat_no = "Bulunamadı"

	mevzuat = re.search(r'(Tarihi+\s:\s([0-9]+.[0-9]+.[0-9]+))',text)
	if mevzuat:
		mevzuat_tarihi = pd.to_datetime(mevzuat.group(2)).strftime("%Y-%m-%d") #parse(mevzuat.group(2)).strftime('%Y-%m-%d')
	else:
		mevzuat_tarihi = "Bulunamadı"	

	matching_madde_list = re.findall(r'([^\“\"\']Madde\s[0-9]+\s–)|([^\“\"\']Madde–\s[0-9]+\/[0-9])|([^\“\"\']Madde\s[0-9]+\/[a-z]\s.)|([^\“\"\']Madde\s[0-9]+\/[A-Z]\s.)|([^\“\"\']Ek\sMadde\s[0-9]+\s[^a-z^0-9^A-Z][^a-z])|([^\“\"\']Geçici\sMadde\s[0-9]+\s[^a-z][^0-9])|([^\“\"\']Ek\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\s[A-Z][a-z]{4}\s[^0-9][^a-z])|(Geçici\sEk\sMadde\s[0-9]\s[^0-9])|(Geçici\sMadde\s[0-9]+\([0-9]+\)\s[^0-9][^a-z])|(Geçici\sMadde\s[0-9]+-[0-9][^0-9][^A-Z][^a-z])|([^\“\"\']Madde\s[0-9]+[a-z]+\s[^0-9][^a-z^A-Z])|(Madde\s[0-9]+\([0-9]+\)\s[^0-9^A-Z^a-z])|(Madde+\s–\s[0-9]+\/[0-9])|(Madde\s[0-9]+-[0-9]+\s[^0-9][^A-Za-z])',text)
	
	return rega_no, mukerrer_no, rega_tarihi, mevzuat_no, mevzuat_tarihi, len(matching_madde_list)
    
def extract_cumhurbaskanligi_kararnamesi_info(text):
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " "))
	# extract rega no
	no = re.search(r'([Numara]+:\s|[Numara] +:\s|[Sayı]+:\s|[Sayı] +:\s|[Sayısı] +:\s|[Sayısı]+:\s|[Numarası]+\s)([0-9]+)',text)
	if no:
		rega_no = int(no.group(2))
	else:
		rega_no = "Bulunamadı"

	# extract mükerrer no
	matching_list = re.findall(r'([0-9]+\s(Mükerrer)\sKarar|[0-9]+\s(Mükerrer)\sKanun|[0-9]+\s(Mükerrer)\s[A-Za-z])',text)
	count_mukerrer = re.search(r'((\s[0-9]).\sMükerrer\s[A-Za-z]+)',text)
	if matching_list:
		mukerrer_no = 1
	elif count_mukerrer == None :
		mukerrer_no = 0
	elif count_mukerrer:
		mukerrer_no = int(count_mukerrer.group(2))
	else:
		mukerrer_no = 0

	# extract rega tarihi, mevzuat no, mevzuat tarihi and madde

	rega = re.search(r'(Tarihi+\s|Tarihi +:\s|Tarihi+:\s|Tarihi+\s)([0-9]+.[0-9]+.[0-9]{4})',text)
	if rega:
		rega_tarihi = pd.to_datetime(rega.group(2)).strftime("%Y-%m-%d") #parse(rega.group(2)).strftime('%Y-%m-%d')
	else:
		rega_tarihi = "Bulunamadı"	

	mev_no = re.search(r'(Kararname\sNumarası:|Kararname\sNumarası:\s|Kararname\sNumarası:|\(Kararname\sSayısı:|Kararname\sSayısı:\s|\(Kararname\sNumarası\s|\(KARARNAME\sNUMARASI:\s)([0-9]+)',text)
	if mev_no:
		mevzuat_no = mev_no.group(2)
	else:
		mevzuat_no = "Bulunamadı"

	mevzuat = re.search(r'([.]\s([0-9]{1,2}\s[A-Z][a-z].+[0-9]{4}))',text)
	if mevzuat:
		mevzuat_tarihi = mevzuat.group(2)
		mevzuat_tarihi = mevzuat_tarihi.replace(" ","/").replace(".", "/")
		mevzuat_tarihi = [mevzuat_tarihi.replace(letter, str(ay_dict[letter])) for letter in mevzuat_tarihi.split("/") if letter in ay_dict][0]
		mevzuat_tarihi = pd.to_datetime(mevzuat_tarihi).strftime("%Y-%m-%d") #parse(mevzuat_tarihi).strftime('%Y-%m-%d')
	else:
		mevzuat_tarihi = 'Bulunamadı'

	matching_madde_list = re.findall(r'([^\“\"\']Madde\s[0-9]+\s–)|([^\“\"\']Madde–\s[0-9]+\/[0-9])|([^\“\"\']Madde\s[0-9]+\/[a-z]\s.)|([^\“\"\']Madde\s[0-9]+\/[A-Z]\s.)|([^\“\"\']Ek\sMadde\s[0-9]+\s[^a-z^0-9^A-Z][^a-z])|([^\“\"\']Geçici\sMadde\s[0-9]+\s[^a-z][^0-9]+[A-Za-z0-9]\s[A-Za-z]+[[^\“\"\'])|([^\“\"\']Ek\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\s[A-Z][a-z]{4}\s[^0-9][^a-z])|(Geçici\sEk\sMadde\s[0-9]\s[^0-9])|(Geçici\sMadde\s[0-9]+\([0-9]+\)\s[^0-9][^a-z])|(Geçici\sMadde\s[0-9]+-[0-9][^0-9][^A-Z][^a-z])|([^\“\"\']Madde\s[0-9]+[a-z]+\s[^0-9][^a-z^A-Z])|(Madde\s[0-9]+\([0-9]+\)\s[^0-9^A-Z^a-z])|(Madde+\s–\s[0-9]+\/[0-9])|(Madde\s[0-9]+-[0-9]+\s[^0-9][^A-Za-z])|([^\“\"\']Madde\s[0-9]+\/[ĞÇİ]\s[^0-9][^A-Z^a-z])',text)

	return rega_no, mukerrer_no, rega_tarihi, mevzuat_no, mevzuat_tarihi, len(matching_madde_list)

def extract_tuzuk_info(text):
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " "))
	# extract rega no
	no = re.search(r'([Numara]+:\s|[Numara] +:\s|[Sayı]+:\s|[Sayı] +:\s|[Sayısı] +:\s|[Sayısı]+:\s|[Numarası]+\s)([0-9]+)',text)
	if no:
		rega_no = no.group(2)
		rega_no = int(rega_no)
	else:
		rega_no = "Bulunamadı"

	# extract mükerrer no
	matching_list = re.findall(r'([0-9]+\s(Mükerrer)\sKarar|[0-9]+\s(Mükerrer)\sKanun|[0-9]+\s(Mükerrer)\s[A-Za-z])',text)
	count_mukerrer = re.search(r'((\s[0-9]).\sMükerrer\s[A-Za-z]+)',text)
	if matching_list:
		mukerrer_no = 1
	elif count_mukerrer == None :
		mukerrer_no = 0
	elif count_mukerrer:
		mukerrer_no = count_mukerrer.group(2)
		mukerrer_no = int(mukerrer_no)
	else:
		mukerrer_no = 0

	# extract rega tarihi, mevzuat_no, mevzuat_tarihi and madde
	rega = re.search(r'(Tarihi+\s|Tarihi +:\s|Tarihi+:\s|Tarihi+\s)([0-9]+.[0-9]+.[0-9]{4})',text)
	if rega:
		rega_tarihi = pd.to_datetime(rega.group(2)).strftime("%Y-%m-%d") #parse(rega.group(2)).strftime('%Y-%m-%d')
	else:
		rega_tarihi = "Bulunamadı"

	mev_no = re.search(r'(BKK\sNo:\s([0-9]+\/[0-9]+|[0-9]+)\s)',text)
	if mev_no:
		mevzuat_no = mev_no.group(2)
	else:
		mevzuat_no = "Bulunamadı"

	mevzuat = re.search(r'(Tarihi:\s([0-9]+.[0-9]+.[0-9]{4}),)',text)
	if mevzuat:
		mevzuat_tarihi = pd.to_datetime(mevzuat.group(2)).strftime("%Y-%m-%d") #parse(mevzuat.group(2)).strftime('%Y-%m-%d')
	else:
		mevzuat_tarihi = "Bulunamadı"

	matching_madde_list = re.findall(r'([^\“\"\']Madde\s[0-9]+\s–)|([^\“\"\']Madde–\s[0-9]+\/[0-9])|([^\“\"\']Madde\s[0-9]+\/[a-z]\s.)|([^\“\"\']Madde\s[0-9]+\/[A-Z]\s.)|([^\“\"\']Ek\sMadde\s[0-9]+\s[^a-z^0-9^A-Z][^a-z])|([^\“\"\']Geçici\sMadde\s[0-9]+\s[^a-z][^0-9])|([^\“\"\']Ek\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\s[A-Z][a-z]{4}\s[^0-9][^a-z])|(Geçici\sEk\sMadde\s[0-9]\s[^0-9])|(Geçici\sMadde\s[0-9]+\([0-9]+\)\s[^0-9][^a-z])|(Geçici\sMadde\s[0-9]+-[0-9][^0-9][^A-Z][^a-z])|([^\“\"\']Madde\s[0-9]+[a-z]+\s[^0-9][^a-z^A-Z])|(Madde\s[0-9]+\([0-9]+\)\s[^0-9^A-Z^a-z])|(Madde+\s–\s[0-9]+\/[0-9])|(Madde\s[0-9]+-[0-9]+\s[^0-9][^A-Za-z])|([^\“\"\']Madde\s[0-9]+\/[ĞÇİ]\s[^0-9][^A-Z^a-z])|([^\“\"\']Madde\s[^0-9^A-Z^a-z])',text)

	return rega_no, mukerrer_no, rega_tarihi, mevzuat_no, mevzuat_tarihi, len(matching_madde_list)

def extract_yonetmelik_info(text):
    
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " "))
	# extract rega no
	no = re.search(r'([Numara]+:\s|[Numara] +:\s|[Sayı]+:\s|[Sayı] +:\s|[Sayısı] +:\s|[Sayısı]+:\s|[Numarası]+\s)([0-9]+)',text)
	if no:
		rega_no = no.group(2)
		rega_no = int(rega_no)
	else:
		rega_no = "Bulunamadı"
	# extract mukerrer no
	matching_list = re.findall(r'([0-9]+\s(Mükerrer)\sKarar|[0-9]+\s(Mükerrer)\sKanun|Sayısı:\s[0-9]+\s(Mükerrer)\s[A-Za-zİÇĞŞ])',text)
	count_mukerrer = re.search(r'(Sayısı:\s[0-9]+\s([0-9]).\sMükerrer\s)|(\(Mükerrer\)\s([0-9]).\sMükerrer)',text)
	if matching_list:
		mukerrer_no = 1
	elif count_mukerrer == None :
		mukerrer_no = 0
	elif count_mukerrer:
		mukerrer_no = count_mukerrer.group(2) or count_mukerrer.group(4)
		mukerrer_no = int(mukerrer_no)
	else:
		mukerrer_no = 0
	# extract rega tarihi, mevzuat_no, mevzuat_tarihi and madde
	rega = re.search(r'(Tarihi+:\s|Tarihi+\s)([0-9]+.[0-9]+.[0-9]{4})',text)
	if rega:
		rega_tarihi = pd.to_datetime(rega.group(2)).strftime("%Y-%m-%d") #parse(rega.group(2)).strftime('%Y-%m-%d')
	else:
		rega_tarihi = "Bulunamadı"

	matching_madde_list = re.findall(r'([^\“\"\']Madde\s[0-9]+\s–)|([^\“\"\']Madde–\s[0-9]+\/[0-9])|([^\“\"\']Madde\s[0-9]+\/[a-z]\s.)|([^\“\"\']Madde\s[0-9]+\/[A-Z]\s.)|([^\“\"\']Ek\sMadde\s[0-9]+\s[^a-z^0-9^A-Z][^a-z])|([^\“\"\']Geçici\sMadde\s[0-9]+\s[^a-z][^0-9])|([^\“\"\']Ek\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\sMadde\s[^0-9][^a-z])|(Geçici\sEk\sMadde\s[0-9]\s[^0-9])|(Geçici\sMadde\s[0-9]+\([0-9]+\)\s[^0-9][^a-z])|(Geçici\sMadde\s[0-9]+-[0-9][^0-9][^A-Z][^a-z])|([^\“\"\']Madde\s[0-9]+[a-z]+\s[^0-9][^a-z^A-Z])|(Madde\s[0-9]+\([0-9]+\)\s[^0-9^A-Z^a-z])|(Madde+\s–\s[0-9]+\/[0-9])|(Madde\s[0-9]+-[0-9]+\s[^0-9][^A-Za-z])|([^\“\"\']Madde\s[0-9]+\/[ĞÇİ]\s[^0-9][^A-Z^a-z])|([^\“\"\']Madde\s[–])|([^\“\"\']Madde\s[I]+\s[^0-9])',text)

	return rega_no, mukerrer_no, rega_tarihi, len(matching_madde_list)

def extract_komisyon_raporu_info(text):
    #Raporlarında donem bilgisi tespiti beklenmemektedir.
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " "))
	#rega_no:Komisyon Raporunun Sıra Sayısı
	sira_no=re.search(r'(S.\sSayısı\s:.\s([0-9]+\s[a-z]+\sek)|S.\sSayısı\s:\s([0-9]+\sek\s[0-9])+\s|.\sSayısı:\s([0-9]+\s[a-z]+\sek|[0-9]+\s[a-z]+\sk̂|[|][0-9][|]\s[a-z]+\sek|[0-9]+\s[a-z]+\s[0-9]+\s[a-z]+\sek)\s|\sS.\sSayısı\s:\s([0-9]+\s[a-z]+\s[0-9]+\s[a-z]+\sek)|S.\sSayisi:\s([0-9]+\s[a-z]+\sek|[0-9]+\syeek)|S.\sSayısı\s:\s([0-9]+\s[a-z]\silâve)|S.\sSayısı:\s([0-9]+\'[a-z]\s[0-9]\s[a-z]+\sEk)|S#\sS\sA\sY\sI\sS\sI\s:\s|([0-9]+\s[0-9]+\s[0-9]\s©\s[0-9])|Sıra\sNo\s([0-9]+\s[a-z]+\silâve|[0-9]+\s[a-z]+\silave)|S\s.\sSayısı\s:\s([0-9]+\se\se\sk)|s.\sSAYISI\s([0-9]+\s[a-z]+\sek)|S.\sSayısı:\s([0-9]+\s[a-zöü]+\s[a-zöçşiıü]+\sek)|S.\sSayısı\s:\s([0-9]+\s[a-z]+\sek)|S.\sSayısı\s:\s([0-9]+[a-z]\sek)|S.\sSayısı:\s([|][0-9]+\s[a-z]+\s[a-z]+\sek)|S.\sSayısı:\s([0-9]+\s[a-z]+\s[0-9]+\s[a-z]\sek)|s.\ssAYisi;\s([0-9]+\s[a-z]+\s[a-z]+\sek)|S.\sSayısı\s:\s([0-9]+\s[a-z]+\s[I]+\s[a-z]+\sek)|.\ssAYisı\s:([0-9]+\s[a-z]+\sek)|S.\sSayısı\s:\s([0-9]+\'[a-z]+\s[0-9]+\s[a-z]+\sEk)|S\sSayısı\s:\s([0-9]+\s[a-z]\s[|]\s[a-z]+\sek)|Sıra\sN,([0-9]+\s[a-z]+\silâve)|S.:\sSayisi\s([0-9]+\s[a-z]\sek)|S.\s:Sayisi\s([0-9]+\s[a-z]+\sek)|Sıra\sNQ\s([0-9]+\s[a-z]+\silâve)|S.\sSayısı:\s([0-9]+\s[a-z]+\silâve)|.\sSayısı:\s([A-Z][0-9]+[a-z]+\sek)|.\sSAYISI\s:\s([0-9]+\s[a-z]+\sek|[0-9]+\s[a-z]+\s©k|[0-9]+©[\\]\'\s[\^]K)|.\sSayısı:\s([0-9]+[\\]\'[a-z]+\s[0-9]\s[a-z]+\sEk)|\(S.\sSayısı:\s([0-9]+)\)|\(S.\sSayısı\s:\s([0-9]+)\)|Sıra\sSayısı:\s([0-9]+)\s|S.\sSayısı\s:\s([0-9]+\se\se\sk)|Sira\sSayisi:\s([0-9]+)\s|S.\sSayýsý\s:\s([0-9]+)\s|(SlRA\sS\sAYISI:\s|SIRA\sSAYISI:\s|SlRA\sSAYISI:\s|SIRA\sSAYISI\s|SiraSayisi:\s)([0-9]+)|S.\sSayısı\s:\s([0-9]+[\\][a-z][0-9]+[a-z]\s[0-9]+\s[a-z]+\sEk)|S.\sSayısı\s=\s([0-9]+\s[0-9]+\s©\s©\sk)|Sıra\sNo\s([0-9]+\s[a-z]+\sİlâve)|SıraNo([0-9]+[a-z]+ilâve)|s.\sSay,,,:\s([0-9]+\s[a-z]+\s[|]\s[a-z]+\sek)|S.\sSay\si\ssi:\s([0-9]+\seek)|S.\sSayısı\s:\s([0-9]+Ve\s[0-9]+\s[a-z]+\sEk)|Sıra\sN,\s|([0-9]+\s[a-z]+\silâve)|S.\sSayısı\s:\s([0-9]+\s[a-z]+\s[|]\s[a-z]+\sek)|Sayısı\s:\s([0-9]\s©\s[0-9]+)|Sayısı\s:\s([0-9]+\s[0-9]+\s[0-9]+\s&\s[0-9]\s))',text)
	if sira_no:
		sira_no = sira_no.group(2) or sira_no.group(3) or sira_no.group(4) or sira_no.group(5) or sira_no.group(6) or sira_no.group(7) or sira_no.group(8) or sira_no.group(9) or sira_no.group(10)  or sira_no.group(11) or sira_no.group(12) or sira_no.group(13) or sira_no.group(14) or sira_no.group(15) or sira_no.group(16) or sira_no.group(17) or sira_no.group(18) or sira_no.group(19) or sira_no.group(20) or sira_no.group(21) or sira_no.group(22) or sira_no.group(23) or sira_no.group(24) or sira_no.group(25) or sira_no.group(26) or sira_no.group(27) or sira_no.group(28) or sira_no.group(29) or sira_no.group(30) or sira_no.group(31) or sira_no.group(32) or sira_no.group(33) or sira_no.group(34) or sira_no.group(35)  or sira_no.group(36) or sira_no.group(38) or sira_no.group(39) or sira_no.group(40) or sira_no.group(41) or sira_no.group(42) or sira_no.group(43) or sira_no.group(44) or sira_no.group(45) or sira_no.group(46) or sira_no.group(47) or sira_no.group(48) or sira_no.group(49) 
	if sira_no==None:
		sira_no="0"
	if '119a ek' in sira_no:
		sira_no=sira_no.replace("119a ek",'119 ek 1')
		
	find = {'ye ek': 'ek 1', 'a ek': 'ek 1', '© 2': 'ek 2' ,'a ilâve': 'ek 1', 'a ilave': 'ek 1', 
			'e ek':'ek 1', '\'e 2 nci Ek':' ek 2','e 1 nci ek': 'ek 1','yek 1':'ek 1','\'e 1 inci Ek':' ek 1',
			'e ilâve':'ek 1','ya ikinci ek':'ek 2','a dördüncü ek':'ek 4','e e k':'ek 1','ye ikinci ek':'ek 2',
				'yeek':'ek 1','|9|':'90','\'ya 2 nci Ek':' ek 2','e I nci ek':'ek 1','e | ci ek':'ek 1','\'a 1 inci Ek':' ek 1',
			'ye I nci ek':'ek 1','e 1 nei ek':'ek 1','e 2 nci ek':'ek 2','Ve 1 inci Ek':' ek 1','\'ya 1 inci Ek':' ek 1',
			'S53':'153 ','ye k̂':'ek 1','a ©k':'ek 1','138 ve 139 ek 1':'139 ek 1','© 1':'ek 1','yek 2':'ek 2','eek':'ek 1',
			' vek 1':'ek 1','© © k':'ek 1','e İlâve':'ek 1','e üçüncü ek':'ek 3','a beşinci ek':'ek 5','e | nciek':'ek 1','ailâve':' ek 1',
			'\'ye 1 inci Ek':' ek 1','e | nci ek':'ek 1','yek':'ek','ya 2 nci ek':'ek 2','a | nci ek':'ek 1','e 3 nci ek':'ek 3','ya 1 nci ek':'ek 1','119a ek':'119 ek 1',
			'|47 ye ikinci ek':'147 ek 2','©':'ek','a 2 nci ek':'ek 2','5 6 0 & 2':'560 ek 2','4 4 5':'445','5 6 3':'563','7 1':'71'}
	for key, value in find.items():
		sira_no = sira_no.replace(key, value)
	if '|' in sira_no:
		sira_no=sira_no.replace('|','1')

	#donem(string): Komisyon Raporunun ait olduğu Dönemi/21. Dönem öncesi Komisyon
	donem=re.search(r'((Dönem:\s|Dönem\s:\s|YASAMA\sDONEMİ\s|YASAMA\sDÖNEMİ\s|YASAMA\sDÖNEMİ\sYASAMA\sYILI\s|yasama\sdönemi\s|Döneni\s:\s|Dönem\sToplantı\s:\s|Döntm\s:\s)([0-9]+|[A-Zİ][0-9]|[a-zı])\s)',text)
	if donem:
		donem=donem.group(3)
		donem=str(donem)+"".join(". Dönem")
	else: 
		donem='Bulunamadı'

	return sira_no, donem

def extract_genelge_info(text):
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " "))
	#mevzuat_tarihi(date:yyyy-mm-dd): Genelgenin Tarihi
	mevzuat=re.search(r'(Tarih\s([0-9]+\/[0-9]+\/[0-9]{4})\s)|(3\s0\sMaYIS\s2ÖÎ2)|[.]\s([0-9]+\s[A-Za-zıiüçşÜIÇŞ]+\s[0-9]{4})\s|\s([0-9]+\/[0-9]+\/[0-9]{4})\s[A-Za-zçı]+\s[A-Za-z]+\sGENELGE|([0-9]+\/\s[0-9]+\/\s[0-9]{4})\sGENELGE|([0-9]+[.][0-9]+[.][0-9]{4})\s|([0-9]+\s[A-Za-zıiüçşŞÜIÇŞ]+\s[0-9]+\s)[0-9]+.+\sGENELGE|\s([0-9]+\/[0-9]+\/[0-9]{4})\sKonu|Konu:\s([0-9]+.[0-9]+.[0-9]{4})\sGENELGE|([0-9]+\/[A-Za-zğüişç]+\/[0-9]{4})\sKonu|([0-9]+\/[0-9]+\/[0-9]{4})\sGENELGE|([0-9]+\/[0-9]+\/[0-9]{4})\sKONU:|([0-9]+\/\s[0-9]+\/\s[0-9]{4})\sKonu|([0-9]+\s\/[0-9]+\/[0-9]+)\sKonu|([0-9]+\/[0-9]+\/[0-9]+)\starihli|([0-9]+\s[A-ZŞĞIiiüşçÇ]+\s[0-9]{4}\s)GENELGE|([0-9]+\s[A-Za-zŞĞIiiüşçÇ]+\s[0-9]+)\sKonu',text)
	if mevzuat:
		mevzuat_tarihi=mevzuat.group(2) or mevzuat.group(3) or mevzuat.group(4) or mevzuat.group(5) or mevzuat.group(6) or mevzuat.group(7) or mevzuat.group(8) or mevzuat.group(9) or mevzuat.group(10) or mevzuat.group(11) or mevzuat.group(12) or mevzuat.group(13) or mevzuat.group(14) or mevzuat.group(15) or mevzuat.group(16) or mevzuat.group(17) or mevzuat.group(18)   
		if '' in mevzuat_tarihi:
			mevzuat_tarihi = re.sub(" +", " ", mevzuat_tarihi)
		if 'MaYIS2ÖÎ2' in mevzuat_tarihi:
			mevzuat_tarihi=mevzuat_tarihi.replace("MaYIS2ÖÎ2","Mayıs2012")
		if mevzuat_tarihi==None:
			mevzuat_tarihi = mevzuat_tarihi
		else:
			find = {'Ocak':'-1-','Şubat':'-2-','Mart':'-3-','Nisan':'-4-','Mayıs':'-5-','Haziran':'-6-',
					'Temmuz':'-7-','Ağustos':'-8-','Eylül':'-9-','Ekim':'-10-','Kasım':'-11-','Aralık':'-12-',
					'OCAK':'-1-','ŞUBAT':'-2-','MART':'-3-','NİSAN':'-4-','MAYIS':'-5-','HAZİRAN':'-6-',
					'TEMMUZ':'-7-','AĞUSTOS':'-8-','EYLÜL':'-9-','EKİM':'-10-','KASIM':'-11-','ARALIK':'-12-',}
			for key, value in find.items():
				mevzuat_tarihi = mevzuat_tarihi.replace(key, value)
				
		mevzuat_tarihi = parse(mevzuat_tarihi).strftime('%Y-%m-%d') #datetime.datetime.strptime(mevzuat_tarihi,'%d/%m/%Y').strftime('%Y-%m-%d')
	else:
		mevzuat_tarihi = "Bulunamadı"

	#mevzuat_no(string): Genelge Numarası (boş değilse)
	mev_no=re.search(r'(GENELGE\sNO:\s|GENELGE\sNO\s|Genelge\sNo:\s|GENELGE\sNO-\s|Genelge:\s|Genelge\sSayısı\s|Genelge\sNo\s:\s)([0-9]+\/[0-9]+\s([0-9]+))|(SIRA\sNO\s:\s|SERİ\sNO\s|SERİ\sNO:\s|SIRA\sNO:\s|SIRA\sNO:|SERİ\sNO\s:\s|GENELGE\s|G\sE\sN\sE\sL\sG\sE\s|GENELGE\sNo:\s|Genelge\sNo:\s|Genelge\sNo:|GENELGE\sNO\s:\s|GENELGE\sNO:|GENELGE\sNO:\s|GENELGE\sNO\s)([0-9]+\s–\s[0-9]+|[0-9]+\s\/\s[0-9]+|[0-9]+\/[0-9]+\s|([0-9]+)\s|[0-9]+\/\s[0-9]+|[0-9]+-[0-9]+|[0-9]+\s\/\s[0-9]+|[0-9]+|[0-9]+\/[0-9]+\s)|(([0-9]+\/[0-9]+|[0-9]+)\sSayılı\s["[A-Za-z].+Konulu\sGenelge)',text)
	if mev_no: 
		mevzuat_no = mev_no.group(3) or mev_no.group(5) or mev_no.group(8) 
		mevzuat_no = mevzuat_no.strip()
	else: 
		mevzuat_no = np.NaN

	belge=re.search(r'(Sayı\s|SAYI\s:\s|Sayı\s:\s|SAYI\s:|SAYI:\s)((B.[0-9]+[0-9]+.[0-9]+.[A-Zİ]+.[0-9]+.|VUK|B.[0-9]+.[0-9]+.[0-9]+.[A-Zİ]+.[0-9]+.)([0-9]+-[0-9]+-[0-9]+\/[0-9]+|[0-9]+.[0-9]+\/[0-9]+-\s[0-9]+\/[0-9]+|[0-9]+\/[0-9]+-[0-9]+-[0-9]+|[0-9]+.[0-9]+-[0-9]+\[[0-9]+-\s[0-9]+\]-[0-9]+|[0-9]+\/[0-9]+-[0-9]+\s\/[0-9]+|[0-9]+-[0-9]+-[0-9]+\/[0-9]+|[0-9]+\/[0-9]+-\s[0-9]+\/[0-9]+|[0-9]+\/[0-9]+-[0-9]+\s\/\s[0-9]+|[0-9]+\/[0-9]+-[0-9]+|[.][0-9]+\/[0-9]+-[0-9]+\/[0-9]+|[0-9]+.[0-9]+-[0-9]+-[0-9]+.[0-9]+-[0-9]+|[0-9]+.[0-9]+\/[0-9]+-[0-9]+\/[0-9]+-[0-9]+|[0-9]+-[0-9]+.[0-9]+\/[0-9]+|[0-9]+\/[0-9]+.[0-9]+\/[0-9]+-[0-9]+|[0-9]+.[0-9]+\/[0-9]+-[0-9]+-[0-9]+|[0-9]+.[0-9]+\/[0-9]+-[0-9]+|[0-9]+.[0-9]+.[0-9]+.[0-9]+.[0-9]+\[[0-9]+\]\/[0-9]+|\s[0-9]+\/[0-9]+-[0-9]+\/[0-9]+)|[0-9]+\/[0-9]+\/[0-9]+\/[0-9]+)',text)
	if belge:
		belge_sayi = belge.group(2)
	else: belge_sayi = np.NaN

	return mevzuat_tarihi, mevzuat_no, belge_sayi

def extract_ozelge_info(text):
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " "))
	##mevzuat_tarihi(date:yyyy-mm-dd): Özelgenin tarihi
	mevzuat=re.search(r'([0-9]+\/[0-9]+\/[0-9]{4}|[0-9]+[.][0-9]+[.][0-9]{4})(\sKonu|Sayı|[*][0-9]+\sKONU|[*][0-9]+\sKonu|\sSAYI\s:|-[0-9]+\sKonu|\sSayı\s:|\s[……]|\/[0-9]+\sKONU|-[0-9]+\s[……]|-[0-9]+\s|\/\s[0-9]+\sSAYI|\sKONU:\s|\sSn:\s|[*]\s[0-9]+|[*][0-9]+|\sSayın)',text)
	if mevzuat:	
		mevzuat_tarihi = mevzuat.group(1)
		if '.' in mevzuat_tarihi:
			mevzuat_tarihi=mevzuat_tarihi.replace(".","/")
		mevzuat_tarihi = pd.to_datetime(mevzuat_tarihi).strftime("%Y-%m-%d") #parse(mevzuat_tarihi).strftime('%Y-%m-%d')
	else:
		mevzuat_tarihi = "Bulunamadı"

	return mevzuat_tarihi

def extract_teblig_info(text):
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " "))
	#rega_no(int): Tebliğ'nin yayınlanmış olduğu Resmi Gazete Numarası
	no=re.search(r'([Sayısı]+:\s|[Numarası]+\s)([0-9]+) ',text)
	if no:
		rega_no = int(no.group(2))
	else:
		rega_no = "Bulunamadı"

	#mukerrer_no:mukerrer_no(int): İçeriğin Resmi Gazete Mükerrer Sayısı (0 ise mükerrer olmadığını)
	find=re.findall(r'([0-9]+\s(Mükerrer)\sKarar|[0-9]+\s(Mükerrer)\sKanun)',text)
	count_mukerrer = re.search(r'((\s[0-9]).\sMükerrer\s[A-Za-z]+)',text)
	if find:
		mukerrer_no=1
	elif count_mukerrer==None :
		mukerrer_no=0
	elif count_mukerrer:
		mukerrer_no=int(count_mukerrer.group(2))
	else:
		mukerrer_no=0

	#rega_tarihi(date:yyyy-mm-dd): İçeriğin yayınlandığı Resmi Gazete Tarihi
	rega=re.search(r'([Tarihi]+:\s|[Tarihi]+\s)([0-9]+.[0-9]+.[0-9]{4})',text)
	if rega:
		rega_tarihi = pd.to_datetime(rega.group(2)).strftime("%Y-%m-%d") #parse(rega.group(2)).strftime('%Y-%m-%d')
	else:
		rega_tarihi = "Bulunamadı"

	return rega_no, mukerrer_no, rega_tarihi

def extract_resmi_gazete_info(text):
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " "))
	#rega_no:Resmi Gazete'nin Sayısı
	rega_no=re.search(r'(Sayı:\s|Sayı\s:\s|SAYI:\s|Say\sı\s:\s|SAYI\s:\s|SAYİ:\s|Sayı:\s,|Sayı:)([0-9]\s.{7}|[0-9]+)|(([0-9]+)\sSayılı\sResmî\sGazete)|(S\sA\sY\s[İI]\s:\s([0-9]\s[0-9]\s[0-9]\s[0-9]))|(S\sA\sY\s[İI]\s:\s([0-9]+))|(S\sa\sy\sı\s:\s([0-9]\s[0-9]\s[0-9]\s[0-9]\s[0-9]+))|(S\sa\sy\sı\s:\s([0-9]+))',text)
	if rega_no:
		rega_no = rega_no.group(2) or rega_no.group(4) or rega_no.group(6) or rega_no.group(8) or rega_no.group(10) or rega_no.group(12)
		rega_no = rega_no.strip()
		rega_no = int(rega_no)
	else:
		rega_no = "Bulunamadı"

	#mukerrer_no:Resmi Gazete'nin ilgili sayısı için kaçıncı Mükerrer olduğu (0 ise mükerrer olmadığını belirtir)
	find=re.findall(r'(:\s[0-9]+\s(Mükerrer)\s[A-Z])',text)
	count_mükerrer=re.search(r'((\s[0-9]).\sMükerrer\s[A-Za-z]+)',text)
	if find:
		mukerrer_no=1
	elif count_mükerrer==None :
		mukerrer_no=0
	elif count_mükerrer:
		mukerrer_no=int(count_mükerrer.group(2))
	else:
		mukerrer_no=0

	#rega_tarihi(date:yyyy-mm-dd): Resmi Gazete ilgili sayısının yayımlandığı tarihi
	rega_tarihi=re.search(r'([0-9]+\s[A-Za-zğŞÇıiüçşğĞÜİ]+\s[0-9]+|[0-9]+\sK\sÂ\sN\sU\sN\sU\sS\sA\sN\sI\s[0-9]{4}|[0-9]+\s[A-Za-zğŞÇiüçşğĞÜİ]+\s[0-9]+[A-Za-z]+|[0-9]+\s[A-Za-zi]+\s[A-Za-zi]\s[A-Za-z]\s[A-Za-z]\s[0-9]{4}|[0-9]+\s[A-Za-z]+\s[A-Za-zğ]\s[A-Za-z]\s[A-Za-z]\s[A-Za-z]\s[A-Za-z]\s[0-9]{4}|[0-9]+\s[A-Z]+\s[A-Z]\s[A-Z]\s[A-ZIÜ]\s[A-Z]\s[0-9]{4}|[0-9]+\s[A-ZİÇĞÜ]+\s[0-9]+S[0-9]+|[0-9]\s[0-9]\s[A-ZIÇŞ]+\s[0-9]\s[0-9]\s[0-9]\s[0-9]|[0-9]+\s[A-ZÎa-zı]+\s[0-9]{4}|[0-9]+\s[A-Za-z]\s[A-ZĞğa-z]\s[A-Za-z]\s[A-Za-z]\s[A-Za-z]\s[A-Za-z]\s[A-Za-z]\s[0-9]{4}|[0-9]+\s[A-ZİĞ]+\s[0-9]\s[0-9]\s[0-9]\s[0-9]|S\s[A-ZŞÇİa-z]+\s[0-9]{4}|[0-9]\s[0-9]\s[A-ZİÇŞ]+\s[0-9]\s[0-9]\s[0-9]\s[0-9]\s|[0-9]+\s[A-Za-z]\s[A-Za-z]\s[A-Za-z]\s[A-Za-z]\s[A-Za-z]\s[0-9]{4}|[0-9]+\s[A-Zİ]+\s[A-ZİI]+[0-9]+|[0-9]+\sKÂNUNUSANİ\s[0-9][A-Z][0-9]+|[0-9]\s[0-9]+\s[A-Za-z]\s[A-Za-z]\s[A-Za-z]\s[A-Za-z]\s[0-9]\s[0-9]\s[0-9]\s[0-9]\s|[0-9]+\s[A-Za-z]\s[A-Za-z]\s[A-Za-z]\s[A-Za-zı]\s[A-Za-z]\s[0-9]{4}|[0-9]\s[0-9]\s[A-Za-zIİ]\s[A-Za-zIİ]\s[A-Za-zIİ]\s[A-Za-zIİ]\s[A-Za-zIİ]\s[0-9]\s[0-9]\s[0-9]\s[0-9]|[0-9]+\sK\sA\sN\sU\sN\sU\sE\sV\sV\sE\sL\sm\si|[0-9]+\sTEŞRİNİEVVEL\s[0-9]{4}|[0-9]+\sKÂNUNUSANİ\s[0-9]{4}|[0-9]+\s[A-Z]+\s[0-9]+[A-Z][0-9]\s|[0-9]+\s[A-Z]\s[A-ZĞ]\s[A-Z]\s[A-Z]\s[A-Z]\s[A-Z]\s[A-Z]\s[0-9]\s[0-9]\s[0-9]\s[0-9]|[0-9]+\sT\sE\sŞ\sR\sİ\sN\sİ\sS\sA\sN\sİ\s[0-9]{4})(\s[A-ZÇŞİÎ]+\sSayı\s:|\sSayı:|\s[A-Zİa-zı]+\s[A-Zİa-zı]\s[A-Zİa-zı]+\s[A-ZŞİa-zı]\s[A-Zİa-zı]+\s[A-Zİa-zı]\s[A-Zİa-zı]+\s[A-Zİa-zı]\s|\s[A-ZÇŞİ]+\sS\sa\sy\sı\s|\sSayı\s:|\s[A-ZÇŞİ]+\sS\sA\sY\sI\s:|\s[A-ZŞÇ]+\sSAYI:|\s[A-ZŞÇ]+\sSAYI\s:|\sTarihli|\s[A-ZİŞÇa-z]+\sSayı:|\sSALı|\sS\sa\sy\sı\s:|\s[A-ZİÇa-zı]+\s[A-Zİa-zı]\s[A-Zİa-zı]+\s[A-ZŞİa-zı]\s[A-Zİa-zı]+\s[A-Zİa-zı]\s[A-Zİa-zı]+\s[A-Zİa-zı]+\s|\s—\sSayı\s:|[A-ZŞÇÜİ]+\s[\\]+\si\sSayı:|\s[A-ZŞÇİ]+\sSayı\s|\s[A-Z]\s[A-Z]\s[A-Z]\s[A-Z]+\sSayı\s:|\sÇARŞAMBA|\s[A-Z]\s[A-Z]\s[A-Z]\s[A-Z]\sSayı:|\s[A-Z]\s[A-Z]\s[A-Z]\s[A-Z]\s[A-Z]\sSayı:|\sGUJVÎAK[.]TESÎ\sSayı:|\s[A-Z]\s[A-Z]\s[A-Z]\s[a-zı]\sSayı:|\s[A-ZİÜÇŞÎ]+\s-\sSayı\s:|\s[A-ZÇŞİÜ]+\sS\sA\sY\sİ\s:|\s[A-ZŞÇİÜ]+\sSAYI:|[A-ZİŞÇ]+\sS\sa\sy\sı\s:|\s[A-Z]\s[A-Z]\s[A-Z]\s[A-Z]\sSay\sı\s:|\s[A-Z]\s[A-Z]\s[A-Z]\s[A-Z]\s[A-Z]\sSayı\s:|\sPAZAR|\sGenel\sMüdürlüğüne\s[A-Za-zŞÇÜİI]+\sı\sBaşvurulur\s[A-Za-zŞÇÜİIÎ]+\sSayı\s:|\s[A-Za-z]\s[A-ZŞÇÜ]\s[A-Z]\s[A-Z]\sSAYI:|S[0-9]+\s[A-Z]+\s[A-Z]\sKANUNLA\sR\sSayı:|\s[A-Z]+\s,\sSayı:|[A-ZŞÇİÜ]+\sYönetim)',text)
	if rega_tarihi:
		rega_tarihi= rega_tarihi.group(1)
	else:
		rega_tarihi = "Bulunamadı"
  
	if '' in rega_tarihi:
		rega_tarihi=rega_tarihi.replace(" ","")
	find = {'Ocak':'-1-','Şubat':'-2-','Mart':'-3-','Nisan':'-4-','Mayıs':'-5-','Haziran':'-6-',
			'Temmuz':'-7-','Ağustos':'-8-','Eylül':'-9-','Ekim':'-10-','Kasım':'-11-','Aralık':'-12-',
			'OCAK':'-1-','ŞUBAT':'-2-','MART':'-3-','NİSAN':'-4-','MAYIS':'-5-','HAZİRAN':'-6-',
			'TEMMUZ':'-7-','AĞUSTOS':'-8-','EYLÜL':'-9-','EKİM':'-10-','KASIM':'-11-','ARALIK':'-12-','INÎSAN':'-4-',
			'KÂNUNUSANI':'-1-','198f':'1981','11965':'1965','8TEŞRİ-4-İ1930':'08-11-1930','25TEŞRİ-4-İ1931':'25-11-1931',
			'KANUNUEVVELmi':'-12-1931','Teşrinievvel':'-10-','TEŞRİNİEVVEL':'-10-','KÂNUNUSANİ':'-1-','EKÎMı':'-10-','EKlM':'-10-',
			'19S2':'1962','1S32':'1932','ACUSTOS':'-8-','117-8-1967':'17-8-1967','3-7-11968':'3-7-1968','8-4-Iİ968':'8-4-1968',
			'S-10-1983':'5-10-1983','»':'3','L3':'13'}
	for key, value in find.items():
		rega_tarihi = rega_tarihi.replace(key, value)
	rega_tarihi= pd.to_datetime(rega_tarihi).strftime("%Y-%m-%d") #parse(rega_tarihi).strftime('%Y-%m-%d')
	

	return rega_no, mukerrer_no, rega_tarihi