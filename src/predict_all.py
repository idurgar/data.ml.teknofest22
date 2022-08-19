import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import pickle
from tensorflow.keras.models import load_model
from modelling import *
from dateutil.parser import parse
import datetime
from text_preprocessing import preprocessing

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
	
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " ").replace("*", " ").lstrip().rstrip())
	preprocessed_text = preprocessing(text)

	if (re.search("Karar Sayısı: KHK/(\d+) Kararnamenin", text)) or \
		(re.search("Karar Sayısı : KHK/(\d+) Kararnamenin", text)) or \
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

	#elif (re.search("HAKKINDA CUMHURBAŞKANLIĞI KARARNAMESİ (Kararname Numarası: (\d+))", preprocessed_text)) or \
	#		(re.search("DAİR CUMHURBAŞKANLIĞI KARARNAMESİ (Kararname Numarası: (\d+))", preprocessed_text)) or \
	#		(re.search("İLİŞKİN CUMHURBAŞKANLIĞI KARARNAMESİ (Kararname Numarası: (\d+))", preprocessed_text)) or \
	#		(re.search("İLİŞKİN CUMHURBAŞKANLIĞI KARARNAMESİ (Kararname Numarası : (\d+))", preprocessed_text)) or \
	#		(re.search("İLİŞKİN CUMHURBAŞKANLIĞI KARARNAMESİ (Kararname Numarası:(\d+))", preprocessed_text)) or \
	#		(re.search("Cumhurbaşkanlığı Kararnamesinin Sayısı : (\d+)", preprocessed_text)) or \
	#		(re.search("Cumhurbaşkanlığı Kararnamesinin Sayısı: (\d+)", preprocessed_text)) or \
	#		(re.search("Cumhurbaşkanlığı Kararnamesinin Sayısı :(\d+)", preprocessed_text)) or \
	#		(re.search("Cumhurbaşkanlığı Kararnamesinin Sayısı:(\d+)", preprocessed_text)) or \
	#		(re.search("HAKKINDA CUMHURBAŞKANLIĞI KARARNAMESİ", preprocessed_text)) or \
	#		(re.search("DAİR CUMHURBAŞKANLIĞI KARARNAMESİ", preprocessed_text)) or \
	#		(re.search("ilişkin cumhurbaşkanlığı kararnamesi kararname numarası", preprocessed_text)) or \
	#		(re.search("hakkında cumhurbaşkanlığı kararnamesi kararname numarası", preprocessed_text)) or \
	#		(re.search("dair cumhurbaşkanlığı kararnamesi kararname numarası", preprocessed_text)) or \
	#		(re.search("hakkında cumhurbaşkanlığı kararnamesi", preprocessed_text)) or \
	#		(re.search("cumhurbaşkanlığı kararnamesinin sayısı", preprocessed_text)) or \
	#		(re.search("dair cumhurbaşkanlığı kararnamesi", preprocessed_text)):
	#	prediction = "Cumhurbaşkanlığı Kararnamesi" 

	else:
		model_prediction = predict(model, word_index, max_len, preprocessed_text)[0]
  
		if model_prediction == 0:
			prediction = "Cumhurbaşkanlığı Kararnamesi"	
 
		elif model_prediction == 1:
			prediction = "Genelge"
			
		elif model_prediction == 2:
			prediction = "Kanun"

		elif model_prediction == 4:
			prediction = "Komisyon Raporu"

		elif model_prediction == 5:
			prediction = "Resmi Gazete"
			
		elif model_prediction == 6:
			prediction = "Tebliğ"
	
		elif model_prediction == 7:
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

		elif model_prediction == 8:
			if re.findall("HAKKINDA TEBLİĞ|DAİR TEBLİĞ|TEBLİĞİ", text):
				prediction = "Tebliğ"
			else:    
				prediction = "Yönetmelik"

		else:
			prediction = "Özelge"

	return prediction

ay_dict = { 'Ocak':'-1-', 'O C A K': '-1-', 'O c a k': "-1-",
			'Şubat':'-2-', 'Ş U B A T': '-2-', 'Ş u b a t': '-2-',
			'Mart':'-3-', 'M A R T': '-3-', 'M a r t':'-3-',
			'Nisan':'-4-',  'N İ S A N': '-4-', "N i s a n": '-4-',
			'Mayıs':'-5-', 'M A Y I S': '-5-', "M a y ı s": '-5-',
			'Haziran':'-6-', 'H A Z İ R A N': '-6-', "H a z i r a n": '-6-',
			'Temmuz':'-7-', 'T E M M U Z': '-7-',  "T e m m u z": '-7-',
			'Ağustos':'-8-', 'A Ğ U S T O S': '-8-', "A ğ u s t o s": '-8-',
			'Eylül':'-9-', 'E Y L Ü L': '-9-', "E y l ü l": '-9-',
			'Ekim':'-10-', 'E K İ M': '-10-', "E k i m": '-10-',
			'Kasım':'-11-', 'K A S I M': '-11-', "K a s ı m": '-11-',
			'Aralık':'-12-', 'A R A L I K': '-12-', "A r a l ı k": '-12-',
			'OCAK':'-1-','ŞUBAT':'-2-','MART':'-3-','NİSAN':'-4-','MAYIS':'-5-','HAZİRAN':'-6-',
			'TEMMUZ':'-7-','AĞUSTOS':'-8-','EYLÜL':'-9-','EKİM':'-10-','KASIM':'-11-','ARALIK':'-12-','INÎSAN':'-4-',
			'KÂNUNUSANI':'-1-','198f':'1981','11965':'1965','8TEŞRİ-4-İ1930':'08-11-1930','25TEŞRİ-4-İ1931':'25-11-1931',
			'KANUNUEVVELmi':'-12-1931','Teşrinievvel':'-10-','TEŞRİNİEVVEL':'-10-','KÂNUNUSANİ':'-1-', 'K Â N U N U S A N I':'-1-', 'EKÎMı':'-10-','EKlM':'-10-',
			'19S2':'1962','1S32':'1932','ACUSTOS':'-8-','117-8-1967':'17-8-1967','3-7-11968':'3-7-1968','8-4-Iİ968':'8-4-1968',
			'S-10-1983':'5-10-1983','»':'3','L3':'13'}

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
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " ").replace("*", " ").lstrip().rstrip())
	try:
		# extract rega no
		no=re.search(r'(Sayısı:\s|Numarası\s|Numarası\s:\s|Numarası:\s|Numarası\s:|Sayısı\s|Sayısı\s:\s[0-9]+\/[0-9]+\/[0-9]{4}\s[-]\s|Sayısı\s:|^Kararnamesinin\sSayısı\s:\s|Gazete\sTarihi\s:\s[0-9]+.[0-9]+.[0-9]+,\sNo\s:\s|Gazete\sTarihi\s:\s[0-9]+.[0-9]+.[0-9]+,\sNo:\s|Gazete\sTarihi\s:\s[0-9]+.[0-9]+.[0-9]+,\sNo\s:|Gazete\sTarihi\s:\s[0-9]+.[0-9]+.[0-9]+,\sNo\s|Sayı\s:\s|Sayı\s|Sayı:\s|Sayı\s:)([0-9]+)',text)
		if no:
			rega_no = no.group(2)
		else:
			rega_no = np.NaN

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
		rega = re.search(r'(Tarihi:\s|Tarihi +:\s|Gazete\sTarihi+:\s|Gazete\sTarihi+\s|Gazete\sTarihi\s:\s|Gazete\sTarihi\s:)([0-9]+.[0-9]+.[0-9]{4})',text)	
		if rega:
			rega_tarihi = rega.group(2)
			if "" in rega_tarihi:
				rega_tarihi = rega_tarihi.replace(" ", "")
			if "." in rega_tarihi:
				rega_tarihi = rega_tarihi.replace(".", "-")
			if "/" in rega_tarihi:
				rega_tarihi = rega_tarihi.replace("/", "-")
			try:
				rega_tarihi= pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				pass
			try:
				rega_tarihi = pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				try:
					rega_tarihi = pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
				except Exception:
					try:
						rega_tarihi = pd.to_datetime("-".join([i.replace(" ", "") for i in rega_tarihi.split("/")])).strftime("%Y-%m-%d")
					except Exception:
						pass
		else:
			rega_tarihi = np.NaN

		mev_no = re.search(r'(No+.\s+|Numarası+.\s:\s|Sayısı\s+:\sKHK\/|Sayısı:\sKHK\/|Sayısı\s:\sKHK\/|No\s:\s|No:\s|No\s:|Numarası\s:\s|Numarası\s|numarası\s:|Numarası:\s)([0-9]+)',text)
		if mev_no:
			mevzuat_no = mev_no.group(2)
		else:
			mevzuat_no = np.NaN

		mevzuat = re.search(r'(Kabul\sTarihi+:\s|Kabul\sTarihi+\s|Kabul\sTarihi\s:\s|Kabul\sTarihi\s:)([0-9]+.[0-9]+.[0-9]+)',text)
		if mevzuat:
			mevzuat_tarihi = mevzuat.group(2)
			if "" in mevzuat_tarihi:
				mevzuat_tarihi = mevzuat_tarihi.replace(" ", "")
			if "." in mevzuat_tarihi:
				mevzuat_tarihi = mevzuat_tarihi.replace(".", "-")
			if "/" in mevzuat_tarihi:
				mevzuat_tarihi = mevzuat_tarihi.replace("/", "-")
			for key, value in ay_dict.items():
				if key in mevzuat_tarihi:
					mevzuat_tarihi = mevzuat_tarihi.replace(key, value)
			try:
				mevzuat_tarihi= pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				pass
			try:
				mevzuat_tarihi = pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				try:
					mevzuat_tarihi = pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
				except Exception:
					try:
						mevzuat_tarihi = pd.to_datetime("-".join([i.replace(" ", "") for i in mevzuat_tarihi.split("/")])).strftime("%Y-%m-%d")
					except Exception:
						pass
		else:
			mevzuat_tarihi = np.NaN	

		matching_madde_list = re.findall(r'([^\“\"\']Madde\s[0-9]+\s–)|([^\“\"\']Madde–\s[0-9]+\/[0-9])|([^\“\"\']Madde\s[0-9]+\/[a-z]\s.)|([^\“\"\']Madde\s[0-9]+\/[A-Z]\s.)|([^\“\"\']Ek\sMadde\s[0-9]+\s[^a-z^0-9^A-Z][^a-z])|([^\“\"\']Geçici\sMadde\s[0-9]+\s[^a-z][^0-9]+[A-Za-z0-9]\s[A-Za-z]+[[^\“\"\'])|([^\“\"\']Ek\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\s[A-Z][a-z]{4}\s[^0-9][^a-z])|(Geçici\sEk\sMadde\s[0-9]\s[^0-9])|(Geçici\sMadde\s[0-9]+\([0-9]+\)\s[^0-9][^a-z])|(Geçici\sMadde\s[0-9]+-[0-9][^0-9][^A-Z][^a-z])|([^\“\"\']Madde\s[0-9]+[a-z]+\s[^0-9][^a-z^A-Z])|(Madde\s[0-9]+\([0-9]+\)\s[^0-9^A-Z^a-z])|(Madde+\s–\s[0-9]+\/[0-9])|(Madde\s[0-9]+-[0-9]+\s[^0-9][^A-Za-z])',text)
		madde_sayisi = len(matching_madde_list) 
	except:
		rega_no, mukerrer_no, rega_tarihi, mevzuat_tarihi, mevzuat_no, madde_sayisi = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN

	return rega_no, mukerrer_no, rega_tarihi, mevzuat_tarihi, mevzuat_no, madde_sayisi

def extract_kanun_hukmunde_kararname_info(text):
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " ").replace("*", " ").lstrip().rstrip())
	try:
		# extract rega no
		no = re.search(r'(Sayısı:\s|Numarası\s|Numarası\s:\s|Numarası:\s|Numarası\s:|Sayısı\s|Sayısı\s:\s[0-9]+\/[0-9]+\/[0-9]{4}\s[-]\s|Sayısı\s:|^Kararnamesinin\sSayısı\s:\s)([0-9]+)',text)
		if no:
			rega_no = no.group(2)
			if not rega_no:
				rega_no = re.search(r'\d+', no.group(0)).group(0)
		else:
			rega_no = np.NaN

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
		rega = re.search(r'(Gazete\sTarihi+:\s|Gazete\sTarihi+\s|Gazete\sTarihi\s:\s|Gazete\sTarihi\s:)([0-9]+.[0-9]+.[0-9]{4})',text)
		if rega:
			rega_tarihi = rega.group(2)
			for key, value in ay_dict.items():
				if key in rega_tarihi:
					rega_tarihi = rega_tarihi.replace(key, value)
			if "" in rega_tarihi:
				rega_tarihi = rega_tarihi.replace(" ", "")
			if "." in rega_tarihi:
				rega_tarihi = rega_tarihi.replace(".", "-")
			if "/" in rega_tarihi:
				rega_tarihi = rega_tarihi.replace("/", "-")
			try:
				rega_tarihi= pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				pass
			try:
				rega_tarihi = pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				try:
					rega_tarihi = pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
				except Exception:
					try:
						rega_tarihi = pd.to_datetime("-".join([i.replace(" ", "") for i in rega_tarihi.split("/")])).strftime("%Y-%m-%d")
					except Exception:
						pass
		else:
			rega_tarihi = np.NaN

		mev_no = re.search(r'(No+.\s+|Numarası+.\s:\s|Sayısı\s+:\sKHK\/|Sayısı:\sKHK\/|Sayısı\s:\sKHK\/|No\s:\s|No:\s|No\s:|Numarası\s:\s|Numarası\s|numarası\s:|Numarası:\s)([0-9]+)',text)
		if mev_no:
			mevzuat_no = mev_no.group(2)
		else:
			mevzuat_no = np.NaN

		mevzuat = re.search(r'(Kararnamenin\sTarihi+:\s|Kararnamenin\sTarihi+\s|Kararnamenin\sTarihi\s:\s|Kararnamenin\sTarihi\s:)([0-9]+.[0-9]+.[0-9]+)',text)
		if mevzuat:
			mevzuat_tarihi = mevzuat.group(2)
			if "" in mevzuat_tarihi:
				mevzuat_tarihi = mevzuat_tarihi.replace(" ", "")
			if "." in mevzuat_tarihi:
				mevzuat_tarihi = mevzuat_tarihi.replace(".", "-")
			if "/" in mevzuat_tarihi:
				mevzuat_tarihi = mevzuat_tarihi.replace("/", "-")
			for key, value in ay_dict.items():
				if key in mevzuat_tarihi:
					mevzuat_tarihi = mevzuat_tarihi.replace(key, value)
			try:
				mevzuat_tarihi= pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				pass
			try:
				mevzuat_tarihi = pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				try:
					mevzuat_tarihi = pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
				except Exception:
					try:
						mevzuat_tarihi = pd.to_datetime("-".join([i.replace(" ", "") for i in mevzuat_tarihi.split("/")])).strftime("%Y-%m-%d")
					except Exception:
						pass
		else:
			mevzuat_tarihi = np.NaN	

		matching_madde_list = re.findall(r'([^\“\"\']Madde\s[0-9]+\s–)|([^\“\"\']Madde–\s[0-9]+\/[0-9])|([^\“\"\']Madde\s[0-9]+\/[a-z]\s.)|([^\“\"\']Madde\s[0-9]+\/[A-Z]\s.)|([^\“\"\']Ek\sMadde\s[0-9]+\s[^a-z^0-9^A-Z][^a-z])|([^\“\"\']Geçici\sMadde\s[0-9]+\s[^a-z][^0-9]+[A-Za-z0-9]\s[A-Za-z]+[[^\“\"\'])|([^\“\"\']Ek\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\s[A-Z][a-z]{4}\s[^0-9][^a-z])|(Geçici\sEk\sMadde\s[0-9]\s[^0-9])|(Geçici\sMadde\s[0-9]+\([0-9]+\)\s[^0-9][^a-z])|(Geçici\sMadde\s[0-9]+-[0-9][^0-9][^A-Z][^a-z])|([^\“\"\']Madde\s[0-9]+[a-z]+\s[^0-9][^a-z^A-Z])|(Madde\s[0-9]+\([0-9]+\)\s[^0-9^A-Z^a-z])|(Madde+\s–\s[0-9]+\/[0-9])|(Madde\s[0-9]+-[0-9]+\s[^0-9][^A-Za-z])',text)
		madde_sayisi = len(matching_madde_list)
	except:
		rega_no, mukerrer_no, rega_tarihi, mevzuat_no, mevzuat_tarihi, madde_sayisi = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN

	return rega_no, mukerrer_no, rega_tarihi, mevzuat_no, mevzuat_tarihi, madde_sayisi
	
def extract_cumhurbaskanligi_kararnamesi_info(text):
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " ").replace("*", " ").lstrip().rstrip())
	try:
		# extract rega no
		no = re.search(r'(Sayısı:\s|Numarası\s|Numarası\s:\s|Numarası:\s|Numarası\s:|Sayısı\s:\s[0-9]+\/[0-9]+\/[0-9]{4}\s[-]\s|Sayısı\s:\s[0-9]+\/[0-9]+\/[0-9]+[0-9]\s–\s|Sayısı\s|Sayısı\s:|^Kararnamesinin\sSayısı\s:\s)([0-9]+)',text)
		if no:
			rega_no = no.group(2)
			if not rega_no:
				rega_no = re.search(r'\d+', no.group(0)).group(0)
		else:
			rega_no = np.NaN

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

		rega = re.search(r'(Tarihi+:\s|Tarihi+\s|Tarihi\s:\s|Tarihi\s:|Tarihi\s–\sSayısı\s:\s)([0-9]+.[0-9]+.[0-9]{4})',text)
		if rega:
			rega_tarihi = rega.group(2)
			if "" in rega_tarihi:
				rega_tarihi = rega_tarihi.replace(" ", "")
			if "." in rega_tarihi:
				rega_tarihi = rega_tarihi.replace(".", "-")
			if "/" in rega_tarihi:
				rega_tarihi = rega_tarihi.replace("/", "-")
			for key, value in ay_dict.items():
				if key in rega_tarihi:
					rega_tarihi = rega_tarihi.replace(key, value)
			try:
				rega_tarihi= pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				pass
			try:
				rega_tarihi = pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				try:
					rega_tarihi = pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
				except Exception:
					try:
						rega_tarihi = pd.to_datetime("-".join([i.replace(" ", "") for i in rega_tarihi.split("/")])).strftime("%Y-%m-%d")
					except Exception:
						pass
		else:
			rega_tarihi = np.NaN	

		mev_no = re.search(r'(Kararname\sNumarası:|Kararname\sNumarası:\s|Kararname\sNumarası:|\(Kararname\sSayısı:|Kararname\sSayısı:\s|\(Kararname\sNumarası\s|\(KARARNAME\sNUMARASI:\s|Kararnamesinin\sSayısı\s:\s|Kararnamesinin\sSayısı\s\s:\s)([0-9]+)',text)
		if mev_no:
			mevzuat_no = mev_no.group(2)
		else:
			mevzuat_no = np.NaN

		mevzuat = re.search(r'([.]\s|yürütür\s)([0-9]{1,2}\s[A-ZŞİĞİ][a-zğçşü].+[0-9]{4})',text)
		if mevzuat:
			mevzuat_tarihi = mevzuat.group(2)
			if "" in mevzuat_tarihi:
				mevzuat_tarihi = mevzuat_tarihi.replace(" ", "")
			if "." in mevzuat_tarihi:
				mevzuat_tarihi = mevzuat_tarihi.replace(".", "-")
			if "/" in mevzuat_tarihi:
				mevzuat_tarihi = mevzuat_tarihi.replace("/", "-")
			for key, value in ay_dict.items():
				if key in mevzuat_tarihi:
					mevzuat_tarihi = mevzuat_tarihi.replace(key, value)
			try:
				mevzuat_tarihi= pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				pass
			try:
				mevzuat_tarihi = pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				try:
					mevzuat_tarihi = pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
				except Exception:
					try:
						mevzuat_tarihi = pd.to_datetime("-".join([i.replace(" ", "") for i in mevzuat_tarihi.split("/")])).strftime("%Y-%m-%d")
					except Exception:
						pass
		else:
			mevzuat_tarihi = np.NaN

		matching_madde_list = re.findall(r'([^\“\"\']Madde\s[0-9]+\s–)|([^\“\"\']Madde–\s[0-9]+\/[0-9])|([^\“\"\']Madde\s[0-9]+\/[a-z]\s.)|([^\“\"\']Madde\s[0-9]+\/[A-Z]\s.)|([^\“\"\']Ek\sMadde\s[0-9]+\s[^a-z^0-9^A-Z][^a-z])|([^\“\"\']Geçici\sMadde\s[0-9]+\s[^a-z][^0-9]+[A-Za-z0-9]\s[A-Za-z]+[[^\“\"\'])|([^\“\"\']Ek\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\s[A-Z][a-z]{4}\s[^0-9][^a-z])|(Geçici\sEk\sMadde\s[0-9]\s[^0-9])|(Geçici\sMadde\s[0-9]+\([0-9]+\)\s[^0-9][^a-z])|(Geçici\sMadde\s[0-9]+-[0-9][^0-9][^A-Z][^a-z])|([^\“\"\']Madde\s[0-9]+[a-z]+\s[^0-9][^a-z^A-Z])|(Madde\s[0-9]+\([0-9]+\)\s[^0-9^A-Z^a-z])|(Madde+\s–\s[0-9]+\/[0-9])|(Madde\s[0-9]+-[0-9]+\s[^0-9][^A-Za-z])|([^\“\"\']Madde\s[0-9]+\/[ĞÇİ]\s[^0-9][^A-Z^a-z])',text)
		madde_sayisi = len(matching_madde_list)

	except:
		rega_no, mukerrer_no, rega_tarihi, mevzuat_no, mevzuat_tarihi, madde_sayisi = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN

	return rega_no, mukerrer_no, rega_tarihi, mevzuat_no, mevzuat_tarihi, madde_sayisi

def extract_tuzuk_info(text):
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " ").replace("*", " ").lstrip().rstrip())
	try:
		# extract rega no
		no=re.search(r'(Sayısı:\s|Numarası\s|Numarası\s:\s|Numarası:\s|Numarası\s:|Sayısı\s|Sayısı\s:\s[0-9]+\/[0-9]+\/[0-9]{4}\s[-]\s|Sayısı\s:|^Kararnamesinin\sSayısı\s:\s|Gazete\sTarihi\s:\s[0-9]+.[0-9]+.[0-9]+,\sNo\s:\s|Gazete\sTarihi\s:\s[0-9]+.[0-9]+.[0-9]+,\sNo:\s|Gazete\sTarihi\s:\s[0-9]+.[0-9]+.[0-9]+,\sNo\s:|Gazete\sTarihi\s:\s[0-9]+.[0-9]+.[0-9]+,\sNo\s)([0-9]+)',text)
		if no:
			rega_no = no.group(2)
			if not rega_no:
				rega_no = re.search(r'\d+', no.group(0)).group(0)
		else:
			rega_no = np.NaN

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
		rega = re.search(r'(Tarihi:\s|Tarihi\s|Tarihi\s:\s|Tarihi\s:)([0-9]+.[0-9]+.[0-9]{4})',text)
		if rega:
			rega_tarihi = rega.group(2)
			if "" in rega_tarihi:
				rega_tarihi = rega_tarihi.replace(" ", "")
			if "." in rega_tarihi:
				rega_tarihi = rega_tarihi.replace(".", "-")
			if "/" in rega_tarihi:
				rega_tarihi = rega_tarihi.replace("/", "-")
			
			for key, value in ay_dict.items():
				if key in rega_tarihi:
					rega_tarihi = rega_tarihi.replace(key, value)
			try:
				rega_tarihi= pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				pass
			try:
				rega_tarihi = pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				try:
					rega_tarihi = pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
				except Exception:
					try:
						rega_tarihi = pd.to_datetime("-".join([i.replace(" ", "") for i in rega_tarihi.split("/")]), format="%d-%m-%Y").strftime("%Y-%m-%d")
					except Exception:
						pass
		else:
			rega_tarihi = np.NaN

		mev_no = re.search(r'(BKK\sNo:\s|BKK\sNo\s:|BKK\sNo\s:\s|Kanunun\sTarihi\s:\s[0-9]+.[0-9]+.[0-9]{4},\sNo\s:\s)(([0-9]+\/[0-9]+|[0-9]+)\s)',text)
		if mev_no:
			mevzuat_no = mev_no.group(2)
			if '' in mevzuat_no:
				mevzuat_no=mevzuat_no.strip()
		else:
			mevzuat_no = np.NaN

		mevzuat = re.search(r'(Karar\sTarihi:\s|Karar\sTarihi\s:\s|Karar\sTarihi\s:|Karar\sTarihi\s)(([0-9]+.[0-9]+.[0-9]{4}),)',text)
		if mevzuat:
			mevzuat_tarihi = mevzuat.group(3)
			if "" in mevzuat_tarihi:
				mevzuat_tarihi = mevzuat_tarihi.replace(" ", "")
			if "." in mevzuat_tarihi:
				mevzuat_tarihi = mevzuat_tarihi.replace(".", "-")
			if "/" in mevzuat_tarihi:
				mevzuat_tarihi = mevzuat_tarihi.replace("/", "-")
			for key, value in ay_dict.items():
				if key in mevzuat_tarihi:
					mevzuat_tarihi = mevzuat_tarihi.replace(key, value)
			try:
				mevzuat_tarihi= pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				pass
			try:
				mevzuat_tarihi = pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				try:
					mevzuat_tarihi = pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
				except Exception:
					try:
						mevzuat_tarihi = pd.to_datetime("-".join([i.replace(" ", "") for i in mevzuat_tarihi.split("/")])).strftime("%Y-%m-%d")
					except Exception:
						pass	
		else:
			mevzuat_tarihi = np.NaN

		matching_madde_list=re.findall(r'([^\“\"\']Madde\s[0-9]+\s–)|([^\“\"\']Madde–\s[0-9]+\/[0-9])|([^\“\"\']Madde\s[0-9]+\/[a-z]\s.)|([^\“\"\']Madde\s[0-9]+\/[A-Z]\s.)|([^\“\"\']Ek\sMadde\s[0-9]+\s[^a-z^0-9^A-Z][^a-z])|([^\“\"\']Geçici\sMadde\s[0-9]+\s[^a-z][^0-9])|([^\“\"\']Ek\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\s[A-Z][a-z]{4}\s[^0-9][^a-z])|(Geçici\sEk\sMadde\s[0-9]\s[^0-9])|(Geçici\sMadde\s[0-9]+\([0-9]+\)\s[^0-9][^a-z])|(Geçici\sMadde\s[0-9]+-[0-9][^0-9][^A-Z][^a-z])|([^\“\"\']Madde\s[0-9]+[a-z]+\s[^0-9][^a-z^A-Z])|(Madde\s[0-9]+\([0-9]+\)\s[^0-9^A-Z^a-z])|(Madde+\s–\s[0-9]+\/[0-9])|(Madde\s[0-9]+-[0-9]+\s[^0-9][^A-Za-z])|([^\“\"\']Madde\s[0-9]+\/[ĞÇİ]\s[^0-9][^A-Z^a-z])|([^\“\"\']Madde\s[^0-9^A-Z^a-z])',text)
		madde_sayisi = len(matching_madde_list)
	except:
		rega_no, mukerrer_no, rega_tarihi, mevzuat_no, mevzuat_tarihi, madde_sayisi = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN

	return rega_no, mukerrer_no, rega_tarihi, mevzuat_no, mevzuat_tarihi, madde_sayisi

def extract_yonetmelik_info(text):
	
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " ").replace("*", " ").lstrip().rstrip())
	try:
		# extract rega no
		no=re.search(r'(Sayısı:\s|Numarası\s|Numarası\s:\s|Numarası:\s|Numarası\s:|Sayısı\s|Sayısı\s:\s[0-9]+\/[0-9]+\/[0-9]{4}\s[-]\s|Sayısı\s:|^Kararnamesinin\sSayısı\s:\s|Gazete\sTarihi\s:\s[0-9]+.[0-9]+.[0-9]+,\sNo\s:\s|Gazete\sTarihi\s:\s[0-9]+.[0-9]+.[0-9]+,\sNo:\s|Gazete\sTarihi\s:\s[0-9]+.[0-9]+.[0-9]+,\sNo\s:|Gazete\sTarihi\s:\s[0-9]+.[0-9]+.[0-9]+,\sNo\s|Sayı\s:\s|Sayı\s|Sayı:\s|Sayı\s:)([0-9]+)',text)
		if no:
			rega_no = no.group(2)
			if not rega_no:
				rega_no = re.search(r'\d+', no.group(0)).group(0)
		else:
			rega_no = np.NaN
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
		rega = re.search(r'(Tarihi+:\s|Tarihi+\s|Tarihi\s:\s|Tarihi\s:|Tarihi\s–\sSayısı\s:\s)([0-9]+.[0-9]+.[0-9]{4})',text)
		if rega:
			rega_tarihi = rega.group(2)
			if '' in rega_tarihi:
				rega_tarihi=rega_tarihi.replace(" ", "")
			if "." in rega_tarihi:
				rega_tarihi = rega_tarihi.replace(".", "-")
			if "/" in rega_tarihi:
				rega_tarihi = rega_tarihi.replace("/", "-")
			for key, value in ay_dict.items():
				if key in rega_tarihi:
					rega_tarihi = rega_tarihi.replace(key, value)
			try:
				rega_tarihi= pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				pass
			try:
				rega_tarihi = pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				try:
					rega_tarihi = pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
				except Exception:
					try:
						rega_tarihi = pd.to_datetime("-".join([i.replace(" ", "") for i in rega_tarihi.split("/")]), format="%d-%m-%Y").strftime("%Y-%m-%d")
					except Exception:
						pass
		else:
			rega_tarihi = np.NaN

		matching_madde_list=re.findall(r'([^\“\"\']Madde\s[0-9]+\s–)|([^\“\"\']Madde–\s[0-9]+\/[0-9])|([^\“\"\']Madde\s[0-9]+\/[a-z]\s.)|([^\“\"\']Madde\s[0-9]+\/[A-Z]\s.)|([^\“\"\']Ek\sMadde\s[0-9]+\s[^a-z^0-9^A-Z][^a-z])|([^\“\"\']Geçici\sMadde\s[0-9]+\s[^a-z][^0-9])|([^\“\"\']Ek\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\sMadde\s[^0-9]\s)|([^\“\"\']Geçici\sMadde\s[^0-9][^a-z])|(Geçici\sEk\sMadde\s[0-9]\s[^0-9])|(Geçici\sMadde\s[0-9]+\([0-9]+\)\s[^0-9][^a-z])|(Geçici\sMadde\s[0-9]+-[0-9][^0-9][^A-Z][^a-z])|([^\“\"\']Madde\s[0-9]+[a-z]+\s[^0-9][^a-z^A-Z])|(Madde\s[0-9]+\([0-9]+\)\s[^0-9^A-Z^a-z])|(Madde+\s–\s[0-9]+\/[0-9])|(Madde\s[0-9]+-[0-9]+\s[^0-9][^A-Za-z])|([^\“\"\']Madde\s[0-9]+\/[ĞÇİ]\s[^0-9][^A-Z^a-z])|([^\“\"\']Madde\s[–])|([^\“\"\']Madde\s[I]+\s[^0-9])',text)
		madde_sayisi = len(matching_madde_list)
	except:
		rega_no, mukerrer_no, rega_tarihi, madde_sayisi = np.NaN, np.NaN, np.NaN, np.NaN
	
	return rega_no, mukerrer_no, rega_tarihi, madde_sayisi

def extract_komisyon_raporu_info(text):
	#Raporlarında donem bilgisi tespiti beklenmemektedir.
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " ").replace("*", " ").lstrip().rstrip())
	try:
		##rega_no:Komisyon Raporunun Sıra Sayısı
		sira_no=re.search(r'(S.\sSayısı\s:.\s|S.\sSayısı\s:\s|S,\sSayısı:\s|S.\sSayisi:\s|S.\sSayısı\s|S.\sSayısı:\s|S\sA\sY\sI\sS\sI\s:\s|Sıra\sNo\s|S\s.\sSayısı\s:\s|SAYISI\s:\s|Sayısı:\s|sAYisı\s:|Sıra\sNQ\s|SAYISI\s|Sayw\s:\s|Sayısı\s:\s|Sıra\'N\?\s|Sıra\sN,\s|:Sayisi\s|Sayisi\s|Sayış[*]\s:\s|SAYMI\s|Sayısı[\"][0-9]+:\s|Say\si\ssi:\s|Say,,,:\s|SıraNo|Ordl\s|SlRA\sSAYISI:\s|SIRA\sSAYISI:\s|SiraSayisi:\s|Sayıs\sı\s:\s|Sayýsý\s:\s|SlRA\sS\sAYISI:\s)([0-9]+\s[a-z]+\sek|[0-9]+\s[a-z]+\s[|I]\s[a-z]+\sek|[0-9]+\sek\s[0-9]+|[0-9]+\s[0-9]+\s[0-9]+\s[\©\&]\s[0-9]+|[a-z][0-9]+\s[a-z]+\silâve|[0-9]+[\'][a-z]+\s[0-9]+\s[a-zi]+\sEk|[0-9]+\s[0-9]+\s[0-9]+\s©\s©\s[\^]|[0-9]+\s[a-z]+\silâve|[0-9]+\se\se\sk|[|]\s[0-9]+\s[0-9]+\sY@|[0-9]+\s[a-z]+\s[a-zşçiöü|]+\sek|[0-9][A-Z]+[a-z]+[0-9]+[a-z]+|[|][0-9]+\s[a-z]+\s[a-z]+\sek|[0-9]+\s©\s[0-9]+|[0-9]+\s[a-z]+\s[0-9]+\s[a-z]+\sek|[0-9]+\syeek|[|][0-9]+[|]\s[a-z]+\sek|[0-9]+\s[0-9]+\s[0-9]+\s[»]\sk|[0-9]+\s[0-9]+\s[0-9]+\s©|[0-9]+©[\\]\'\s[\^]K|[0-9]+Ve\s[0-9]+\s[a-z]+\sEk|[0-9]+\s[a-z]+\sk̂|[0-9]+\s[0-9]+\s[0-9]+\s[\^]\s[0-9]+\s[A-Z]+\s[A-Z]+\s[A-ZÜ]+\se\sk|[0-9]+\s[a-z]+\s©k|[\^][0-9]+\s[a-z]+\sek|[0-9]+[A-Z]+\s[a-z]+\sEk|[0-9]+\s[a-z]+\silave|[0-9]+\seek|[0-9]+ailâve|[0-9]+\s[a-z]+\s[|I]\s[a-z]+|[0-9]+\s[a-z]+\sİlâve|[0-9a-z]+\sek|[0-9]+\s©\s©[\^])',text)
		if sira_no:
			sira_no=sira_no.group(2)
		elif re.search(r'(Sayısı\s:|S.\sSayısı\s:.\s|S.\sSayısı\s:\s|S,\sSayısı:\s|S.\sSayisi:\s|S.\sSayısı\s|S.\sSayısı:\s|S\sA\sY\sI\sS\sI\s:\s|Sıra\sNo\s|S\s.\sSayısı\s:\s|SAYISI\s:\s|Sayısı:\s|sAYisı\s:|Sıra\sNQ\s|SAYISI\s|Sayw\s:\s|Sayısı\s:\s|Sıra\'N\?\s|Sıra\sN,\s|:Sayisi\s|Sayisi\s|Sayış[*]\s:\s|SAYMI\s|Sayısı[\"][0-9]+:\s|Say\si\ssi:\s|Say,,,:\s|SıraNo|Ordl\s|SlRA\sSAYISI:\s|SIRA\sSAYISI:\s|SiraSayisi:\s|Sayıs\sı\s:\s|Sayýsý\s:\s|SlRA\sS\sAYISI:\s)([0-9]+)',text):
			sira_no=re.search(r'(Sayısı\s:|S.\sSayısı\s:.\s|S.\sSayısı\s:\s|S,\sSayısı:\s|S.\sSayisi:\s|S.\sSayısı\s|S.\sSayısı:\s|S\sA\sY\sI\sS\sI\s:\s|Sıra\sNo\s|S\s.\sSayısı\s:\s|SAYISI\s:\s|Sayısı:\s|sAYisı\s:|Sıra\sNQ\s|SAYISI\s|Sayw\s:\s|Sayısı\s:\s|Sıra\'N\?\s|Sıra\sN,\s|:Sayisi\s|Sayisi\s|Sayış[*]\s:\s|SAYMI\s|Sayısı[\"][0-9]+:\s|Say\si\ssi:\s|Say,,,:\s|SıraNo|Ordl\s|SlRA\sSAYISI:\s|SIRA\sSAYISI:\s|SiraSayisi:\s|Sayıs\sı\s:\s|Sayýsý\s:\s|SlRA\sS\sAYISI:\s)([0-9]+)',text).group(2)
		if sira_no==None:
			sira_no="0"
		if '|' in sira_no:
			sira_no=sira_no.replace('|','1')  
		find = {'ye ek': 'ek 1', 'a ek': 'ek 1', '© 2': 'ek 2' ,'a ilâve': 'ek 1', 'a ilave': 'ek 1', 
				'e ek':'ek 1', '\'e 2 nci Ek':' ek 2','e 1 nci ek': 'ek 1','yek 1':'ek 1','\'e 1 inci Ek':' ek 1',
				'e ilâve':'ek 1','ya ikinci ek':'ek 2','a dördüncü ek':'ek 4','e e k':'ek 1','ye ikinci ek':'ek 2',
					'yeek':'ek 1','|9|':'90','\'ya 2 nci Ek':' ek 2','e I nci ek':'ek 1','e | ci ek':'ek 1','\'a 1 inci Ek':' ek 1',
				'ye I nci ek':'ek 1','e 1 nei ek':'ek 1','e 2 nci ek':'ek 2','Ve 1 inci Ek':' ek 1','\'ya 1 inci Ek':' ek 1',
				'S53':'153 ','ye k̂':'ek 1','a ©k':'ek 1','138 ve 139 ek 1':'139 ek 1','© 1':'ek 1','yek 2':'ek 2','eek':'ek 1',
				' vek 1':'ek 1','© © k':'ek 1','e İlâve':'ek 1','e üçüncü ek':'ek 3','a beşinci ek':'ek 5','e | nciek':'ek 1','ailâve':' ek 1',
				'\'ye 1 inci Ek':' ek 1','e | nci ek':'ek 1','yek':'ek','ya 2 nci ek':'ek 2','a | nci ek':'ek 1','e 3 nci ek':'ek 3','ya 1 nci ek':'ek 1','119a ek':'119 ek 1',
				'|47 ye ikinci ek':'147 ek 2','©':'ek','a 2 nci ek':'ek 2','5 6 0 & 2':'560 ek 2','f49 ek 1':'49 ek 1','3 5 3 ek ek ^':'353 ek 1','ek^':'1',
				'1 3 2 Y@':'132 ek 1','2IOa2nciek':'210 ek 2','1 7 0 » k':'170 ek 1','9 3 8 ek':'938 ek 1','3 6 9 ^ 4 H C Ü e k':'369 ek 4','^58 ek 1':'58 ek 1','VI inci Ek':' ek 1',
				'5 4 3 ek':'543 ek 1','ye 1nci ek':'ek 1','ye I nciek':'ek 1','2 0 3 ek':'203 ek 1','e 1 nciek':'ek 1','a 1 nci ek':'ek 1','e 1 ci ek':'ek 1','4 4 5':'445','5 6 3':'563','7 1':'71'}
		for key, value in find.items():
			sira_no = sira_no.replace(key, value)
		space=re.match(r'(([0-9]+)ek)',sira_no)
		if space:
			sira_no=sira_no.replace(space.group(2),space.group(2)+' ')

		##donem(string): Komisyon Raporunun ait olduğu Dönemi/21. Dönem öncesi Komisyon
		##Raporlarında donem bilgisi tespiti beklenmemektedir.
		donem=re.search(r'((Dönem:\s|Dönem\s:\s|YASAMA\sDONEMİ\s|YASAMA\sDÖNEMİ\s|YASAMA\sDÖNEMİ\sYASAMA\sYILI\s|yasama\sdönemi\s|Döneni\s:\s|Döntm\s:\s|önem:\s|Dönem\s)([0-9]+|[A-Zİ][0-9]|[a-zı])\s)',text)
		if donem:
			donem=donem.group(3)
			find = {'ı':'1','i':'1','İ':'1'}
			for key, value in find.items():
				donem = donem.replace(key, value)
			donem=str(donem)+"".join(". Dönem")
		else: 
			donem = np.NaN
   
	except:
		sira_no = np.NaN
		donem = np.NaN
  
	return sira_no, donem

def extract_genelge_info(text):
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " ").replace("*", " ").lstrip().rstrip())
	try:
		mevzuat_tarihi=re.search(r'(Tarih\s([0-9]+\/[0-9]+\/[0-9]{4})\s)|(3\s0\sMaYIS\s2ÖÎ2)|[.]\s([0-9]+\s[A-Za-zıiüçşÜIÇŞ]+\s[0-9]{4})\s|\s([0-9]+\/[0-9]+\/[0-9]{4})\s[A-Za-zçı]+\s[A-Za-z]+\sGENELGE|([0-9]+\/\s[0-9]+\/\s[0-9]{4})\sGENELGE|([0-9]+[.][0-9]+[.][0-9]{4})\s|([0-9]+\s[A-Za-zıiüçşŞÜIÇŞ]+\s[0-9]+\s)[0-9]+.+\sGENELGE|\s([0-9]+\/[0-9]+\/[0-9]{4})\sKonu|Konu:\s([0-9]+.[0-9]+.[0-9]{4})\sGENELGE|([0-9]+\/[A-Za-zğüişç]+\/[0-9]{4})\sKonu|([0-9]+\/[0-9]+\/[0-9]{4})\sGENELGE|([0-9]+\/[0-9]+\/[0-9]{4})\sKONU:|([0-9]+\/\s[0-9]+\/\s[0-9]{4})\sKonu|([0-9]+\s\/[0-9]+\/[0-9]+)\sKonu|([0-9]+\/[0-9]+\/[0-9]+)\starihli|([0-9]+\s[A-ZŞĞIiiüşçÇ]+\s[0-9]{4}\s)GENELGE|([0-9]+\s[A-Za-zŞĞIiiüşçÇ]+\s[0-9]+)\sKonu',text)
		if mevzuat_tarihi:
			mevzuat_tarihi=mevzuat_tarihi.group(2) or mevzuat_tarihi.group(3) or mevzuat_tarihi.group(4) or mevzuat_tarihi.group(5) or mevzuat_tarihi.group(6) or mevzuat_tarihi.group(7) or mevzuat_tarihi.group(8) or mevzuat_tarihi.group(9) or mevzuat_tarihi.group(10) or mevzuat_tarihi.group(11) or mevzuat_tarihi.group(12) or mevzuat_tarihi.group(13) or mevzuat_tarihi.group(14) or mevzuat_tarihi.group(15) or mevzuat_tarihi.group(16) or mevzuat_tarihi.group(17) or mevzuat_tarihi.group(18)   
			if '' in mevzuat_tarihi:
				mevzuat_tarihi=mevzuat_tarihi.replace(" ","")
			if 'MaYIS2ÖÎ2' in mevzuat_tarihi:
				mevzuat_tarihi=mevzuat_tarihi.replace("MaYIS2ÖÎ2","Mayıs2012")
		
			for key, value in ay_dict.items():
				if key in mevzuat_tarihi:
					mevzuat_tarihi = mevzuat_tarihi.replace(key, value)
			if '.' in mevzuat_tarihi:
				mevzuat_tarihi=mevzuat_tarihi.replace(".","-")    
			if '//' in mevzuat_tarihi:
				mevzuat_tarihi=mevzuat_tarihi.replace("//","-") 
			if "/" in mevzuat_tarihi:
				mevzuat_tarihi = mevzuat_tarihi.replace("/", "-")
			for key, value in ay_dict.items():
				if key in mevzuat_tarihi:
					mevzuat_tarihi = mevzuat_tarihi.replace(key, value)
			try:
				mevzuat_tarihi= pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				pass
			try:
				mevzuat_tarihi = pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				try:
					mevzuat_tarihi = pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
				except Exception:
					try:
						mevzuat_tarihi = pd.to_datetime("-".join([i.replace(" ", "") for i in mevzuat_tarihi.split("/")]), format="%d-%m-%Y").strftime("%Y-%m-%d")
					except Exception:
						pass
		else:
			mevzuat_tarihi = np.NaN

		#mevzuat_no(string): Genelge Numarası (boş değilse)
		mevzuat_no=re.search(r'(GENELGE\s[0-9]+\s|GENELGE\sNO:\s[0-9]+\/[0-9]+\s|GENELGE\sNO\s[0-9]+\/[0-9]+\s|Genelge\sNo:\s[0-9]+\/[0-9]+\s|GENELGE\sNO-\s[0-9]+\/[0-9]+\s|Genelge:\s[0-9]+\/[0-9]+\s|Genelge\sSayısı\s[0-9]+\/[0-9]+\s|Genelge\sNo\s:\s[0-9]+\/[0-9]+\s)([0-9]+\/[0-9]+|[0-9]{4})|(SIRA\sNO:\s|GENELGESİ\sSERİ\sNO\s:\s|GENELGE\s|SIRA\sNO\s:\s|SERİ\sNO:|SERİ\sNO\s|GENELGESİ\sSERİ\sNO:\s|SIRA\sNO:|GENELGE\sSERİ\sNO\s:\s|GENELESİ\sSERİ\sNO:\s|G\sE\sN\sE\sL\sG\sE\s|GENELGE\sNo:\s|Genelge\sNo:\s|Genelge\sNo:|GENELGE\s[•]\s|GENELGE\sNO:\s|Genelge\sNo\s:\s|Genelge\sSayısı\s|Genelge:\s|GENELGE\sNO-\s|GENELGE\sNO\s|GENELGE\sNO:|GENELGE\sNO\s:\s|Genelge\sNo\s:)([0-9]+-\sH|[0-9]+\s\/\s[0-9a-z]+|[0-9]+\s\/\s[0-9]+|[0-9]+\/[0-9]+|[0-9]+-[0-9]+|[0-9]+\s–\s[0-9]+|[0-9]+\/\s[0-9]+|[0-9]+–[0-9]+|[0-9]+\s-\s[0-9]+|[0-9]+\s-[0-9]+|[0-9]+\s\/|[0-9]+[A-Z]\/\/\/[0-9]+|[0-9]{5}[.][0-9]+|[0-9]+\s!.l|[0-9]+\/\s[A-Z]+\'|[0-9]+\/\s[\/]|[0-9]+[A-Z][0-9]|[0-9]+-\s[0-9]+|[0-9]+\/[\^][0-9]+|[•]\s[0-9]+\/\s£,2|[0-9]+-\s[A-Z][.]|[0-9]+\/\s[\^]|[0-9]+-\s|[0-9]+\s)',text)
		if mevzuat_no: 
			mevzuat_no=mevzuat_no.group(2) or mevzuat_no.group(4) 

			if '' in mevzuat_no:
				mevzuat_no=mevzuat_no.replace(" ",'')

			find = {'2015- ':'2015/11','2015/^':'2015/9','2015-A.':'2015-01','2014/^0':'2014/30','•2014/£,2':'2014/22','2014//':'2014/11','H':'11',
				'2013/JS':'2013/35','2012!.l':'2012/26','3D///30':'2011/30','2010/ ':'2010/17','2017U2':'2017/12','1998/47504':'1998/5','2015/2j3':'2015/28','20127.15':'2017.15'}
			for key, value in find.items():
				for key, value in find.items():
					if key in mevzuat_no:
						mevzuat_no = mevzuat_no.replace(key, value)
		
			if '.' in mevzuat_no:
				mevzuat_no=mevzuat_no.replace(".",'/')
			if '–' in mevzuat_no:
				mevzuat_no=mevzuat_no.replace("–",'/')
			if '-'in mevzuat_no:
				mevzuat_no=mevzuat_no.replace("-",'/')
   
		elif re.search(r'([0-9]+\/[0-9]+|[0-9]+)(\sSayılı"|\sSayılı\s.*Konulu\sGenelge|\sSAYILI\sGENELGE)',text):
			mevzuat_no=re.search(r'([0-9]+\/[0-9]+|[0-9]+)(\sSayılı"|\sSayılı\s.*Konulu\sGenelge|\sSAYILI\sGENELGE)',text).group(1)
			if '' in mevzuat_no:
				mevzuat_no=mevzuat_no.replace(" ",'')
			
			find = {'2015- ':'2015/11','2015/^':'2015/9','2015-A.':'2015-01','2014/^0':'2014/30','•2014/£,2':'2014/22','2014//':'2014/11','H':'11',
				'2013/JS':'2013/35','2012!.l':'2012/26','3D///30':'2011/30','2010/ ':'2010/17','2017U2':'2017/12','1998/47504':'1998/5','2015/2j3':'2015/28','20127.15':'2017.15'}
			for key, value in find.items():
				for key, value in find.items():
					if key in mevzuat_no:
						mevzuat_no = mevzuat_no.replace(key, value)
		
			if '.' in mevzuat_no:
				mevzuat_no=mevzuat_no.replace(".",'/')
			if '–' in mevzuat_no:
				mevzuat_no=mevzuat_no.replace("–",'/')
			if '-'in mevzuat_no:
				mevzuat_no=mevzuat_no.replace("-",'/')
    
		else:
			mevzuat_no = np.NaN
	
		belge_sayi=re.search(r'(Sayı\s:\s|SAYI\s:\s|Sayı\s:\s|SAYI\s:|SAYI:\s|Sayı\s|SAYI:|Sayı\s|Sayı\s:|Sayı:|Sayı:\s|Sayı\s:|Sayi\s:\s|SAYI:\s|Sayı\sU\s)((B.[0-9]+[0-9]+.[0-9]+.[A-Zİ]+.[0-9]+.|VUK|B.[0-9]+.[0-9]+.[0-9]+.[A-Zİ]+.[0-9]+.|B.\s[0-9]+.[0-9]+.[A-Z]+.[0-9]+.|B.[0-9]+.[A-Z]+.|VRS\/|GEL:|VRS:|Bl\s[A-ZŞ]+-[0-9]+-[A-Z]+-[A-Z]+\s|BİAŞ-[0-9]+-[A-Z]+-[0-9]+.|BÎAŞ-[0-9]+-[A-Z]+-[0-9]+.|B[0-9]+.[0-9]+.|B.[0-9]+.[a-zA-Z]+.|B.\s[0-9]+.\s[0-9]+.\s|[0-9]+-|[0-9]+|E-[0-9]+-)([0-9]+.[0-9]+.[0-9]+.[0-9]+\/\s[0-9]+|[0-9]+.[0-9]+.[0-9]+\/\s[0-9]+|[0-9]+.[0-9]+.[0-9]+-[A-Z].[0-9]+|\/[0-9]+-[0-9]+\/[0-9]+|[0-9]+.[0-9]+.[0-9]+-[A-Za-z]+.\s[0-9]+|\/\s[0-9]+|[A-ZİÇŞÜa-z0-9]+-\/[0-9]+-|[0-9]+.[0-9]+\s-[0-9]+|[0-9]+.[0-9]+-\s[0-9]+|[0-9]+.[0-9]+.[0-9]+-[0-9]+|[0-9]+-[0-9]+-[0-9]+\/[0-9]+|[0-9]+.[0-9]+\/[0-9]+-\s[0-9]+\/[0-9]+|[0-9]+\/[0-9]+-[0-9]+-[0-9]+|[0-9]+.[0-9]+-[0-9]+\[[0-9]+-\s[0-9]+\]-[0-9]+|[0-9]+\/[0-9]+-[0-9]+\s\/[0-9]+|[0-9]+-[0-9]+-[0-9]+\/[0-9]+|[0-9]+\/[0-9]+-\s[0-9]+\/[0-9]+|[0-9]+\/[0-9]+-[0-9]+\s\/\s[0-9]+|[0-9]+\/[0-9]+-[0-9]+|[.][0-9]+\/[0-9]+-[0-9]+\/[0-9]+|[0-9]+.[0-9]+-[0-9]+-[0-9]+.[0-9]+-[0-9]+|[0-9]+.[0-9]+\/[0-9]+-[0-9]+\/[0-9]+-[0-9]+|[0-9]+-[0-9]+.[0-9]+\/[0-9]+|[0-9]+\/[0-9]+.[0-9]+\/[0-9]+-[0-9]+|[0-9]+.[0-9]+\/[0-9]+-[0-9]+-[0-9]+|[0-9]+.[0-9]+\/[0-9]+-[0-9]+|[0-9]+.[0-9]+.[0-9]+.[0-9]+.[0-9]+\[[0-9]+\]\/[0-9]+|\s[0-9]+\/[0-9]+-[0-9]+\/[0-9]+|[0-9]+.[0-9]+.[0-9]+\/[0-9]+|[0-9]+.[0-9]+.[0-9]+\/[A-Z]+-[0-9]+-[0-9]+|[0-9]+.[0-9]+-[0-9]+\[[0-9]+-[0-9]+\]|[0-9]+.[0-9]+\/[0-9]+-|[0-9]+\/[0-9]+-[A-Za-z]+.[A-Za-z]+.[0-9]+-[0-9]+|[0-9]+\/[0-9]+-|[0-9]+.[0-9]+-[0-9]+.[0-9]+.[0-9]+|[0-9]+\/[0-9]+|[0-9]+.[0-9]+.[0-9]+-|[0-9]+\s[0-9]+.[0-9]+.[0-9]+\s-[0-9]+|[A-ZİÇŞÜ0-9]+-[0-9]+\/\s[0-9]+-[0-9]+-\([0-9]+\)|[0-9]+.[0-9]+.[0-9]+|.[0-9]+.[0-9]+.[0-9]+\/[0-9]+)|[0-9]+\/[0-9]+\/[0-9]+\/[0-9]+|[0-9]+[.][0-9][A-Za-z][A-Z][.][A-Z]+[0-9]+[.][A-Z0-9]+\/[a-z]+|[0-9]+.[0-9]+.[0-9]+\/[0-9]+)|([0-9]+-[0-9]+-[0-9]+\/[0-9]+\/[0-9]+\s)(Sayılı\s.+Genelgesi)',text)
		if belge_sayi:
			belge_sayi=belge_sayi.group(2) or belge_sayi.group(5)
		else: belge_sayi = np.NaN

	except:
		mevzuat_tarihi = np.NaN
		mevzuat_no = np.NaN
		belge_sayi = np.NaN
	
	return mevzuat_tarihi, mevzuat_no, belge_sayi

def extract_ozelge_info(text):
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " ").replace("*", " ").lstrip().rstrip())
	try:
		##mevzuat_tarihi(date:yyyy-mm-dd): Özelgenin tarihi
		mevzuat=re.search(r'([0-9]+\/[0-9]+\/[0-9]{4}|[0-9]+[.][0-9]+[.][0-9]{4})(\sKonu|Sayı|[*][0-9]+\sKONU|[*][0-9]+\sKonu|\sSAYI\s:|-[0-9]+\sKonu|\sSayı\s:|\s[……]|\/[0-9]+\sKONU|-[0-9]+\s[……]|-[0-9]+\s|\/\s[0-9]+\sSAYI|\sKONU:\s|\sSn:\s|[*]\s[0-9]+|[*][0-9]+|\sSayın)',text)
		if mevzuat:	
			mevzuat_tarihi = mevzuat.group(1)
			if '' in mevzuat_tarihi:
				mevzuat_tarihi=mevzuat_tarihi.replace(" ", "")
			if "." in mevzuat_tarihi:
				mevzuat_tarihi = mevzuat_tarihi.replace(".", "-")
			if "/" in mevzuat_tarihi:
				mevzuat_tarihi = mevzuat_tarihi.replace("/", "-")
			for key, value in ay_dict.items():
				if key in mevzuat_tarihi:
					mevzuat_tarihi = mevzuat_tarihi.replace(key, value)
			try:
				mevzuat_tarihi= pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				pass
			try:
				mevzuat_tarihi = pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				try:
					mevzuat_tarihi = pd.to_datetime(mevzuat_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
				except Exception:
					try:
						mevzuat_tarihi = pd.to_datetime("-".join([i.replace(" ", "") for i in mevzuat_tarihi.split("/")]), format="%d-%m-%Y").strftime("%Y-%m-%d")
					except Exception:
						pass
		else:
			mevzuat_tarihi = np.NaN
	except:
		mevzuat_tarihi = np.NaN

	return mevzuat_tarihi

def extract_teblig_info(text):
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " ").replace("*", " ").lstrip().rstrip())
	try:
		#rega_no(int): Tebliğ'nin yayınlanmış olduğu Resmi Gazete Numarası
		no=re.search(r'(Sayısı:\s|Numarası\s|Numarası\s:\s|Numarası:\s|Numarası\s:|Sayısı\s|Sayısı\s:\s[0-9]+\/[0-9]+\/[0-9]{4}\s[-]\s|Sayısı\s:|^Kararnamesinin\sSayısı\s:\s|Gazete\sTarihi\s:\s[0-9]+.[0-9]+.[0-9]+,\sNo\s:\s|Gazete\sTarihi\s:\s[0-9]+.[0-9]+.[0-9]+,\sNo:\s|Gazete\sTarihi\s:\s[0-9]+.[0-9]+.[0-9]+,\sNo\s:|Gazete\sTarihi\s:\s[0-9]+.[0-9]+.[0-9]+,\sNo\s|Sayı\s:\s|Sayı\s|Sayı:\s|Sayı\s:)([0-9]+)',text)
		if no:
			rega_no = no.group(2)
			if not rega_no:
				rega_no = re.search(r'\d+', no.group(0)).group(0)
		else:
			rega_no = np.NaN

		#mukerrer_no:mukerrer_no(int): İçeriğin Resmi Gazete Mükerrer Sayısı (0 ise mükerrer olmadığını)
		find = re.findall(r'([0-9]+\s(Mükerrer)\sKarar|[0-9]+\s(Mükerrer)\sKanun|Sayısı:\s[0-9]+\s(Mükerrer)\s[A-Z])',text)
		count_mukerrer = re.search(r'((\s[0-9]).\sMükerrer)(\s[A-ZÇŞÜİa-z]+|\s[0-9]+\s[A-ZŞÇÜİ])',text)
		if find:
			mukerrer_no=1
		elif count_mukerrer==None :
			mukerrer_no=0
		elif count_mukerrer:
			mukerrer_no=int(count_mukerrer.group(2))
		else:
			mukerrer_no=0

		#rega_tarihi(date:yyyy-mm-dd): İçeriğin yayınlandığı Resmi Gazete Tarihi
		rega=re.search(r'(Tarihi+:\s|Tarihi+\s|Tarihi\s:\s|Tarihi\s:|Tarihi\s–\sSayısı\s:\s)([0-9]+.[0-9]+.[0-9]{4})|([0-9]+\s[A-Za-zÇŞÜİüiç]+\s[0-9]{4})(\s–\sSayı\s:\s)',text)
		if rega:
			rega_tarihi=rega.group(2) or rega.group(3) 
   
			for key, value in ay_dict.items():
				rega_tarihi = rega_tarihi.replace(key, value)
			if '' in rega_tarihi:
				rega_tarihi=rega_tarihi.replace(" ", "")
			if "." in rega_tarihi:
				rega_tarihi = rega_tarihi.replace(".", "-")
			if "/" in rega_tarihi:
				rega_tarihi = rega_tarihi.replace("/", "-")
			for key, value in ay_dict.items():
				if key in rega_tarihi:
					rega_tarihi = rega_tarihi.replace(key, value)
			try:
				rega_tarihi= pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				pass
			try:
				rega_tarihi = pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				try:
					rega_tarihi = pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
				except Exception:
					try:
						rega_tarihi = pd.to_datetime("-".join([i.replace(" ", "") for i in rega_tarihi.split("/")]), format="%d-%m-%Y").strftime("%Y-%m-%d")
					except Exception:
						pass
		else:
			rega_tarihi = np.NaN
	except:
		rega_no, mukerrer_no, rega_tarihi = np.NaN, np.NaN, np.NaN

	return rega_no, mukerrer_no, rega_tarihi

def extract_resmi_gazete_info(text):
	text = re.sub(" +", " ", text.replace("\r", " ").replace("\n", " ").replace("*", " ").lstrip().rstrip())
	try:
		#rega_no:Resmi Gazete'nin Sayısı
		no=re.search(r'((S\sa\sy\s[İI]\s:\s([0-9]\s[0-9]\s[0-9]\s[0-9]))|Sayı:\s|Sayı\s:\s|SAYI:\s|Say\sı\s:\s|SAYI\s:\s|SAYİ:\s|Sayı:\s,|Sayı:)([0-9]\s.{7}|[0-9]+)|(([0-9]+)\sSayılı\sResmî\sGazete)|(S\sA\sY\s[İI]\s:\s([0-9]\s[0-9]\s[0-9]\s[0-9]))|(S\sA\sY\s[İI]\s:\s([0-9]+))|(S\sa\sy\sı\s:\s([0-9]\s[0-9]\s[0-9]\s[0-9]\s[0-9]+))|(S\sa\sy\sı\s:\s([0-9]+))',text)
		if no:
			rega_no=no.group(2) or no.group(4) or no.group(6) or no.group(8) or no.group(10) or no.group(12)
			if not rega_no:
				rega_no = re.search(r'\d+', no.group(0)).group(0)
		else:
			rega_no = np.NaN  
		if '' in rega_no:
			rega_no=rega_no.replace(" ","")
		if '1101' in rega_no:
			rega_no=rega_no.replace("1101","11010")

		#mukerrer_no:Resmi Gazete'nin ilgili sayısı için kaçıncı Mükerrer olduğu (0 ise mükerrer olmadığını belirtir)
		find = re.findall(r'(:\s[0-9]+\s(Mükerrer)\s[A-Z])',text)
		count_mükerrer = re.search(r'((\s[0-9]).\sMükerrer\s[A-Za-z]+)',text)
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
			for key, value in ay_dict.items():
				rega_tarihi = rega_tarihi.replace(key, value)
			if '' in rega_tarihi:
				rega_tarihi=rega_tarihi.replace(" ", "")
			if "." in rega_tarihi:
				rega_tarihi = rega_tarihi.replace(".", "-")
			if "/" in rega_tarihi:
				rega_tarihi = rega_tarihi.replace("/", "-")
			for key, value in ay_dict.items():
				if key in rega_tarihi:
					rega_tarihi = rega_tarihi.replace(key, value)
			try:
				rega_tarihi= pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				pass
			try:
				rega_tarihi = pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
			except Exception:
				try:
					rega_tarihi = pd.to_datetime(rega_tarihi, format="%d-%m-%Y").strftime("%Y-%m-%d")
				except Exception:
					try:
						rega_tarihi = pd.to_datetime("-".join([i.replace(" ", "") for i in rega_tarihi.split("/")]), format="%d-%m-%Y").strftime("%Y-%m-%d")
					except Exception:
						pass
		else:
			rega_tarihi = np.NaN

	except:
		rega_no, mukerrer_no, rega_tarihi = np.NaN, np.NaN, np.NaN

	return rega_no, mukerrer_no, rega_tarihi