import pandas as pd
import re
from soynlp.normalizer import *
from hanspell import spell_checker
from PyKomoran import *
import os
import unicodedata
import kss
from konlpy.tag import  Okt, Mecab
import time
from pykospacing import Spacing
mecab = Mecab()
okt = Okt()
spacing = Spacing()

stopwords_path = '/home/ubuntu/prep/stopwords.tsv'
pd_stopwords = pd.read_csv(stopwords_path)
stopwords = []

for i in range(len(pd_stopwords)):
  stopwords.append(pd_stopwords.iloc[i][0])

# 숫자, 영어 등 필요하지 않은 문자 및 이모티콘 제거
def clean(text):
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', } 
    
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text.strip()

# E-mail, URL, 한글 자모, 태그 및 특수기호 제거
def clean_str(text):
    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)' # E-mail제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '<[^>]*>'         # HTML 태그 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '[^\w\s\n]'         # 특수기호제거
    text = re.sub(pattern=pattern, repl='', string=text)
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', string=text)
    text = re.sub('\n', '.', string=text)
    return text 

def tokenize(userInput):
    # 1. 학습에 불필요한 문자 제거
  clean_text = clean_str(clean(userInput))

  # 2-1. 띄어쓰기 구분
  # 2-2. 문장 종결 및 분리
  divided_sentences_text = kss.split_sentences(clean_text)
  words_list = [];
  long_sentence = " ".join(divided_sentences_text)
  
  # 2-3. 문법 체크
  if(len(long_sentence) != 0):
      spaced = spacing(long_sentence)
      norm_spaced = okt.normalize(spaced)
      spell_checked_text = norm_spaced.split(" ")
      for key in spell_checked_text:
          normalized_word = repeat_normalize(key, num_repeats=2)
          words_list.append(normalized_word)
  
  # # 4~5. nouns 추출
  usersCleanedTextBeforeStopwordsCheck = mecab.nouns(" ".join(words_list))
  
  # 6. 불용어 처리
  cleaned_words = []
  for this_word in usersCleanedTextBeforeStopwordsCheck:
      if this_word not in stopwords:
          cleaned_words.append(this_word)

  usersCleanedText = " ".join(cleaned_words)
  return usersCleanedText

from tqdm import tqdm

path = '/home/ubuntu/prep/hidoc_data'

file_names = os.listdir(path)

for file_name in file_names:
  whole_data = []
  file_path = os.path.join(path, file_name)
  val = pd.read_csv(file_path)
  division = file_name.split('.')[0]
  division = unicodedata.normalize('NFC', division)
  
  if(division == "내과"):
    num_label = '0'
  elif(division == "외과"):
    num_label = '1'
  elif(division == "비뇨기과"):
    num_label = '2'
  elif(division == "산부인과"):
    num_label = '3'
  elif(division == "성형외과"):
    num_label = '4'
  elif(division == "소아과"):
    num_label = '5'
  elif(division == "신경과"):
    num_label = '6'
  elif(division == "안과"):
    num_label = '7'
  elif(division == "이비인후과"):
    num_label = '8'
  elif(division == "재활의학과"):
    num_label = '9'
  elif(division == "정신건강의학과"):
    num_label = '10'
  elif(division == "정형외과"):
    num_label = '11'
  elif(division == "치과"):
    num_label = '12'
  elif(division == "피부과"):
    num_label = '13'
  elif(division == "약국"):
    num_label = '14'
  elif(division == "한방과"):
    num_label = '15'
  elif(division == "응급실"):
    num_label = '16'

  for i in tqdm(range(len(val)), desc=division):
    if(type(val.iloc[i]["0"]) == str):
        docu = tokenize(val.iloc[i]["0"])
    else:
        continue
    whole_data.append([docu, num_label])
    
  whole_df = pd.DataFrame(whole_data)
  whole_df.to_csv('/home/ubuntu/prep/'+ division + '_input.csv', index=False, header=None)
