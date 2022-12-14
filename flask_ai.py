from flask import Flask, request
import re

# torch
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np

#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#tokenize
import pandas as pd
import re
from soynlp.normalizer import *
from hanspell import spell_checker
from PyKomoran import *
import kss
from konlpy.tag import  Okt, Mecab
import json

mecab = Mecab()
okt = Okt()

#gpu 사용
device = torch.device("cuda:0")

#BERT 모델, Vocabulary 불러오기 
bertmodel, vocab = get_pytorch_kobert_model()

# KoBERT에 입력될 데이터셋 정리
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

# 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=14,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

# Setting parameters
max_len = 128
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

## 학습 모델 로드
PATH = './models/'
model = torch.load(PATH + 'KoBERT_MLC.pt', map_location=device)
model.load_state_dict(torch.load(PATH + 'model_state_dict.pt', map_location=device))

#토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

def new_softmax(a) :
    c = np.max(a) # 최댓값
    exp_a = np.exp(a-c) # 각각의 원소에 최댓값을 뺀 값에 exp를 취한다. (이를 통해 overflow 방지)
    sum_exp_a = np.sum(exp_a)
    y = (exp_a / sum_exp_a) * 100
    return np.round(y, 3)


# 예측 모델 설정
def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()
            min_v = min(logits)
            total = 0
            probability = []
            logits = np.round(new_softmax(logits), 3).tolist()
            for logit in logits:
                probability.append(np.round(logit, 3))

            if np.argmax(logits) == 0:  division = "내과"
            elif np.argmax(logits) == 1: division = "외과"
            elif np.argmax(logits) == 2: division = '비뇨의학과'
            elif np.argmax(logits) == 3: division = '산부인과'
            elif np.argmax(logits) == 4: division = '성형외과'
            elif np.argmax(logits) == 5: division = '소아청소년과'
            elif np.argmax(logits) == 6: division = '신경과'
            elif np.argmax(logits) == 7: division = '안과'
            elif np.argmax(logits) == 8: division = '이비인후과'
            elif np.argmax(logits) == 9: division = '재활의학과'
            elif np.argmax(logits) == 10: division = '정신과'
            elif np.argmax(logits) == 11: division = '정형외과'
            elif np.argmax(logits) == 12: division = '치과'
            elif np.argmax(logits) == 13: division = '피부과'

    return [np.argmax(logits), probability.index(sorted(probability, reverse=True)[1])]

# stopwords 불러오기
stopwords = [
    "이",
    "안",
    "있",
    "하",
    "것",
    "들",
    "그",
    "되",
    "수",
    "이",
    "보",
    "나",
    "사람",
    "주",
    "등",
    "같",
    "우리",
    "년",
    "가",
    "한",
    "지",
    "대하",
    "오",
    "그렇",
    "위하",
    "때문",
    "그것",
    "두",
    "말하",
    "알",
    "그러나",
    "받",
    "못하",
    "일",
    "그런",
    "또",
    "문제",
    "더",
    "사회",
    "많",
    "그리고",
    "좋",
    "따르",
    "중",
    "나오",
    "가지",
    "씨",
    "시키",
    "만들",
    "지금",
    "생각하",
    "그러",
    "속",
    "하나",
    "집",
    "살",
    "모르",
    "적",
    "월",
    "데",
    "자신",
    "안",
    "어떤",
    "내",
    "경우",
    "명",
    "생각",
    "시간",
    "그녀",
    "다시",
    "이런",
    "앞",
    "보이",
    "번",
    "나",
    "다른",
    "어떻",
    "개",
    "전",
    "들",
    "사실",
    "이렇",
    "점",
    "싶",
    "말",
    "정도",
    "좀",
    "원",
    "잘",
    "통하",
    "안",
    "제가",
    "게",
    "한",
    "때",
    "거",
    "수",
    "좀",
    "너무",
    "있는",
    "것",
    "정도",
    "하고",
    "같은",
    "그리고",
    "후",
    "많이",
    "..",
    "잘",
    "더",
    "계속",
    "해야",
    "건가요",
    "이런",
    "왜",
    "그냥",
    "어떻게",
    "그",
    "이",
    "있나요",
    "근데",
    "통증이",
    "하는데",
    "할",
    "다시",
    "있습니다",
    "건",
    "했는데",
    "하는",
    "그런",
    "갑자기",
    "혹시",
    "조금",
    "지금",
    "있는데",
    "수술",
    "어떤",
];

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
      spell_checked_text = spell_checker.check(long_sentence)
      if type(spell_checked_text.words) != list:
        for key, _ in spell_checked_text.words.items():
            # 2-4. 2글자 초과하여 반복되는 문자를 축약 후 단어 리스트에 저장
           normalized_word = repeat_normalize(key, num_repeats=2)
           words_list.append(normalized_word)
      else:
        for key in spell_checked_text.words:
          normalized_word = repeat_normalize(key, num_repeats=2)
          words_list.append(normalized_word)
  
  # # 4~5. nouns 추출
  usersCleanedTextBeforeStopwordsCheck = (words_list)

  extract_nouns = mecab.nouns(" ".join(words_list))
  extract_nouns = " ".join(extract_nouns)
  if((type(extract_nouns) is not float and len(extract_nouns) != 0)):
        extract_nouns = re.sub("대한의사협회|상 담 사 세|감 피 피 너 뮤|곳|의 미|박헌구|지식인|뭔가|게시물|진단 치료 를|진료|모스 코피|를|이광준|오늘|평 식|장 현채|권 순모|중간|다음 날|최근|노 무사|노무사|교통사고|변호사|노동청|민원|조언|상담 사 세|내공|의 심|대부분|쾌차|이게|움|가이드라인|도록|하루|네이버|김상범|이용|법|상담|달|김석준|사진|대부분|안녕|버|이미지|이양구|전문의|피부과전문의|이광준|권순모|박현구|김봉수|장현채|한경호|닥톡|신홍범|전평식|김태만|이세라|김선영|구오섭|박종원|최연철|한형일|이형근|이재성|변상권|이정찬|중요|참고|감사|네이버|지식|상담 사|전문의|진단|문의|검진|정확|여학생|남학생|뭔가|변상|댓글 성 심것 껏|답변|외|하루|료|화질 선택 옵션 자동|정찬|검|대처|추천|립니|입 니|치료 결정|연철|유감|사장|주방|상담 사 형|이것|입니다|구 섭|촛|이상|최소|최대|곳|상태|느낌|영향|답변|감사|도움|값|참고|중요", "", extract_nouns)
        extract_nouns = re.sub("시 력", "시력", extract_nouns)
        extract_nouns = re.sub("마사 지기", "마사지기", extract_nouns)
        extract_nouns = re.sub("광수 용체", "광수용체", extract_nouns)
        extract_nouns = re.sub("망막 색 소변 증", "망막색소변성증", extract_nouns)
        extract_nouns = re.sub("백 내장", "백내장", extract_nouns)
        extract_nouns = re.sub("강화 술", "강화술", extract_nouns)
        extract_nouns = re.sub("목소 리", "목소리", extract_nouns)
        extract_nouns = re.sub("슬개건 염", "슬개건염", extract_nouns)
        extract_nouns = re.sub("체 감량", "체중감량", extract_nouns)
        extract_nouns = re.sub("표 피낭", "표피낭", extract_nouns)
        extract_nouns = re.sub("병 리", "병리", extract_nouns)
        extract_nouns = re.sub("포 러스 연고", "포러스 연고", extract_nouns)
        extract_nouns = re.sub("덱 스핀 정", "덱스핀정", extract_nouns)
        extract_nouns = re.sub("조 증 장애", "조증장애", extract_nouns)
        extract_nouns = re.sub("근육 막이", "근육막", extract_nouns)
        extract_nouns = re.sub("임 플란트", "임플란트", extract_nouns)
        extract_nouns = re.sub("류 마티스", "류마티스", extract_nouns)
        extract_nouns = re.sub("폭 센", "폭센", extract_nouns)
        extract_nouns = re.sub("발 기 부전", "발기부전", extract_nouns)
        extract_nouns = re.sub("립 선", "전립선", extract_nouns)
        extract_nouns = re.sub("조 스프레이", "조루스프레이", extract_nouns)
        extract_nouns = re.sub("마 사지|맛 사지", "마사지", extract_nouns)
        extract_nouns = extract_nouns.split(" ")
  # 6. 불용어 처리
  cleaned_words = []
  nouns_cleaned_words = []

  return [usersCleanedTextBeforeStopwordsCheck, extract_nouns]

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
@app.route('/predict', methods=['POST'])
def division_prediction():
    preprocessed = json.loads(request.get_data(), encoding='utf-8')["token"]
    
    if len(preprocessed) != 0:
        predict_division = predict(preprocessed)
        
        result = {"division": str(predict_division[0]), "prob": str(predict_division[1])}
        return json.dumps(result, ensure_ascii=False)
    else:
        return "Input symptoms"

@app.route('/tokenize', methods=['POST'])
def text_token():
    symptoms = json.loads(request.get_data(), encoding='utf-8')["text"]
    
    token_result = tokenize(symptoms)
    doc = {"tok_symptoms":" ".join(token_result[0]), "nouns":token_result[1]}
    return json.dumps(doc, ensure_ascii=False)

if __name__ == '__main__':
   app.run('0.0.0.0',port=5000,debug=True)
