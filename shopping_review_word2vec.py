import re
import numpy as np
import pandas as pd
import urllib.request

from keras.utils import pad_sequences
from konlpy.tag import Mecab
from collections import Counter
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Embedding, GRU, Dense
from tensorflow.python.keras.models import Sequential, load_model


#----------- 데이터 삽입 --------------
urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt", filename="content/ratings_total.txt")

total_data = pd.read_table('content/ratings_total.txt', names=['ratings', 'reviews'])
print('전체 리뷰 개수 :',len(total_data))

total_data['label'] = np.select([total_data.ratings > 3], [1], default=0)
total_data.drop_duplicates(subset=['reviews'], inplace=True) #중복 제거

train_data, test_data = train_test_split(total_data, test_size = 0.25, random_state = 42)
print('\n전처리 전 훈련용 리뷰의 개수 :', len(train_data))
print('전처리 전 테스트용 리뷰의 개수 :', len(test_data))


#----------- 전처리 --------------
train_data.drop_duplicates(subset = ['reviews'], inplace=True) #중복 제거
train_data['reviews'] = train_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True) #정규 표현식 수행
train_data['reviews'].replace('', np.nan, inplace=True) #공백은 Null 값으로 변경
train_data = train_data.dropna(how='any') # Null 값 제거

test_data.drop_duplicates(subset = ['reviews'], inplace=True) #중복 제거
test_data['reviews'] = test_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True) #정규 표현식 수행
test_data['reviews'].replace('', np.nan, inplace=True) #공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거

print('\n전처리 후 훈련용 샘플의 개수 :',len(train_data))
print('전처리 후 테스트용 샘플의 개수 :',len(test_data))


#----------- 토큰화 --------------
mecab = Mecab('C:\mecab\mecab-ko-dic')

stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게', '네요', '는데', '어요']
train_data['tokenized'] = train_data['reviews'].apply(mecab.morphs)
train_data['tokenized'] = train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
test_data['tokenized'] = test_data['reviews'].apply(mecab.morphs)
test_data['tokenized'] = test_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

negative_words = np.hstack(train_data[train_data.label == 0]['tokenized'].values)
positive_words = np.hstack(train_data[train_data.label == 1]['tokenized'].values)
positive_word_count = Counter(positive_words)
negative_word_count = Counter(negative_words)

text_len = train_data[train_data['label']==1]['tokenized'].map(lambda x: len(x))
print('\n긍정 리뷰의 평균 길이 :', np.mean(text_len))
text_len = train_data[train_data['label']==0]['tokenized'].map(lambda x: len(x))
print('부정 리뷰의 평균 길이 :', np.mean(text_len))

X_train = train_data['tokenized'].values
y_train = train_data['label'].values
X_test= test_data['tokenized'].values
y_test = test_data['label'].values


#----------- 정수 인코딩 --------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

threshold = 2
total_cnt = len(tokenizer.word_index)
rare_cnt = 0
total_freq = 0
rare_freq = 0

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('\n단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

vocab_size = total_cnt - rare_cnt + 2
tokenizer = Tokenizer(vocab_size, oov_token = 'OOV')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


#----------- 패딩 --------------
def below_threshold_len(max_len, nested_list):
  count = 0
  for sentence in nested_list:
    if(len(sentence) <= max_len):
        count = count + 1
  print('\n전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))

max_len = 80
below_threshold_len(max_len, X_train)
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)


#----------- 모델링 --------------
embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(GRU(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('model/shopping_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)

loaded_model = load_model('model/shopping_model.h5')
print("\n테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))


#----------- 리뷰 예측 --------------
def sentiment_predict(new_sentence):
  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = mecab.morphs(new_sentence)
  new_sentence = [word for word in new_sentence if not word in stopwords]
  encoded = tokenizer.texts_to_sequences([new_sentence])
  pad_new = pad_sequences(encoded, maxlen = max_len)

  score = float(loaded_model.predict(pad_new))
  if(score > 0.5):
    print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
  else:
    print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))

sentiment_predict('이 상품 진짜 좋아요... 저는 강추합니다. 대박')
sentiment_predict('진짜 배송도 늦고 개짜증나네요. 뭐 이런 걸 상품이라고 만듬?')
sentiment_predict('판매자님... 너무 짱이에요.. 대박나삼')
sentiment_predict('ㅁㄴㅇㄻㄴㅇㄻㄴㅇ리뷰쓰기도 귀찮아')