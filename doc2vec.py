import pandas as pd

from tqdm import tqdm
from konlpy.tag import Mecab
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument

df = pd.read_csv('/content/dart.csv',  sep=',')
df = df.dropna()

mecab = Mecab()
tagged_corpus_list = []

for index, row in tqdm(df.iterrows(), total=len(df)):
  text = row['business']
  tag = row['name']
  tagged_corpus_list.append(TaggedDocument(tags=[tag], words=mecab.morphs(text)))

print('문서의 수 :', len(tagged_corpus_list))

model = doc2vec.Doc2Vec(vector_size=300, alpha=0.025, min_alpha=0.025, workers=8, window=8)

# Vocabulary 빌드
model.build_vocab(tagged_corpus_list)

# Doc2Vec 학습
model.train(tagged_corpus_list, total_examples=model.corpus_count, epochs=50)

# 모델 저장
model.save('/content/dart.doc2vec')

similar_doc = model.docvecs.most_similar('동화약품')
print(similar_doc)