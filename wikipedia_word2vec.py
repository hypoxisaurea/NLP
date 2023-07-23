from tqdm import tqdm
from konlpy.tag import Mecab
from gensim.models import Word2Vec

mecab = Mecab('C:\mecab\mecab-ko-dic')

f = open('content/wiki.txt.', encoding="utf8")
lines = f.read().splitlines()
print(len(lines))

result = []

for line in tqdm(lines):
  if line:
    result.append(mecab.morphs(line))

len(result)

model = Word2Vec(result, window=5, min_count=5, workers=4, sg=0)
model_result1 = model.wv.most_similar("대한민국")
print(model_result1)