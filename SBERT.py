import pandas as pd
import urllib.request

from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="content/ChatBotData.csv")
train_data = pd.read_csv('content/ChatBotData.csv')
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
train_data['embedding'] = train_data.apply(lambda row: model.encode(row.Q), axis = 1)

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def return_answer(question):
    embedding = model.encode(question)
    train_data['score'] = train_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
    return train_data.loc[train_data['score'].idxmax()]['A']

return_answer('나랑 커피 먹을래?')