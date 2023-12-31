from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
# print(torch.__version__)

tokenizer=AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model= AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

tokens=tokenizer.encode('I am Sony Thomas!', return_tensors='pt')
print(tokens)
print(tokenizer.decode(tokens[0]))

result=model(tokens)
print(f"the result is of the form {result}\nAnd the actual result is the logits which mention the probility of the class {result.logits}")
print(int(torch.argmax(result.logits ))+1)

#collect reviews
r= requests.get('https://www.yelp.com/biz/mejico-sydney-2')
soup=BeautifulSoup(r.text,'html.parser')
regex= re.compile('.*comment.*')
results=soup.find_all('p',{'class':regex})
reviews=[result.text for result in results]

# print(f"every text in the search scraped is:\n{r.text}\n\n")
# print(reviews)

'''
Load review through a data frame and run through model
'''
df= pd.DataFrame(np.array(reviews),columns=['review'])
print(df['review'].iloc[0])

def sentiment_score(review)->int:
    tokens=tokenizer.encode(review, return_tensors='pt')
    print(tokens)
    print(tokenizer.decode(tokens[0]))

    result=model(tokens)
    print(f"the result is of the form {result}\nAnd the actual result is the logits which mention the probility of the class {result.logits}")
    print(int(torch.argmax(result.logits ))+1)
    return int(torch.argmax(result.logits)+1)
df['sentiment']=df['review'].apply(lambda x:sentiment_score(x[:512]))
print(df.head())