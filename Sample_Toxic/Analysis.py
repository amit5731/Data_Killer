import csv
from textblob import TextBlob
import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.corpus import stopwords
import string
import nltk 
import textmining
import matplotlib.pyplot as plt
from wordcloud import WordCloud


os.chdir("S:/New folder/train.csv")
df=pd.read_csv("train.csv")
df_ts.head(2)
# Extract stop words
stop = set(stopwords.words("english"))

# Remove punctuation marks
exclude = set(string.punctuation)
for j in  range(0,df_ts.shape[0]):
    post_trial=df_ts.iloc[j,1]
    #print(post_trial)
    for k in exclude:
        post_trial=post_trial.replace(k,'')
    #print(post_trial)
        df_ts.iloc[j,1]=post_trial
# Text pre processing
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    #punc_free = " ".join([ch for ch in stop_free.lower().split() if ch not in exclude])
    num_free = ''.join(i for i in stop_free if not i.isdigit())
    d=["et","sh","pele","wikipedia","article",'pag','am', 'im' ,'av', 'wa', 'aa',  'ing', 'on',  'don', 'op', 'di', 'at', 'gt', 'it', 'lik','dont','page','get', 'like','second', 'life','white', 'holocaust',  'research', 'mum', 'full', 'bias', 'spreading', 'wide', 'topics', 'scjessey','dvd', 'mentions', 'guide', 'mike', 'knob', 'ignores','example','spread', 'false', 'information', 'person', 'ignorant', 'sources', 'let', 'arses', 'user', 'makes','make', 'first', 'last', 'warning','jews', 'shave', 'head', 'bald', 'go', 'meetings', 'doubt', 'words', 'bible','uh', 'two', 'become', 'done', 'didnt', 'take', 'long', 'nope', 'certainly', 'aware', 'sitting', 'front',]
    for i in d:
        num_free=num_free.replace(i,'') 
        
    return num_free
post_corpus = [clean(df_ts.iloc[i,1]) for i in range(0, df_ts.shape[0])]
df_trial=pd.DataFrame({'Comm':post_corpus})
# Create document term matrix
tdm_severetoxic = textmining.TermDocumentMatrix()
for i in post_corpus:
    tdm_severetoxic.add_doc(i)
df_tri=pd.DataFrame({'Comm':post_corpus})
#Plot wordcloud
wordcloud = WordCloud(width = 1000, hieght = 500, stopwords = STOPWORDS, background_color = 'white').generate(
                        ''.join(df_trial['Comm']))

plt.figure(figsize = (15,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


import os
os.chdir("S:/New folder/train.csv")
List=open("toxics.txt").readlines()
l=[]
for i in List:
    l.append(i.lower().split(","))

# Create engine
FinalResults_Vader = pd.DataFrame()
analyzer = SentimentIntensityAnalyzer()
h=int(input("Enter the number of comment u want to check"))
k=[]
for i in range(0,h):
    a=df.iloc[i,1]
    for k in exclude:
        a=a.replace(k,'')
    x=''.join(a)
    temp=0
    snt = analyzer.polarity_scores(a)
    b=list(snt.items())[3][1]
    for j in l[0]:
        for i in x.split():
            #print("x is ",i)
            if(i.lower()==j):
                analyzer.polarity_scores(j)
                temp=temp+list(snt.items())[3][1]
                #print("Temp in the loop")
                b=0
    temp=b+temp
    #print("Temp outside the loop",temp)
    if(temp>0.8):
        print("This is good comment.",temp)
    
    elif(temp>=-1 or temp<0.8):   
       # c = pd.DataFrame({'Toxic': temp,'treat':(temp*0.73)/100,'severe_toxic':(temp*.27)/100,'obsence':(temp*11.52)/100,'insult':(temp*7.91)/100,'hate':(0.89*temp)/100}, index = [0])  
        dict_test={'Toxic': temp,'treat':(temp*0.73)/100,'severe_toxic':(temp*.27)/100,'obsence':(temp*11.52)/100,'insult':(temp*7.91)/100,'hate':(0.89*temp)/100}
        c = pd.DataFrame.from_dict(list(dict_test.items())) 
        pd.DataFrame.from_dict(dict_test, orient = 'index')
        s = pd.Series(dict_test, name = a)
        dl = pd.DataFrame(s)
        new=dl.T
        FinalResults_Vader = FinalResults_Vader.append(new)
FinalResults_Vader.to_csv("My Final value.csv")
print("Negative toxity is saved in My Final value.csv")
    
    

