#The dataset belongs to a competition on Kaggle 'Natural Language Processing with Disaster Tweets'
#Goal: to predict if the tweets are about real disasters or not
#Metrics for evaluation: f1_score,precision and recall 

import pandas as pd
from collections import Counter 
import matplotlib.pyplot as plt 
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.svm import SVC   #since our data is will be hing dimensional 


train_path = "data/train.csv"
train_df = pd.read_csv(train_path)

print(train_df.head())
print(train_df.columns)
print(train_df.isna().sum())
train_df.info()

#target distribution 
counts = Counter(train_df['target'])
print(counts)

plt.bar(counts.keys(),counts.values(),width=0.3,color='blue')
plt.xlabel('Class')
plt.ylabel('Counts')
plt.show() #0: 4342, 1: 3271; The graph shows class imbalance; terfore accuracy alone is misleading as a metric

#unique values in keyword:
counts = Counter(train_df['keyword'])
print(len(counts)) #222 unique keywords and 61 missing

#not dropping missing values rather filling them with 'missing', just in case they carry some cementic meaning 
train_df['keyword'] = train_df['keyword'].fillna('missing')
# #adding the keyword to the text to make a combined column:
# train_df['text_combined'] = train_df['keyword'] + ' ' + train_df['text'] #boosts signal/ good with tf-idf


'''During reviewing of data i observed that the test consisted of links,special_characters, @, numbers
these introduce high-cardinality noise so removing them will be a good idea'''

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+','',text)   #removing url's/replacing them with ''
    text = re.sub(r'#','',text)   #removing hashtags/ replacing them with ''
    text = re.sub(r'@\w+','',text)  #removing mentions/replacing them with ''
    text = re.sub(r'\d+','',text)  #removing numbers/ replacing them with ''
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)  #removing special characters

    return text


train_df['clean_text'] = train_df['text'].apply(clean_text)
print(train_df[['text','clean_text']].head())

#seperating the data required for baseline model('clean_text') and the target:
X = train_df['clean_text']
y = train_df['target']


#wordcloud to see which words occur most frequently
text = " ".join(X)
wc = WordCloud(height=400,width=800,random_state=42,colormap='plasma').generate(text)
plt.figure(figsize=(10,5))
plt.imshow(wc)
plt.axis('off')
plt.show() #except some random words, words like fire,wepon,burning etc seem to have appeared often


#---------------
'''Baseline Models'''
#---------------
#trying both Naivebayes as well as logistic regression to see which performs better as baseline to select one for hypertuning
pipeline_NB = Pipeline([
    ('Tfidf',TfidfVectorizer()),
    ('model',MultinomialNB())
])

pipeline_LR = Pipeline([
    ('Tfidf',TfidfVectorizer()),
    ('model',LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
        ))
])

pipeline_svm = Pipeline([
    ('Tfidf',TfidfVectorizer()),
    ('model',SVC())
])

#splitting data into train/validation split to evaluate baseline model
X_train,X_val,y_train,y_val = train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)

#training and evaluatin on validation set 
pipeline_NB.fit(X_train,y_train)
pipeline_LR.fit(X_train,y_train)
pipeline_svm.fit(X_train,y_train)

y_pred_val_NB = pipeline_NB.predict(X_val)
y_pred_val_LR = pipeline_LR.predict(X_val)
y_pred_val_svm = pipeline_svm.predict(X_val)

print(f'val_f1_score,precision,recall of NB baseline: ',f1_score(y_val,y_pred_val_NB),
                                                         precision_score(y_val,y_pred_val_NB),
                                                         recall_score(y_val,y_pred_val_NB))
print(f'val_f1_score,precision,recall of LR baseline: ',f1_score(y_val,y_pred_val_LR),
                                                         precision_score(y_val,y_pred_val_LR),
                                                         recall_score(y_val,y_pred_val_LR))
print(f'val_f1_score,precision,recall of SVM baseline: ',f1_score(y_val,y_pred_val_svm),
                                                         precision_score(y_val,y_pred_val_svm),
                                                         recall_score(y_val,y_pred_val_svm))


'''Baseline_results:
val_f1_score,precision,recall of NB baseline:  0.7591623036649214 0.8841463414634146 0.6651376146788991
val_f1_score,precision,recall of LR baseline:  0.7671913835956918 0.8372513562386981 0.7079510703363915
val_f1_score,precision,recall of SVM baseline:  0.764406779661017 0.8574144486692015 0.6896024464831805
'''
'''Both Multinomial Naive Bayes and Logistic Regression were evaluated as baseline models using TF-IDF features.
Logistic Regression achieved a higher F1-score and significantly better recall, making it more suitable for disaster detection 
where false negatives are costly.But at the same time SVM showed signs of performing well on the dataset with a slightly lower f1 but better precision.'''

#--------------
'''Submission 1 Baseline'''
#--------------
test_path = "data/test.csv"
test_df = pd.read_csv(test_path)

test_df['keyword'] = test_df['keyword'].fillna('missing')
test_df['clean_text'] = test_df['text'].apply(clean_text)

X_test = test_df['clean_text']

pipeline_LR.fit(X,y)
y_pred = pipeline_LR.predict(X_test)

submission = pd.DataFrame({
    'id' : test_df['id'],
    'target' : y_pred
})

submission.to_csv('submission1_baselineLR.csv',index=False) #0.79895, baseline LR submission ranked 374/717

#predictions using baseline SVM
pipeline_svm.fit(X,y)
y_pred = pipeline_svm.predict(X_test)

submission = pd.DataFrame({
    'id' : test_df['id'],
    'target' : y_pred
})

submission.to_csv('submission1_baselineSVM.csv',index=False) #0.79987, slightly better performance shown by baseline SVM leaderboard_rank = 348/717

