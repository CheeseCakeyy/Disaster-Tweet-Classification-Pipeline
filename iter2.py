#This is iteration 2 of 'Natural Language Processing with Disaster Tweets'
#This iteration will focus on feature engineering, hypertuning of baseline model and betterment of f1_score and recall
#Going ahead with logReg and SVM

import pandas as pd 
from clean_text import clean_text
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

train_path = "data/train.csv"
train_df = pd.read_csv(train_path)

print(train_df.head())

def feature_creation(df):
    df['keyword'] = df['keyword'].fillna('missing') #handling missing keywords

    #some locations are missing so we can create a feature called 'has_loc', same for keyword
    df['has_loc'] = df['location'].notna().astype(int)
    df['has_keyword'] = (df['keyword'] != 'missing').astype(int)

    #adding the keyword to the text to make a combined column:
    # df['text_combined'] = df['keyword'] + ' ' + df['text'] #boosts signal/ good with tf-idf (Its adding noise too)
    #cleaning the text 
    df['clean_text'] = df['text'].apply(clean_text)

    # # ! count and ? count can also be features (were adding noise so gotta comment 'em out)
    # df['excl_count'] = df['text'].str.count('!')
    # df['que_count'] = df['text'].str.count(r'\?')

    #dropping unnecessary columns
    useless_cols = ['id','location','keyword','text']
    df = df.drop(columns = useless_cols)

    return df

train_df = feature_creation(train_df)

#seperating target and feature
X = train_df.drop('target',axis=1)
y = train_df['target']


#-------------------
'''Preprocessing'''
#-------------------
tf_idf_cols = 'clean_text'
remaining_cols = ['has_loc','has_keyword',] #'excl_count','que_count'

preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf',TfidfVectorizer(
            ngram_range=(1,3), #includes uni and bigrams
            stop_words='english' #removes the words like 'is,the,a' etc which carry less cementic meaning
            ),tf_idf_cols),
        ('scaler',StandardScaler(with_mean=False),remaining_cols)
    ],
    remainder='drop'
)

pipeline_LR = Pipeline([
    ('prep',preprocessor),
    ('model',LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        penalty='l2'
    ))
])

pipeline_svm = Pipeline([
    ('prep',preprocessor),
    ('model',SVC(kernel='linear',random_state=42,C=1))
])

#---------------
'''Validation'''
#---------------
#splitting data into train/validate split
X_train,X_val,y_train,y_val = train_test_split(X,y,random_state=42,stratify=y,test_size=0.2)

#training and validating 
pipeline_LR.fit(X_train,y_train)
pipeline_svm.fit(X_train,y_train)

y_pred_val_LR = pipeline_LR.predict(X_val)
y_pred_val_svm = pipeline_svm.predict(X_val)

print(f'val_f1_score,precision,recall with log_reg: ',f1_score(y_val,y_pred_val_LR),
                                                         precision_score(y_val,y_pred_val_LR),
                                                         recall_score(y_val,y_pred_val_LR),
                                                         accuracy_score(y_val,y_pred_val_LR))
print(f'val_f1_score,precision,recall with SVM: ',f1_score(y_val,y_pred_val_svm),
                                                         precision_score(y_val,y_pred_val_svm),
                                                         recall_score(y_val,y_pred_val_svm),
                                                         accuracy_score(y_val,y_pred_val_svm))


'''val_f1_score,precision,recall with log_reg:  0.75809199318569 0.8557692307692307 0.6804281345565749 0.8135259356533159
val_f1_score,precision,recall with SVM:  0.7702161729383507 0.8084033613445378 0.735474006116208 0.8115561391989494
the f1 score decreased in case of logReg but a rise was seen in f1 of SVC with improved accuracy'''

#--------------
'''Submission 1 iter2 '''
#--------------
test_path = "data/test.csv"
test_df = pd.read_csv(test_path)

X_test = feature_creation(test_df)

#logReg
pipeline_LR.fit(X,y)
y_pred = pipeline_LR.predict(X_test)

submission = pd.DataFrame({
    'id' : test_df['id'],
    'target' : y_pred
})

# submission.to_csv('submission1_LR_iter2.csv',index=False) #0.79098

#SVM
pipeline_svm.fit(X,y)
y_pred = pipeline_svm.predict(X_test)

submission = pd.DataFrame({
    'id' : test_df['id'],
    'target' : y_pred
})


submission.to_csv('submission1_svm_iter2.csv',index=False) #0.80324 best so far, kaggle rank 304/713

