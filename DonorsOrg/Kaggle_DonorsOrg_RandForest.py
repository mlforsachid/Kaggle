import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from keras.layers import Embedding
# Read data

df_train = pd.read_csv("C:/KaggleForum/DonorsOrg/train/train.csv",index_col="id", parse_dates=["project_submitted_datetime"])
df_test = pd.read_csv("C:/KaggleForum/DonorsOrg/test/test.csv", index_col="id", parse_dates=["project_submitted_datetime"])
df_resource =  pd.read_csv("C:/KaggleForum/DonorsOrg/resources/resources.csv", index_col="id").fillna("NA")

# modify dftrain to use stratified sampling to balnace
# class
df_train.sort_values(by='project_is_approved', ascending=True, inplace=True) #easier for stratified sampling

df_sample = df_train.iloc[:110936,:]

Y = df_sample['project_is_approved'].copy()
df_all = pd.concat([df_sample.drop('project_is_approved', axis=1), df_test], axis = 0)

trainidx = df_sample.index
tstidx = df_test.index
allidx = df_all.index

# Take the agg
agg_rc = df_resource.reset_index().groupby('id').agg(
    dict(quantity = 'sum',
         price = 'sum',
         description = lambda x: ' *appender* '.join(x)))
agg_rc["totvalue"] = agg_rc.quantity*agg_rc.price

# merge resource df with main df
df_all = pd.merge(df_all, agg_rc, left_index=True, right_index=True, how="left")

#del agg_rc, df_train, df_test, df_resource
#gc.collect()

# Start Feature engineering
#1] concatenate all essays into one column
df_all["projdet"] = df_all.apply(lambda x: ' '.join([
    str(x['project_essay_1']), 
    str(x['project_essay_2']), 
    str(x['project_essay_3']), 
    str(x['project_essay_4'])]), axis=1)

#2] Get quarter, year and month from project_submitted_datetime column

df_all['quarter'] = df_all.project_submitted_datetime.dt.quarter
df_all['month'] = df_all.project_submitted_datetime.dt.month
df_all['year'] = df_all.project_submitted_datetime.dt.year

#3] create dummy variables from project categories and sub categories
#4] Split the strings at the comma, and treat them as dummies
df_all = pd.merge(df_all, df_all["project_subject_categories"].str.get_dummies(sep=', '),
              left_index=True, right_index=True, how="left")
df = pd.merge(df_all, df_all["project_subject_subcategories"].str.get_dummies(sep=', '),
              left_index=True, right_index=True, how="left")

#5] Encode all the categorical variables
dumyvars= ['school_state','project_grade_category']
timevars = ['quarter','month','year']
encode = ['teacher_prefix']
categorical_features = dumyvars + timevars + encode
lbl = preprocessing.LabelEncoder()
#for col in categorical_features:
#     df_all[col] = lbl.fit_transform(df_all[col].astype(str))

dummies  = pd.get_dummies(df_all[categorical_features])

df_all = pd.concat([df_all, dummies], axis=1)
# 6] Drop un-necessary columns
text_cols = ["project_resource_summary", "project_title","description","text"]

df_all.drop(['project_subject_categories',"project_subject_subcategories","project_submitted_datetime",
         "project_essay_1","project_essay_2","project_essay_3","project_essay_4"
        ],axis=1,inplace=True)
 
#7 ] add tf idf terms
tfidf_para = {
    "sublinear_tf":True,
    "strip_accents":'unicode',
    "stop_words":"english",
    "analyzer":'word',
    "token_pattern":r'\w{1,}',
    #"ngram_range":(1,1),
    "dtype":np.float32,
    "norm":'l2',
    "min_df":5,
    "max_df":.9,
    "smooth_idf":False
}

def get_col(col_name):
    return lambda x: x[col_name]

df_all["project_title_count"] = df_all["project_title"].copy()
textcols = ["teacher_id","projdet","project_resource_summary","project_title", "project_title_count","description"]
vectorizer = FeatureUnion([
        ('projdet',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            **tfidf_para,
            preprocessor=get_col('projdet'))),
        ('project_resource_summary',TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            max_features=2000,
            preprocessor=get_col('project_resource_summary'))),
        ('project_title',TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            max_features=1500,
            preprocessor=get_col('project_title'))),
        ('project_title_count',CountVectorizer(
            ngram_range=(1, 2),
            max_features=1500,
            preprocessor=get_col('project_title_count'))),
        ('description',TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            max_features=2400,
            preprocessor=get_col('description'))),
#         ('Non_text',DictVectorizer())
    ])

ready_df = vectorizer.fit_transform(df_all.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
df_all.drop(textcols,axis=1, inplace=True)

columnnames = df_all.columns.tolist()+tfvocab
# # Sort and Print
# feature_array = np.array(vectorizer.get_feature_names())
# tfidf_sorting = np.argsort(ready_df.toarray()).flatten()[::-1]
# print("Most Important Words in All Vectorization:\n",feature_array[tfidf_sorting][:20])

#df_all_trim = df_all[['teacher_prefix', 'school_state','project_grade_category','teacher_number_of_previously_posted_projects', 'quantity', 'price','totvalue', 'quarter', 'month', 'year', 'Applied Learning', 'Care & Hunger', 'Health & Sports',       'History & Civics', 'Literacy & Language', 'Math & Science', 'Music & The Arts', 'Special Needs', 'Warmth']]

#X = hstack([csr_matrix(df_all.loc[trainidx,:].values),ready_df[0:trainidx.shape[0]]])
#XVal = hstack([csr_matrix(df_all.loc[tstidx,:].values),ready_df[trainidx.shape[0]:]])

dropcol = ['teacher_id', 'teacher_prefix', 'school_state', 'project_grade_category', 'project_title', 'project_resource_summary','description']

df_all.drop(dropcol, axis= 1, inplace=True)
X = csr_matrix(df_all.loc[trainidx,:].values)
XVal = csr_matrix(df_all.loc[tstidx,:].values)


# Read Feature Set
def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    cv_score = cross_validation.cross_val_score(alg, dtrain, predictors, cv=cv_folds, scoring='roc_auc')
    
    print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
# Try gradient boosting
clf1 = GradientBoostingClassifier( random_state=10)
modelfit(clf1, X,Y)

# Try random forest      
clf = RandomForestClassifier(random_state=10)
modelfit(clf, X,Y)

# Try logistic regression
model = LogisticRegression()
modelfit(model, X,Y)

# Try XG boost
imbalance_weight = Y.value_counts(normalize=True)[0]/Y.value_counts(normalize=True)[1]

model1 = XGBClassifier(max_depth=12, learning_rate = 0.05, subsample = 0.8, colsample_bytree = 0.75, scale_pos_weight=imbalance_weight, objective='binary:logistic', eval_metric='auc', seed=23)
modelfit(model1, X,Y)

# pick up the best model

model1.fit(X, Y)
prob = model1.predict_proba(XVal )
sub1 = pd.DataFrame(prob[:,1] ,columns=["project_is_approved"],index=tstidx)
sub1.to_csv("C:/KaggleForum/DonorsOrg/set6.csv")

#-----------------------------------------------------------------------
# Refinement using Keras and Tensorflow
#-----------------------------------------------------------------------

# Importing libraries for building the neural network
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM

def create_LSTM():
    model = Sequential()
    model.add(Embedding(output_dim=256, input_dim=27419))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    return model

def create_baseline():
    model = Sequential()
    model.add(Dense(200, input_dim=79, activation='relu', kernel_initializer='uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu', kernel_initializer='uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model

model = create_baseline() #KerasClassifier(build_fn=create_baseline, epochs=10, verbose=1, shuffle="batch")

from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced', np.unique(Y), Y)

df_input = pd.read_csv("C:/KaggleForum/DonorsOrg/finalwork.csv", index_col="id")

df_input.drop(['Unnamed: 0'], axis=1, inplace=True)

XTest = df_input[df_input.project_is_approved.isnull() != True]
XVal = df_input[df_input.project_is_approved.isnull()]

# since class is imbalance, use stratefied sampling to balance the class

XTest.sort_values(by='project_is_approved', ascending=True, inplace=True) #easier for stratified sampling

df_sample = XTest.iloc[:110936,:]
df_sample.project_is_approved.value_counts()

YTest = df_sample.project_is_approved
df_sample.drop(["project_is_approved"], axis = 1, inplace=True)
XVal.drop(["project_is_approved"], axis = 1, inplace=True)

Y.groupby(Y)

train_hist = model.fit(csr_matrix(X), Y, batch_size=500, shuffle=True,  epochs=200)
prob = model.predict_proba(csr_matrix(XVal) )
sub1 = pd.DataFrame(prob ,columns=["project_is_approved"],index=tstidx)
sub1.to_csv("C:/KaggleForum/DonorsOrg/set9.csv")