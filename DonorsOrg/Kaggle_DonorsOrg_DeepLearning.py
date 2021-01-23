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
from sklearn.preprocessing import StandardScaler
# Read data
df_train = pd.read_csv("C:/KaggleForum/DonorsOrg/train/train.csv",index_col="id", parse_dates=["project_submitted_datetime"])
df_test = pd.read_csv("C:/KaggleForum/DonorsOrg/test/test.csv", index_col="id", parse_dates=["project_submitted_datetime"])
df_resource =  pd.read_csv("C:/KaggleForum/DonorsOrg/resources/resources.csv", index_col="id").fillna("NA")

Y = df_train['project_is_approved'].copy()
df_all = pd.concat([df_train.drop('project_is_approved', axis=1), df_test], axis = 0)

trainidx = df_train.index
tstidx = df_test.index
allidx = df_all.index

#1]Calculate Total Price and 
# add features like min, max, mean, count and sum of total price
df_resource["totprice"] = df_resource.quantity * df_resource.price

min_total_price = pd.DataFrame(df_resource.groupby('id').totprice.min())
min_total_price.rename(columns={'totprice':'mintotprice'}, inplace=True)

max_total_price = pd.DataFrame(df_resource.groupby('id').totprice.max())
max_total_price.rename(columns={'totprice':'maxtotprice'}, inplace=True)

mean_total_price = pd.DataFrame(df_resource.groupby('id').totprice.mean())
mean_total_price.rename(columns={'totprice':'meantotprice'}, inplace=True)

count_total_price = pd.DataFrame(df_resource.groupby('id').totprice.count())
count_total_price.rename(columns={'totprice':'counttotprice'}, inplace=True)

sum_total_price = pd.DataFrame(df_resource.groupby('id').totprice.sum())
sum_total_price.rename(columns={'totprice':'sumtotprice'}, inplace=True)

df_all = pd.concat([df_all, min_total_price, max_total_price, mean_total_price, count_total_price, sum_total_price],axis=1)

#2] Get quarter, year and month from project_submitted_datetime column

df_all['quarter'] = df_all.project_submitted_datetime.dt.quarter
df_all['month'] = df_all.project_submitted_datetime.dt.month
df_all['year'] = df_all.project_submitted_datetime.dt.year

df_all.drop('project_submitted_datetime', axis=1, inplace=True)
#3] Combine all text feature into single column and drop redundent feature
char_cols = ['project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4', 'project_resource_summary']

df_all["text"] = df_all.apply(lambda x: ' '.join([
    str(x['project_title']), 
    str(x['project_essay_1']), 
    str(x['project_essay_2']), 
    str(x['project_essay_3']), 
    str(x['project_essay_4']),
    str(x['project_resource_summary'])]), axis=1)

df_all.drop(char_cols, axis=1, inplace=True)

#4] Hash categorical feature
cat_features = ["teacher_prefix", "school_state", "year", "quarter", "month", "project_grade_category", "project_subject_categories", "project_subject_subcategories"]
cat_features_hash = [col+"_hash" for col in cat_features]

max_size=15000#0
def feature_hash(df, max_size=max_size):
    for col in cat_features:
        df[col+"_hash"] = df[col].apply(lambda x: hash(x)%max_size)
    df.drop(cat_features, axis=1, inplace=True)
    return df

feature_hash(df_all)

#5] Standardize numerical features
num_features = ['teacher_number_of_previously_posted_projects', 'mintotprice', 'maxtotprice', 'meantotprice', 'counttotprice', 'sumtotprice']
scaler = StandardScaler()
df_all[num_features] = scaler.fit_transform(df_all[num_features])

 
#6 ] Add word embeddings for Text column
from keras.preprocessing import text, sequence
import re
def preprocess(string):
    string = re.sub(r'(\")', ' ', string)
    string = re.sub(r'(\r)', ' ', string)
    string = re.sub(r'(\n)', ' ', string)
    string = re.sub(r'(\r\n)', ' ', string)
    string = re.sub(r'(\\)', ' ', string)
    string = re.sub(r'\t', ' ', string)
    string = re.sub(r'\:', ' ', string)
    string = re.sub(r'\"\"\"\"', ' ', string)
    string = re.sub(r'_', ' ', string)
    string = re.sub(r'\+', ' ', string)
    string = re.sub(r'\=', ' ', string)
    return string

df_all["text"]=df_all["text"].apply(preprocess)

max_features = 100000#50000
maxlen = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df_all["text"].tolist())
list_tokenized = tokenizer.texts_to_sequences(df_all["text"].tolist())
padtoken = pd.DataFrame(sequence.pad_sequences(list_tokenized, maxlen=maxlen), index = allidx)
padlist = sequence.pad_sequences(list_tokenized, maxlen=maxlen)
df_all = pd.concat([df_all, padtoken], axis=1)
# Drop text and teacher_id column

df_all.drop(['teacher_id', 'text'],axis=1, inplace=True)

df_train = df_all.loc[trainidx,:]
df_test = df_all.loc[tstidx,:]

X_train_num = np.array(df_train[num_features], dtype=np.int)
X_test_num = np.array(df_test[num_features], dtype = np.int)
X_train_cat = np.array(df_train[cat_features_hash], dtype=np.int)
X_test_cat = np.array(df_test[cat_features_hash], dtype=np.int)
X_train_words = padtoken[0:182080]
X_test_words = padtoken[182080:]

#] Leverage standard embedding vector
# This is for determining weights

EMBEDDING_FILE = 'C:/KaggleForum/DonorsOrg/crawl-300d-2M/crawl-300d-2M.vec'
embed_size=300
embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

word_index = tokenizer.word_index
npadlist = np.zeros(padlist.shape)
for j, l in enumerate(padlist):
    nl = np.zeros((300, 300))
    for i, wordidx in enumerate(l):
        if wordidx !=0:
            word = list(word_index.keys())[list(word_index.values()).index(wordidx)]
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                nl[i] = embedding_vector
    npadlist[j] = nl.mean()
#prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

from keras.layers import Input, Dense, Embedding, Flatten, concatenate, Dropout, Convolution1D, \
GlobalMaxPool1D,SpatialDropout1D,CuDNNGRU,Bidirectional,PReLU,GRU
from keras.models import Model
from keras import optimizers

def get_model():
    input_cat = Input((len(cat_features_hash), ))
    input_num = Input((len(num_features), ))
    input_words = Input((maxlen, ))
    
    x_cat = Embedding(max_size, 10)(input_cat)
    
    x_cat = SpatialDropout1D(0.3)(x_cat)
    x_cat = Flatten()(x_cat)
    
    x_words = Embedding(max_features, 300,
                            weights=[embedding_matrix],
                            trainable=False)(input_words)
    x_words = SpatialDropout1D(0.3)(x_words)
    x_words =Bidirectional(GRU(50, return_sequences=True))(x_words)
    x_words = Convolution1D(100, 3, activation="relu")(x_words)
    x_words = GlobalMaxPool1D()(x_words)

    
    x_cat = Dense(100, activation="relu")(x_cat)
    x_num = Dense(100, activation="relu")(input_num)

    x = concatenate([x_cat, x_num, x_words])

    x = Dense(50, activation="relu")(x)
    x = Dropout(0.25)(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[input_cat, input_num, input_words], outputs=predictions)
    model.compile(optimizer=optimizers.Adam(0.0005, decay=1e-6),
              loss='binary_crossentropy',
              metrics=['accuracy'])

    return model

model = get_model()
history = model.fit([X_train_cat, X_train_num, X_train_words], Y, validation_split=0.1,   verbose=2, epochs=5, batch_size=256)
prob = model.predict_proba([X_test_cat, X_test_num, X_test_words], batch_size=2000)
sub1 = pd.DataFrame(prob ,columns=["project_is_approved"],index=tstidx)
sub1.to_csv("C:/KaggleForum/DonorsOrg/set9.csv")


# try XGBoost
imbalance_weight = Y.value_counts(normalize=True)[0]/Y.value_counts(normalize=True)[1]
model1 = XGBClassifier(max_depth=12, learning_rate = 0.05, subsample = 0.8, colsample_bytree = 0.75, scale_pos_weight=imbalance_weight, objective='binary:logistic', eval_metric='auc', seed=23)

# pick up the best model

model1.fit(df_train, Y)
prob = model1.predict_proba(df_test )
sub1 = pd.DataFrame(prob[:,1] ,columns=["project_is_approved"],index=tstidx)
sub1.to_csv("C:/KaggleForum/DonorsOrg/set11.csv")