import numpy as np
import pandas as pd
import operator
import nltk
#nltk.download('wordnet')
import matplotlib.pyplot as plt
from sklearn import model_selection
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def ReadTrainingSet():
    dftrain = pd.read_csv("C:/KaggleForum/DonorsOrg/train/train.csv")
    dftrain.index = dftrain.id
    return dftrain
def ReadTestSet():
    dftest =  pd.read_csv("C:/KaggleForum/DonorsOrg/test/test.csv")
    dftest["project_is_approved"] = np.NaN
    dftest.index = dftest.id
    return dftest
def ReadResourceSet():
    dfresource =  pd.read_csv("C:/KaggleForum/DonorsOrg/resources/resources.csv")
    dfresource = dfresource.drop(["description"], axis=1)
    dfresource.index = dfresource.id
    return dfresource

def GetProjResourceTextVector(df_input):
    lmtzr = WordNetLemmatizer()
    TextFeature = df_input.project_resource_summary
    #1] Make lover case
    TextFeature = TextFeature.map(lambda x: x.lower())

    #2] Replace "my students need"
    TextFeature = TextFeature.map(lambda x: x.replace("my students need", ""))
    TextFeature = TextFeature.map(lambda x: x.replace("balls", "ball"))
    TextFeature = TextFeature.map(lambda x: x.replace("fires", "fire"))
    TextFeature = TextFeature.map(lambda x: x.replace("minis", "mini"))
    TextFeature = TextFeature.map(lambda x: x.replace("sets", "set"))
    
    #3] retrieve first three words
    TextFeature = TextFeature.map(lambda x: ' '.join(x.split()[:3]))
    
    #4] Remove digits
    TextFeature = TextFeature.map(lambda x: ''.join("%d" if c.isdigit() else c for c in x))
    
    #5] Lemmatize data
    TextFeature = TextFeature.map(lambda x: lmtzr.lemmatize(x))
    vectorizer = CountVectorizer(max_features=100,ngram_range=(2,2), stop_words=["a", "an", "the", "to", "of", "and", "for", "that", "with", "in", "order", "these", "items", "will", "on", "such", "as", "so", "that", "they", "be", "able", "stand", "up", "are", "use", "leveled" "read", "their", "own", "more", "at", "two", "supplies", "like","wide", "variety", "about", "reading", "high", "interest", "read", "help", "our", "basic", "school", "high", "quality", "dd", "have", "access","three", "non", "fiction","engaging", "new" "four", "five", "six", "seven","some", "new"])
    features = vectorizer.fit_transform(TextFeature)
    df_feature_projres = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names(), index=df_input.id)
    return df_feature_projres

    
# Read Data
df_train = ReadTrainingSet()
df_test = ReadTestSet()
df_resource = ReadResourceSet()


# Combine Train and Test set so that we can do combine feature engg
frames = [df_train, df_test]
df_complete = pd.concat(frames, axis=0)
df_projres = GetProjResourceTextVector(df_complete)

df_complete.columns

df_work = df_complete[["id", "teacher_prefix", "school_state", "project_submitted_datetime", "project_grade_category", "teacher_number_of_previously_posted_projects", "project_is_approved"]]
df_work["odds"] = C.odds
df_work.to_csv("C:/KaggleForum/DonorsOrg/work.csv")

print(df_work.columns)
C.to_csv("C:/KaggleForum/DonorsOrg/LogOdds.csv)
frames = [df_complete, C]
pd.concat(frames, axis = 1)
# we need to combine Resource df with proj resource feature vector
# We cna perform clasification on resource set to kno what kind
# of projects are getting approved based on the resource asks

# Add total price as a calculated column
# Total Price = qualtity * price

df_resource["totprice"] = df_resource.quantity * df_resource.price

# Calcualte minimum values
minval = df_resource.groupby("id").min()
minval.rename(columns={minval.columns[0]:"min qty", minval.columns[1]:"min price", minval.columns[2]:"min tot price"}, inplace=True)
# Calculate maximum values
maxval = df_resource.groupby("id").max()
maxval.rename(columns={maxval.columns[0]:"max qty", maxval.columns[1]:"max price", maxval.columns[2]:"max tot price"}, inplace=True)
# Calculate mean values
meanval = df_resource.groupby("id").mean()
meanval.rename(columns={meanval.columns[0]:"mean qty", meanval.columns[1]:"mean price", meanval.columns[2]:"mean tot price"}, inplace=True)
# Calculate total values
sumval = df_resource.groupby("id").sum()
sumval["totprice"] = sumval.quantity*sumval.price
sumval.rename(columns={sumval.columns[0]:"sum qty", sumval.columns[1]:"sum price", sumval.columns[2]:"sum tot price"}, inplace=True)
# combine these data frames

df_target = df_complete.project_is_approved
df_target.index = df_complete.id

frames = [df_projres, minval, maxval, meanval, sumval, df_target]
frames = [sumval, df_target]
sumval = pd.concat(frames, axis=1)
print(df_projres_Complete.shape)

# Save the created dataframe so that
# we can re-use it later

#df_projres_Complete.to_csv("C:/KaggleForum/DonorsOrg/df_projres_Complete.csv")

df_projres_Complete = pd.read_csv("C:/KaggleForum/DonorsOrg/df_projres_Complete.csv")

# Separate Train and Test Data
df_projres = sumval[sumval.project_is_approved.isnull() == False]
df_projresval = sumval[sumval.project_is_approved.isnull()]

# Evaluate using different models


def EvaluateModels(X, Y):
    seed = 7
    # prepare models
    models = []
    models.append(('LR', linear_model.LogisticRegression(C=0.8, penalty='l1', tol=1e-6)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    #models.append(('SVM', SVC()))
    #evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
 
	
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

X = df_projres[df_projres.columns[0:-1]]
Y = df_projres[df_projres.columns[-1]]

Xval = df_projresval[df_projres.columns[0:-1]]
Yval = df_projresval[df_projres.columns[-1]]

df_projres.to_csv("C:\KaggleForum/DonorsOrg/df_projres.csv")
seed = 10
scoring = 'accuracy'
for c in np.arange(0.1, 1, 0.1):
    model = linear_model.LogisticRegression(C=c, penalty='l1', tol=1e-6)    
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    msg = "%s: %f (%f)" % ('LR', cv_results.mean(), cv_results.std())
    print(msg)
# Scale the data to be between -1 and 1
scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(data = scaler.transform(X), columns=["qualtity", "price", "tot price"])

from scipy import stats

pricebx = np.asarray(X[['price']].values)
dftprice = stats.boxcox(pricebx)[0]

quantitybx = np.asarray(X[['quantity']].values)
dftquantity = stats.boxcox(quantitybx)[0]
totpricebx = np.asarray(X[['totprice']].values)
dfttotprice = stats.boxcox(totpricebx)[0]

X["price"] = dftprice
X["quantity"] = dftquantity
X["totprice"] = dfttotprice

pricebx = np.asarray(Xval[['price']].values)
dftprice = stats.boxcox(pricebx)[0]

quantitybx = np.asarray(Xval[['quantity']].values)
dftquantity = stats.boxcox(quantitybx)[0]
totpricebx = np.asarray(Xval[['totprice']].values)
dfttotprice = stats.boxcox(totpricebx)[0]

Xval["price"] = dftprice
Xval["quantity"] = dftquantity
Xval["totprice"] = dfttotprice

model  = linear_model.LogisticRegression()#C=0.8, penalty='l1', tol=1e-6)
fit = model.fit(X,Y)
pred = fit.predict_proba(df_workval)
predlg = pred[:,1]/pred[:,0]
Xval["odds"] = predlg
df_workval["project_is_approved"] = pred[:,1]
pred = fit.predict_proba(X)
predlg = pred[:,1]/pred[:,0]
df_workval["odds"] = predlg

Xval.to_csv("C:/KaggleForum/DonorsOrg/firstset.csv")
frames = [X,Xval]
C = pd.concat(frames, axis=0)
dd = pricebx[pricebx<=0]
X.loc[:,"price"]=   dft
X1 = X.copy()
X1.columns
X1 = X1.drop(['Unnamed: 0'], axis=1)
X1 = X[["min qty", "min price", "min tot price", "max qty", "max price", "max tot price", "mean qty", "mean price", "mean tot price", "sum qty", "sum price", "sum tot price"] ]

X2 = (X2-X2.mean())/X2.std()
EvaluateModels(X,Y)

# check the random forest model

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)



model = RandomForestRegressor(n_jobs=-1)
estimators = np.arange(10, 200, 10)
scores = []
for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(X_train, Y_train)
    mscore = model.score(X_test, Y_test)
    print(mscore)
    scores.append(mscore)
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)

 plt.hist(X['totprice'], bins=100)
 plt.show()
 
 transform = np.asarray(X[['totprice']].values)
    
 # transform values and store as "dft"
 dft = stats.boxcox(transform)[0]
 # plot the transformed data
 plt.hist(dft, bins=100)
 plt.show()
 
 #----------------------------------------------------------------------------
 # Feature Engineering for final set
 #----------------------------------------------------------------------------
 
 df_work = pd.read_csv("C:/KaggleForum/DonorsOrg/work.csv")
 
 # drop id.1 column
 df_work = df_work.drop("id.1", axis = 1)

# Get mode of teacher_prefix column and replace that with missing values
 
df_work['teacher_prefix'].fillna(df_work['teacher_prefix'].mode()[0], inplace=True)

import datetime as dt

df_work['project_submitted_datetime'] = pd.to_datetime(df_work['project_submitted_datetime'])
df_work['quarter'] = df_work.project_submitted_datetime.dt.quarter
df_work['month'] = df_work.project_submitted_datetime.dt.month
df_work['year'] = df_work.project_submitted_datetime.dt.year

# Replace project_grade_category varaible with respective mean

df_work['project_grade_category'].replace(['Grades PreK-2'] , np.mean(list(range(0,3))),inplace=True)
df_work['project_grade_category'].replace(['Grades 3-5'] , np.mean(list(range(3,6))),inplace=True)
df_work['project_grade_category'].replace(['Grades 6-8'] , np.mean(list(range(6,9))),inplace=True)
df_work['project_grade_category'].replace(['Grades 9-12'] , np.mean(list(range(9,13))),inplace=True)

df_work.project_grade_category.unique()

# drop project_submitted_datetime column
df_work = df_work.drop('project_submitted_datetime', axis = 1)

# Get one hot encoding for teacher_prefix columns

teacherpref_onehot = pd.get_dummies(df_work['teacher_prefix'])
df_work = df_work.drop('teacher_prefix', axis=1)
df_work = df_work.join(teacherpref_onehot)

# Get one hot encoding for school_state columns

schoolstate_onehot = pd.get_dummies(df_work['school_state'])
df_work = df_work.drop('school_state', axis=1)
df_work = df_work.join(schoolstate_onehot)

# separate dataframe into training and validation set

df_work.to_csv('C:/KaggleForum/DonorsOrg/finalwork.csv')
df_worktrain = df_work[df_work.project_is_approved.isnull() != True]
df_workval = df_work[df_work.project_is_approved.isnull()]
df_workval = df_workval.drop('id', axis=1)

df_id = df_work[df_work.project_is_approved.isnull()].id
df_prob = df_workval['project_is_approved'] 
frame = [df_id, df_prob]
df_sub = pd.concat(frame, axis=1)
X = df_worktrain.drop(['id','project_is_approved'], axis=1)
Y = df_worktrain.project_is_approved

df_sub.to_csv('C:/KaggleForum/DonorsOrg/set2.csv')